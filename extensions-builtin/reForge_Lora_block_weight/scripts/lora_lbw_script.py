import logging
import gradio as gr
import re
import os
from modules import scripts, shared
from lbw_lora.lora_block_weight import LoraLoaderBlockWeight, load_lbw_preset

class LoraBlockWeightScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.category_filter = "All"
        self.strength_model = 1.0
        self.strength_clip = 1.0
        self.inverse = False
        self.seed = 0
        self.A = 4.0
        self.B = 1.0
        self.preset = "Preset"
        self.block_vector = "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"
        self.bypass = False
        self.lora_applied = False

    sorting_priority = 15

    def title(self):
        return "LoRA Block Weight for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        preset_list = ["Preset"]
        preset_list += load_lbw_preset("lbw-preset.txt")
        preset_list += load_lbw_preset("lbw-preset.custom.txt")
        preset_list = [name for name in preset_list if not name.startswith('@')]

        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable LoRA Block Weight",
                    value=self.enabled
                )
                bypass = gr.Checkbox(
                    label="Bypass",
                    value=self.bypass
                )

            with gr.Row():
                strength_model = gr.Slider(
                    label="Model Strength",
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.01,
                    value=self.strength_model
                )
                strength_clip = gr.Slider(
                    label="CLIP Strength",
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.01,
                    value=self.strength_clip
                )

            with gr.Row():
                inverse = gr.Checkbox(
                    label="Inverse",
                    value=self.inverse
                )
                seed = gr.Number(
                    label="Seed",
                    value=self.seed,
                    minimum=0,
                    maximum=0xffffffffffffffff,
                    step=1,  # Add step=1 to ensure integer values
                    precision=0  # Add precision=0 to ensure integer values
                )

            with gr.Row():
                A = gr.Slider(
                    label="A",
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.01,
                    value=self.A
                )
                B = gr.Slider(
                    label="B",
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.01,
                    value=self.B
                )

            preset = gr.Dropdown(
                label="Preset",
                choices=preset_list,
                value=self.preset
            )

            block_vector = gr.Textbox(
                label="Block Weight Vector",
                value=self.block_vector,
                lines=3,
                placeholder="Format: number,number,... (e.g., 1,0,0,0,1,1)"
            )

        return (enabled, bypass, strength_model, strength_clip, inverse, 
                seed, A, B, preset, block_vector)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 10:
            (self.enabled, self.bypass, self.strength_model, self.strength_clip,
             self.inverse, seed, self.A, self.B, self.preset,
             self.block_vector) = args[:10]
            # Ensure seed is integer
            self.seed = int(seed) if isinstance(seed, (float, str)) else seed
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled or self.bypass:
            return

        if not hasattr(p.sd_model, 'forge_objects_original'):
            p.sd_model.forge_objects_original = p.sd_model.forge_objects.shallow_copy()

        try:
            # First let normal LoRA system handle loading
            if not self.enabled or self.bypass:
                return

            # Then look for block weight specifications
            block_weight_pattern = r'<lorabw:(.*?):(.*?)>'
            block_weight_specs = re.findall(block_weight_pattern, p.prompt)

            if block_weight_specs:
                # Get the LoRA directory
                lora_dir = shared.cmd_opts.lora_dir

                # Remove block weight specs from prompt
                for lora_name, weights in block_weight_specs:
                    p.prompt = p.prompt.replace(f"<lorabw:{lora_name}:{weights}>", "")

                # Apply specified block weights
                if hasattr(p.sd_model, 'forge_objects_original'):
                    p.sd_model.forge_objects_original = p.sd_model.forge_objects.shallow_copy()

                for lora_name, weights in block_weight_specs:
                    # Find the LoRA file
                    lora_path = None
                    possible_extensions = ['.safetensors', '.pt', '.ckpt']
                    
                    for ext in possible_extensions:
                        potential_path = os.path.join(lora_dir, lora_name + ext)
                        if os.path.exists(potential_path):
                            lora_path = potential_path
                            break
                        
                        # Check in subdirectories
                        for root, dirs, files in os.walk(lora_dir):
                            for file in files:
                                if file == lora_name + ext:
                                    lora_path = os.path.join(root, file)
                                    break
                            if lora_path:
                                break
                    
                    if not lora_path:
                        logging.warning(f"Could not find LoRA file for {lora_name}")
                        continue

                    lbw = LoraLoaderBlockWeight()
                    new_model, new_clip, _ = lbw.doit(
                        p.sd_model.forge_objects.unet,
                        p.sd_model.forge_objects.clip,
                        lora_path,  # Pass the full path
                        self.strength_model,
                        self.strength_clip,
                        self.inverse,
                        self.seed,
                        self.A,
                        self.B,
                        self.preset,
                        weights,
                        self.bypass
                    )
                    p.sd_model.forge_objects.unet = new_model
                    p.sd_model.forge_objects.clip = new_clip
                    self.lora_applied = True

                    # Add to generation parameters with specific block weights used
                    p.extra_generation_params[f"lora_block_weight_{lora_name}"] = weights

            # Update general generation parameters
            p.extra_generation_params.update({
                "lora_block_weight_enabled": self.enabled,
                "lora_block_weight_model_strength": self.strength_model,
                "lora_block_weight_clip_strength": self.strength_clip,
                "lora_block_weight_inverse": self.inverse,
                "lora_block_weight_seed": self.seed,
                "lora_block_weight_A": self.A,
                "lora_block_weight_B": self.B,
                "lora_block_weight_preset": self.preset,
                "lora_block_weight_vector": self.block_vector,
            })

        except Exception as e:
            logging.error(f"Error in LoRA Block Weight: {str(e)}")
            raise e

    def postprocess(self, p, processed, *args):
        """Restore original model after generation"""
        if self.lora_applied and hasattr(p.sd_model, 'forge_objects_original'):
            p.sd_model.forge_objects = p.sd_model.forge_objects_original
            self.lora_applied = False

        return