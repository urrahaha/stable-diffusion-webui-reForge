import logging
import gradio as gr
import re
import os
from modules import scripts, shared
from lbw_lora.lora_block_weight import LoraLoaderBlockWeight, load_lbw_preset
import ldm_patched.modules.utils

class LoraBlockWeightScript(scripts.Script):
    def __init__(self):
        self.enabled = False
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
        # Cache for loaded LoRAs
        self.lora_cache = {}

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
                    value=self.inverse,
                    info="Apply the following weights for each block: True: 1 - weight, False: weight"
                )
                seed = gr.Number(
                    label="Seed",
                    value=self.seed,
                    minimum=0,
                    maximum=0xffffffffffffffff,
                    step=1,
                    precision=0
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
                placeholder="block weight vectors"
            )

        return (enabled, bypass, strength_model, strength_clip, inverse,
                seed, A, B, preset, block_vector)

    def get_cached_lora(self, lora_path):
        """Get LoRA from cache or load it if not cached"""
        if lora_path not in self.lora_cache:
            try:
                lora_sd = ldm_patched.modules.utils.load_torch_file(lora_path, safe_load=True)
                self.lora_cache[lora_path] = lora_sd
                # Keep cache size reasonable - remove oldest if more than 5 LoRAs cached
                if len(self.lora_cache) > 5:
                    oldest_key = next(iter(self.lora_cache))
                    del self.lora_cache[oldest_key]
                logging.info(f"Cached LoRA: {os.path.basename(lora_path)}")
            except Exception as e:
                logging.error(f"Error loading LoRA {lora_path}: {str(e)}")
                return None
        return self.lora_cache[lora_path]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 10:
            (self.enabled, self.bypass, self.strength_model, self.strength_clip,
            self.inverse, seed, self.A, self.B, self.preset,
            self.block_vector) = args[:10]
            self.seed = int(seed) if isinstance(seed, (float, str)) else seed
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled or self.bypass:
            return

        try:
            # First let normal LoRA system handle loading
            block_weight_pattern = r'<lorabw:(.*?):(.*?)>'
            block_weight_specs = re.findall(block_weight_pattern, p.prompt)

            if block_weight_specs:
                # Remove block weight specs from prompt
                for lora_name, weights in block_weight_specs:
                    p.prompt = p.prompt.replace(f"<lorabw:{lora_name}:{weights}>", "")
                    # Store the original block weight specification
                    p.extra_generation_params[f"lora_block_weight_{lora_name}"] = weights.strip()

                # Store original model state if needed
                if not hasattr(p.sd_model, 'forge_objects_original'):
                    p.sd_model.forge_objects_original = p.sd_model.forge_objects.shallow_copy()

                for lora_name, weights in block_weight_specs:
                    # Get the LoRA directory
                    lora_dir = shared.cmd_opts.lora_dir
                    lora_path = None

                    # Find the LoRA file
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

                    # Load and process LoRA using cache
                    lora_sd = self.get_cached_lora(lora_path)
                    if lora_sd is None:
                        continue
                    
                    try:
                        # Process block weights
                        block_weights, muted_weights, _ = LoraLoaderBlockWeight.load_lbw(
                            p.sd_model.forge_objects.unet,
                            p.sd_model.forge_objects.clip,
                            lora_sd,
                            self.inverse,
                            self.seed,
                            self.A,
                            self.B,
                            weights
                        )

                        # Apply the blocks
                        new_modelpatcher = p.sd_model.forge_objects.unet.clone()
                        new_clip = p.sd_model.forge_objects.clip.clone()
                        muted_weights = set(muted_weights)

                        for k, v in block_weights.items():
                            weights, ratio = v
                            if k in muted_weights:
                                continue
                            elif 'text' in k or 'encoder' in k:
                                new_clip.add_patches({k: weights}, self.strength_clip * ratio)
                            else:
                                new_modelpatcher.add_patches({k: weights}, self.strength_model * ratio)

                        p.sd_model.forge_objects.unet = new_modelpatcher
                        p.sd_model.forge_objects.clip = new_clip
                        self.lora_applied = True

                    except Exception as e:
                        logging.error(f"Error processing LoRA {lora_name}: {str(e)}")
                        continue

            # Update general generation parameters with clean string values
            params = {
                "lora_block_weight_enabled": str(self.enabled),
                "lora_block_weight_model_strength": f"{float(self.strength_model):.4f}",
                "lora_block_weight_clip_strength": f"{float(self.strength_clip):.4f}",
                "lora_block_weight_inverse": str(self.inverse),
                "lora_block_weight_seed": str(self.seed),
                "lora_block_weight_A": f"{float(self.A):.4f}",
                "lora_block_weight_B": f"{float(self.B):.4f}",
                "lora_block_weight_preset": self.preset,
                "lora_block_weight_vector": self.block_vector
            }

            # Update generation parameters
            p.extra_generation_params.update(params)

        except Exception as e:
            logging.error(f"Error in LoRA Block Weight: {str(e)}")
            raise e

    def postprocess(self, p, processed, *args):
        """Restore original model after generation"""
        if self.lora_applied and hasattr(p.sd_model, 'forge_objects_original'):
            p.sd_model.forge_objects = p.sd_model.forge_objects_original
            self.lora_applied = False

        return