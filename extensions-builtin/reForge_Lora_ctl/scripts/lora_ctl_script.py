import logging
import gradio as gr
from modules import scripts, shared_options as opts
from lora_ctl.parser import parse_prompt_schedules
from lora_ctl.utils import unpatch_model, clone_model, set_callback, apply_loras_from_spec

class LoRaControlScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.schedule_text = ""
        self.split_sampling = False
        # Add block weight related fields
        self.block_weight_enabled = False
        self.block_weight_preset = "Simple"
        self.custom_block_weights = ""
        self.weight_presets = {
            "Simple": "BASE:1",
            "Block In/Out": "IN:1,OUT:1",
            "Full Block": "BASE:1,IN:1,OUT:1,MIDDLE:1",
            "Detailed": "BASE:1,IN00:1,IN01:1,IN02:1,IN03:1,IN04:1,IN05:1,IN06:1,IN07:1,IN08:1,IN09:1,IN10:1,IN11:1,OUT00:1,OUT01:1,OUT02:1,OUT03:1,OUT04:1,OUT05:1,OUT06:1,OUT07:1,OUT08:1,OUT09:1,OUT10:1,OUT11:1,M00:1",
            "Custom": ""
        }

    sorting_priority = 15

    def title(self):
        return "LoRA Control for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Tab("Schedule"):
                gr.HTML("<p><i>Control LoRA application during generation steps.</i></p>")
                enabled = gr.Checkbox(label="Enable LoRA Control", value=self.enabled)
                schedule_text = gr.TextArea(
                    label="LoRA Schedule",
                    value=self.schedule_text,
                    placeholder="Enter your LoRA schedule here...",
                    lines=5
                )
                split_sampling = gr.Checkbox(
                    label="Enable Split Sampling",
                    value=self.split_sampling
                )

            with gr.Tab("Block Weights"):
                block_weight_enabled = gr.Checkbox(
                    label="Enable Block Weights",
                    value=self.block_weight_enabled
                )
                block_weight_preset = gr.Dropdown(
                    label="Block Weight Preset",
                    choices=list(self.weight_presets.keys()),
                    value=self.block_weight_preset
                )
                custom_block_weights = gr.TextArea(
                    label="Custom Block Weights",
                    value=self.custom_block_weights,
                    placeholder="Format: BLOCK:WEIGHT,BLOCK:WEIGHT (e.g., BASE:1,IN:0.5,OUT:0.8)",
                    visible=False
                )

                def update_custom_weights(preset):
                    return {
                        custom_block_weights: gr.update(
                            visible=(preset == "Custom"),
                            value=self.weight_presets.get(preset, "")
                        )
                    }

                block_weight_preset.change(
                    update_custom_weights,
                    inputs=[block_weight_preset],
                    outputs=[custom_block_weights]
                )

        return (enabled, schedule_text, split_sampling, 
                block_weight_enabled, block_weight_preset, custom_block_weights)

    def parse_block_weights(self, preset, custom_weights):
        if preset == "Custom":
            weights_str = custom_weights
        else:
            weights_str = self.weight_presets.get(preset, "")

        block_weights = {}
        if weights_str:
            try:
                for part in weights_str.split(','):
                    if ':' in part:
                        block, weight = part.strip().split(':')
                        block_weights[block.strip()] = float(weight)
            except Exception as e:
                logging.error(f"Error parsing block weights: {str(e)}")
                return None
        return block_weights

    def apply_block_weights(self, model, block_weights):
        if not block_weights:
            return model

        try:
            # Clone the model to avoid modifying the original
            m = model.clone()
            
            # Apply block weights to the model
            for block_name, weight in block_weights.items():
                if block_name == "BASE":
                    # Apply base model weight
                    if hasattr(m, "set_base_weight"):
                        m.set_base_weight(weight)
                else:
                    # Apply specific block weights
                    if hasattr(m, "set_block_weight"):
                        m.set_block_weight(block_name, weight)
            
            return m
        except Exception as e:
            logging.error(f"Error applying block weights: {str(e)}")
            return model

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 6:
            (self.enabled, self.schedule_text, self.split_sampling,
             self.block_weight_enabled, self.block_weight_preset,
             self.custom_block_weights) = args[:6]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not (self.enabled or self.block_weight_enabled):
            return

        try:
            # Store original UNet for restoration
            if not hasattr(p.sd_model.forge_objects.unet, '_original_model'):
                p.sd_model.forge_objects.unet._original_model = p.sd_model.forge_objects.unet.model

            unet = p.sd_model.forge_objects.unet.clone()

            # Apply block weights if enabled
            if self.block_weight_enabled:
                block_weights = self.parse_block_weights(
                    self.block_weight_preset,
                    self.custom_block_weights
                )
                if block_weights:
                    unet = self.apply_block_weights(unet, block_weights)

            # Continue with LoRA schedule if enabled
            if self.enabled and self.schedule_text.strip():
                schedules = parse_prompt_schedules(self.schedule_text)
                
                # Create state for handling LoRA applications
                state = {
                    "model": unet,
                    "applied_loras": {},
                }
                
                orig_model = clone_model(unet)
                orig_model.model_options["pc_schedules"] = schedules
                orig_model.model_options["pc_split_sampling"] = self.split_sampling

                lora_cache = {}

                def sampler_cb(orig_sampler, is_custom, *args, **kwargs):
                    if is_custom:
                        steps = len(args[4])
                        logging.info(
                            "SamplerCustom detected, using sigma length as steps: %s",
                            steps,
                        )
                    else:
                        steps = args[2]
                    
                    start_step = kwargs.get("start_step") or 0
                    state["model"] = args[0]

                    orig_cb = kwargs.get("callback")

                    def step_callback(*args, **kwargs):
                        current_step = args[0] + start_step
                        self.apply_lora_for_step(schedules, current_step, steps, state, orig_model, lora_cache)
                        if orig_cb:
                            return orig_cb(*args, **kwargs)

                    kwargs["callback"] = step_callback
                    
                    self.apply_lora_for_step(schedules, start_step, steps, state, orig_model, lora_cache)
                    
                    args = list(args)
                    args[0] = state["model"]
                    result = orig_sampler(*args, **kwargs)
                    
                    unpatch_model(state["model"])
                    return result

                set_callback(orig_model, sampler_cb)
                unet = orig_model

            p.sd_model.forge_objects.unet = unet

            # Add parameters to generation info
            p.extra_generation_params.update({
                "lora_control_enabled": self.enabled,
                "lora_schedule": self.schedule_text if self.enabled else None,
                "split_sampling": self.split_sampling if self.enabled else None,
                "block_weight_enabled": self.block_weight_enabled,
                "block_weight_preset": self.block_weight_preset if self.block_weight_enabled else None,
                "custom_block_weights": self.custom_block_weights if self.block_weight_enabled and self.block_weight_preset == "Custom" else None,
            })

        except Exception as e:
            logging.error(f"Error in LoRA Control: {str(e)}")
            raise e

    def apply_lora_for_step(self, schedules, step, total_steps, state, original_model, lora_cache):
        sched = schedules.at_step(step + 1, total_steps)
        lora_spec = sched[1]["loras"]

        if state["applied_loras"] != lora_spec:
            logging.debug("At step %s, applying lora_spec %s", step, lora_spec)
            m, _ = apply_loras_from_spec(
                lora_spec,
                model=state["model"],
                orig_model=original_model,
                cache=lora_cache,
                patch=True,
                applied_loras=state["applied_loras"],
            )
            state["model"] = m
            state["applied_loras"] = lora_spec

    def postprocess(self, p, processed, *args):
        """Restore original model after generation"""
        if hasattr(p.sd_model.forge_objects.unet, '_original_model'):
            p.sd_model.forge_objects.unet.model = p.sd_model.forge_objects.unet._original_model
            del p.sd_model.forge_objects.unet._original_model

        return