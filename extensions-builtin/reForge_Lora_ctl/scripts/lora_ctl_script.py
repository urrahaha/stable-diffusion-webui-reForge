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

    sorting_priority = 15

    def title(self):
        return "LoRA Control for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
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

        return (enabled, schedule_text, split_sampling)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 3:
            self.enabled, self.schedule_text, self.split_sampling = args[:3]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        try:
            # Store original UNet for restoration
            if not hasattr(p.sd_model.forge_objects.unet, '_original_model'):
                p.sd_model.forge_objects.unet._original_model = p.sd_model.forge_objects.unet.model

            unet = p.sd_model.forge_objects.unet.clone()

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