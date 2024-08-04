import logging
import gradio as gr
from modules import scripts
import torch
from skimmed_CFG.skimmed_CFG import get_skimming_mask, skimmed_CFG

class SkimmedCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.mode = "Single Scale"
        self.skimming_cfg = 7.0
        self.razor_skim = False
        self.lin_interp_cfg = 5.0
        self.skimming_cfg_positive = 5.0
        self.skimming_cfg_negative = 5.0

    def title(self):
        return "NOT WORKING-Skimmed CFG for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Skimmed CFG.</i></p>")
            
            enabled = gr.Checkbox(label="Enable Skimmed CFG", value=self.enabled)
            
            with gr.Group():
                mode = gr.Radio(
                    ["Single Scale", "Replace", "Linear Interpolation", "Dual Scales"],
                    label="Skimmed CFG Mode",
                    value=self.mode
                )
                
                with gr.Group() as single_scale_group:
                    skimming_cfg = gr.Slider(label="Skimming CFG", minimum=0.0, maximum=7.0, step=0.01, value=self.skimming_cfg)
                    razor_skim = gr.Checkbox(label="Razor Skim", value=self.razor_skim)

                with gr.Group() as lin_interp_group:
                    lin_interp_cfg = gr.Slider(label="Skimming CFG", minimum=0.0, maximum=10.0, step=0.01, value=self.lin_interp_cfg)

                with gr.Group() as dual_scales_group:
                    skimming_cfg_positive = gr.Slider(label="Skimming CFG Positive", minimum=0.0, maximum=10.0, step=0.01, value=self.skimming_cfg_positive)
                    skimming_cfg_negative = gr.Slider(label="Skimming CFG Negative", minimum=0.0, maximum=10.0, step=0.01, value=self.skimming_cfg_negative)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Single Scale")),
                    gr.Group.update(visible=(mode == "Linear Interpolation")),
                    gr.Group.update(visible=(mode == "Dual Scales"))
                )

            mode.change(
                update_visibility,
                inputs=[mode],
                outputs=[single_scale_group, lin_interp_group, dual_scales_group]
            )

            enabled.change(
                lambda x: self.update_enabled(x),
                inputs=[enabled]
            )

        return (enabled, mode, skimming_cfg, razor_skim, lin_interp_cfg, skimming_cfg_positive, skimming_cfg_negative)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 7:
            self.enabled, self.mode, self.skimming_cfg, self.razor_skim, self.lin_interp_cfg, self.skimming_cfg_positive, self.skimming_cfg_negative = args[:7]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        def pre_cfg_function(model, cond, uncond, x, timestep, model_options):
            cond_scale = model_options.get("cond_scale", 1.0)

            print("Debug: x shape:", x.shape)
            print("Debug: timestep:", timestep)
            print("Debug: cond_scale:", cond_scale)
            print("Debug: cond keys:", cond[0].keys())
            print("Debug: uncond keys:", uncond[0].keys())
            
            for key in cond[0].keys():
                if isinstance(cond[0][key], torch.Tensor):
                    print(f"Debug: cond[0][{key}] shape:", cond[0][key].shape)
                else:
                    print(f"Debug: cond[0][{key}] type:", type(cond[0][key]))
            
            for key in uncond[0].keys():
                if isinstance(uncond[0][key], torch.Tensor):
                    print(f"Debug: uncond[0][{key}] shape:", uncond[0][key].shape)
                else:
                    print(f"Debug: uncond[0][{key}] type:", type(uncond[0][key]))

            if not any(uncond):
                return model, cond, uncond, x, timestep, model_options

            def pad_or_truncate(tensor, target_length):
                if tensor.shape[1] > target_length:
                    return tensor[:, :target_length]
                elif tensor.shape[1] < target_length:
                    pad_size = target_length - tensor.shape[1]
                    return torch.cat([tensor, tensor[:, -1:].repeat(1, pad_size, 1)], dim=1)
                return tensor

            def apply_skimmed_cfg(cond_dict, uncond_dict):
                result = uncond_dict.copy()
                if 'cross_attn' in cond_dict and 'cross_attn' in uncond_dict:
                    cond_attn = cond_dict['cross_attn']
                    uncond_attn = uncond_dict['cross_attn']
                    
                    uncond_attn = pad_or_truncate(uncond_attn, cond_attn.shape[1])
                    
                    # We're not using x here anymore, as it's handled in the skimmed_CFG function
                    result['cross_attn'] = skimmed_CFG(None, cond_attn, uncond_attn, cond_scale, self.skimming_cfg if not self.razor_skim else 0)
                
                return result

            def apply_replace(cond_dict, uncond_dict):
                result = uncond_dict.copy()
                if 'cross_attn' in cond_dict and 'cross_attn' in uncond_dict:
                    cond_attn = cond_dict['cross_attn']
                    uncond_attn = uncond_dict['cross_attn']
                    
                    uncond_attn = pad_or_truncate(uncond_attn, cond_attn.shape[1])
                    
                    skim_mask = get_skimming_mask(x, cond_attn, uncond_attn, cond_scale)
                    result['cross_attn'] = uncond_attn.clone()
                    result['cross_attn'][skim_mask] = cond_attn[skim_mask]
                
                return result

            def apply_linear_interpolation(cond_dict, uncond_dict):
                result = uncond_dict.copy()
                if 'cross_attn' in cond_dict and 'cross_attn' in uncond_dict:
                    cond_attn = cond_dict['cross_attn']
                    uncond_attn = uncond_dict['cross_attn']
                    
                    uncond_attn = pad_or_truncate(uncond_attn, cond_attn.shape[1])
                    
                    fallback_weight = self.lin_interp_cfg / cond_scale
                    skim_mask = get_skimming_mask(x, cond_attn, uncond_attn, cond_scale)
                    result['cross_attn'] = uncond_attn.clone()
                    result['cross_attn'][skim_mask] = cond_attn[skim_mask] * (1 - fallback_weight) + uncond_attn[skim_mask] * fallback_weight
                
                return result

            def apply_dual_scales(cond_dict, uncond_dict):
                result = uncond_dict.copy()
                if 'cross_attn' in cond_dict and 'cross_attn' in uncond_dict:
                    cond_attn = cond_dict['cross_attn']
                    uncond_attn = uncond_dict['cross_attn']
                    
                    uncond_attn = pad_or_truncate(uncond_attn, cond_attn.shape[1])
                    
                    fallback_weight_positive = self.skimming_cfg_positive / cond_scale
                    fallback_weight_negative = self.skimming_cfg_negative / cond_scale
                    skim_mask_pos = get_skimming_mask(x, cond_attn, uncond_attn, cond_scale)
                    skim_mask_neg = get_skimming_mask(x, uncond_attn, cond_attn, cond_scale)
                    result['cross_attn'] = uncond_attn.clone()
                    result['cross_attn'][skim_mask_pos] = cond_attn[skim_mask_pos] * (1 - fallback_weight_positive) + uncond_attn[skim_mask_pos] * fallback_weight_positive
                    result['cross_attn'][skim_mask_neg] = cond_attn[skim_mask_neg] * (1 - fallback_weight_negative) + uncond_attn[skim_mask_neg] * fallback_weight_negative
                
                return result

            cond = cond[0]  # Assuming cond is a list with one element
            uncond = uncond[0]  # Assuming uncond is a list with one element

            if self.mode == "Single Scale":
                uncond = apply_skimmed_cfg(cond, uncond)
                cond = apply_skimmed_cfg(uncond, cond)
            elif self.mode == "Replace":
                uncond = apply_replace(cond, uncond)
                uncond = apply_replace(uncond, cond)
            elif self.mode == "Linear Interpolation":
                uncond = apply_linear_interpolation(cond, uncond)
                uncond = apply_linear_interpolation(uncond, cond)
            elif self.mode == "Dual Scales":
                uncond = apply_dual_scales(cond, uncond)

            return model, [cond], [uncond], x, timestep, model_options

        unet.set_model_sampler_pre_cfg_function(pre_cfg_function)

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update({
            "skimmed_cfg_enabled": self.enabled,
            "skimmed_cfg_mode": self.mode,
            "skimming_cfg": self.skimming_cfg if self.mode == "Single Scale" else None,
            "razor_skim": self.razor_skim if self.mode == "Single Scale" else None,
            "lin_interp_cfg": self.lin_interp_cfg if self.mode == "Linear Interpolation" else None,
            "skimming_cfg_positive": self.skimming_cfg_positive if self.mode == "Dual Scales" else None,
            "skimming_cfg_negative": self.skimming_cfg_negative if self.mode == "Dual Scales" else None,
        })

        logging.debug(f"Skimmed CFG: Enabled: {self.enabled}, Mode: {self.mode}")

        return
