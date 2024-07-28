import sys
import os

# Add the parent directory of the extension to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
extension_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.dirname(extension_dir))

print(f"Extension directory added to sys.path: {os.path.dirname(extension_dir)}")

import gradio as gr
from modules import scripts

# Now import from your package
from forge_jankhidiffusion.raunet import ApplyRAUNet, ApplyRAUNetSimple, UPSCALE_METHODS
from forge_jankhidiffusion.msw_msa_attention import ApplyMSWMSAAttention, ApplyMSWMSAAttentionSimple

print("Imports successful in RAUNet script")
opApplyRAUNet = ApplyRAUNet()
opApplyMSWMSA = ApplyMSWMSAAttention()

class RAUNetScript(scripts.Script):
    sorting_priority = 15  # Adjust this as needed

    def title(self):
        return "RAUNet and MSW-MSA for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Tab("RAUNet"):
                raunet_enabled = gr.Checkbox(label="RAUNet Enabled", value=False)
                
                with gr.Group(visible=False) as raunet_options:
                    input_blocks = gr.Text(label="Input Blocks", value="3")
                    output_blocks = gr.Text(label="Output Blocks", value="8")
                    time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                    start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                    end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.45)
                    two_stage_upscale = gr.Checkbox(label="Two Stage Upscale", value=False)
                    upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="Upscale Mode")
            
                    with gr.Accordion(open=False, label="Cross-Attention Settings"):
                        ca_start_time = gr.Slider(label="CA Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        ca_end_time = gr.Slider(label="CA End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                        ca_input_blocks = gr.Text(label="CA Input Blocks", value="4")
                        ca_output_blocks = gr.Text(label="CA Output Blocks", value="8")
                        ca_upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="CA Upscale Mode")

            with gr.Tab("MSW-MSA"):
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Enabled", value=False)
                
                with gr.Group(visible=False) as mswmsa_options:
                    mswmsa_input_blocks = gr.Text(label="Input Blocks", value="1,2")
                    mswmsa_middle_blocks = gr.Text(label="Middle Blocks", value="0")
                    mswmsa_output_blocks = gr.Text(label="Output Blocks", value="9,10,11")
                    mswmsa_time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                    mswmsa_start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
                    mswmsa_end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0)

            gr.HTML("<p><i>Note: MSW-MSA seems to not be working at the moment.</i></p>")

        # Add JavaScript to handle visibility of options
        raunet_enabled.change(
            fn=lambda x: gr.Group.update(visible=x),
            inputs=[raunet_enabled],
            outputs=[raunet_options]
        )
        
        mswmsa_enabled.change(
            fn=lambda x: gr.Group.update(visible=x),
            inputs=[mswmsa_enabled],
            outputs=[mswmsa_options]
        )

        return (raunet_enabled, input_blocks, output_blocks, time_mode, start_time, end_time, 
                two_stage_upscale, upscale_mode, ca_start_time, ca_end_time, ca_input_blocks, 
                ca_output_blocks, ca_upscale_mode, mswmsa_enabled, mswmsa_input_blocks, 
                mswmsa_middle_blocks, mswmsa_output_blocks, mswmsa_time_mode, mswmsa_start_time, 
                mswmsa_end_time)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            raunet_enabled, input_blocks, output_blocks, time_mode, start_time, end_time, two_stage_upscale, upscale_mode,
            ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode,
            mswmsa_enabled, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time
        ) = script_args

        unet = p.sd_model.forge_objects.unet

        if raunet_enabled:
            unet = opApplyRAUNet.patch(
                True, unet, input_blocks, output_blocks, time_mode, start_time, end_time, two_stage_upscale, upscale_mode,
                ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode
            )[0]

            p.extra_generation_params.update(
                dict(
                    raunet_enabled=raunet_enabled,
                    raunet_input_blocks=input_blocks,
                    raunet_output_blocks=output_blocks,
                    raunet_time_mode=time_mode,
                    raunet_start_time=start_time,
                    raunet_end_time=end_time,
                    raunet_two_stage_upscale=two_stage_upscale,
                    raunet_upscale_mode=upscale_mode,
                    raunet_ca_start_time=ca_start_time,
                    raunet_ca_end_time=ca_end_time,
                    raunet_ca_input_blocks=ca_input_blocks,
                    raunet_ca_output_blocks=ca_output_blocks,
                    raunet_ca_upscale_mode=ca_upscale_mode,
                )
            )

        if mswmsa_enabled:
            unet = opApplyMSWMSA.patch(
                unet, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time
            )[0]

            p.extra_generation_params.update(
                dict(
                    mswmsa_enabled=mswmsa_enabled,
                    mswmsa_input_blocks=mswmsa_input_blocks,
                    mswmsa_middle_blocks=mswmsa_middle_blocks,
                    mswmsa_output_blocks=mswmsa_output_blocks,
                    mswmsa_time_mode=mswmsa_time_mode,
                    mswmsa_start_time=mswmsa_start_time,
                    mswmsa_end_time=mswmsa_end_time,
                )
            )

        p.sd_model.forge_objects.unet = unet
        