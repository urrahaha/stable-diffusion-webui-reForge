import gradio as gr
from modules import scripts
from ldm_patched.modules import model_management

class NeverOOMForForge(scripts.Script):
    sorting_priority = 18
    def __init__(self):
        self.previous_vram_state = None
        self.original_vram_state = model_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(
                label="Enable VRAM Management",
                value=False,
                info="Turn on to adjust VRAM usage settings"
            )
            
            with gr.Group(visible=False) as options_group:
                vram_options = gr.Radio(
                    choices=[
                        "Disabled",
                        "No VRAM (Maximum Offload)",
                        "Low VRAM",
                        "Normal VRAM",
                        "High VRAM",
                    ],
                    label="VRAM Management Options",
                    value="Disabled",
                    info="Choose how VRAM is managed"
                )
                
                vae_options = gr.Checkbox(
                    label="Enable VAE Tiling",
                    value=False,
                    info="Enable to use tiled VAE processing, which can help with memory usage"
                )
                
                feedback = gr.Markdown("Current status: VRAM Management disabled")
                
                with gr.Group():
                    gr.Markdown("### VRAM Management Options Info")
                    gr.Markdown("""
                    - **Disabled**: Use default VRAM settings
                    - **No VRAM (Maximum Offload)**: Use when low VRAM mode isn't sufficient (Always maximize offload)
                    - **Low VRAM**: Split the U-Net into parts to use less VRAM
                    - **Normal VRAM**: Standard VRAM usage
                    - **High VRAM**: Keep models in GPU memory after use instead of unloading to CPU memory
                    """)
            
            def toggle_options(enabled):
                return gr.Group.update(visible=enabled)

            def update_feedback(enabled, vram_option, vae_enabled):
                if not enabled:
                    return "Current status: VRAM Management disabled"
                status = f"Current status: VRAM Management set to {vram_option}"
                if vae_enabled:
                    status += ", VAE Tiling enabled"
                return status

            enabled.change(toggle_options, inputs=[enabled], outputs=[options_group])
            enabled.change(update_feedback, inputs=[enabled, vram_options, vae_options], outputs=[feedback])
            vram_options.change(update_feedback, inputs=[enabled, vram_options, vae_options], outputs=[feedback])
            vae_options.change(update_feedback, inputs=[enabled, vram_options, vae_options], outputs=[feedback])

        return enabled, vram_options, vae_options

    def process(self, p, enabled, vram_option, vae_enabled):
        if not enabled:
            if self.previous_vram_state is not None:
                model_management.vram_state = self.original_vram_state
                print(f'VRAM Management disabled. VRAM State Reset to Original: {self.original_vram_state.name}')
                self.previous_vram_state = None
            return

        if vae_enabled:
            print('VAE Tiling Enabled')
            model_management.VAE_ALWAYS_TILED = True
        else:
            model_management.VAE_ALWAYS_TILED = False

        vram_state_map = {
            "Disabled": self.original_vram_state,
            "No VRAM (Maximum Offload)": model_management.VRAMState.NO_VRAM,
            "Low VRAM": model_management.VRAMState.LOW_VRAM,
            "Normal VRAM": model_management.VRAMState.NORMAL_VRAM,
            "High VRAM": model_management.VRAMState.HIGH_VRAM,
        }
        
        new_vram_state = vram_state_map.get(vram_option, self.original_vram_state)
        
        if self.previous_vram_state != new_vram_state:
            model_management.unload_all_models()
            model_management.vram_state = new_vram_state
            if new_vram_state != self.original_vram_state:
                print(f'VRAM State Changed To {new_vram_state.name}')
            self.previous_vram_state = new_vram_state
        
        return
