import logging
import gradio as gr
from modules import scripts
from APGIsYourCFG.nodes_APGImYourCFGNow import APG_ImYourCFGNow


class APGIsNowYourCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.apg_moment = 0.5
        self.apg_adaptive_moment = 0.180
        self.apg_norm_thr = 15.0
        self.apg_eta = 1.0

    sorting_priority = 15

    def title(self):
        return '"APG\'s now your CFG" for reForge'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for APG's now your CFG.</i></p>")
            enabled = gr.Checkbox(label="Enable APG's now your CFG", value=self.enabled)
            apg_momentum = gr.Slider(
                label="APG Momentum",
                minimum=-1.5,
                maximum=0.5,
                step=0.01,
                value=self.apg_moment,
            )
            apg_adaptive_momentum = gr.Slider(
                label="APG Momentum",
                minimum=-1.5,
                maximum=0.5,
                step=0.01,
                value=self.apg_adaptive_moment,
            )
            apg_norm_thr = gr.Slider(
                label="APG Norm Threshold",
                minimum=0.5,
                maximum=50.0,
                step=0.5,
                value=self.apg_norm_thr,
            )
            apg_eta = gr.Slider(
                label="APG Eta", minimum=0.0, maximum=1.0, step=0.1, value=self.apg_eta
            )

        enabled.change(lambda x: self.update_enabled(x), inputs=[enabled])

        return (enabled, apg_momentum, apg_adaptive_momentum, apg_norm_thr, apg_eta)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 5:
            (
                self.enabled,
                self.apg_moment,
                self.apg_adaptive_moment,
                self.apg_norm_thr,
                self.apg_eta,
            ) = args[:5]
        else:
            logging.warning(
                "Not enough arguments provided to process_before_every_sampling"
            )
            return

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        unet = APG_ImYourCFGNow().patch(
            unet,
            momentum=self.apg_moment,
            adaptive_momentum=self.apg_moment,
            norm_threshold=self.apg_norm_thr,
            eta=self.apg_eta
        )[0]

        p.sd_model.forge_objects.unet = unet
        args = {
            "apgisyourcfg_enabled": True,
            "apgisyourcfg_momentum": self.apg_moment,
            "apgisyourcfg_adaptive_momentum": self.apg_adaptive_moment,
            "apgisyourcfg_norm_thr": self.apg_norm_thr,
            "apgisyourcfg_eta": self.apg_eta,
        }
        p.extra_generation_params.update(args)
        str_args:str = ", ".join([f"{k}:\"{v}\"" for k,v in args.items()])
        logging.debug("WOLOLO: \"APG is now your CFG!\"")
        logging.debug(str_args)

        return
