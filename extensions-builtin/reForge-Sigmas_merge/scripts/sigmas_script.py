import gradio as gr
from modules import scripts
from sigmas.sigmas_merge import (
    sigmas_merge, sigmas_gradual_merge, multi_sigmas_average, sigmas_mult,
    sigmas_concat, the_golden_scheduler, GaussianTailScheduler, aligned_scheduler,
    manual_scheduler, get_sigma_float, sigmas_to_graph, sigmas_min_max_out_node
)

class SigmasMergeScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.mode = "Merge sigmas by average"
        self.proportion_1 = 0.5
        self.factor = 1.0
        self.sigmas_1_until = 10
        self.rescale_sum = False
        self.steps = 20
        self.sgm = False
        self.model_type = "SD1"
        self.force_sigma_min = False
        self.custom_sigmas_manual_schedule = "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"
        self.color = "blue"
        self.print_as_list = False

    sorting_priority = 15

    def title(self):
        return "Sigmas Merge for Forge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Sigmas Merge.</i></p>")

            enabled = gr.Checkbox(label="Enable Sigmas Merge", value=self.enabled)

            with gr.Group():
                mode = gr.Dropdown(
                    ["Merge sigmas by average", "Merge sigmas gradually", "Merge many sigmas by average",
                     "Multiply sigmas", "Split and concatenate sigmas", "The Golden Scheduler",
                     "Gaussian Tail Scheduler", "Aligned Scheduler", "Manual scheduler",
                     "Get sigmas as float", "Graph sigmas", "Output min/max sigmas"],
                    label="Sigmas Merge Mode",
                    value=self.mode
                )

                with gr.Group() as merge_group:
                    proportion_1 = gr.Slider(label="Proportion 1", minimum=0.0, maximum=1.0, step=0.01, value=self.proportion_1)

                with gr.Group() as multiply_group:
                    factor = gr.Slider(label="Factor", minimum=0.0, maximum=100.0, step=0.01, value=self.factor)

                with gr.Group() as concat_group:
                    sigmas_1_until = gr.Slider(label="Sigmas 1 Until", minimum=0, maximum=1000, step=1, value=self.sigmas_1_until)
                    rescale_sum = gr.Checkbox(label="Rescale Sum", value=self.rescale_sum)

                with gr.Group() as scheduler_group:
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100000, step=1, value=self.steps)
                    sgm = gr.Checkbox(label="SGM", value=self.sgm)
                    model_type = gr.Dropdown(["SD1", "SDXL", "SVD"], label="Model Type", value=self.model_type)
                    force_sigma_min = gr.Checkbox(label="Force Sigma Min", value=self.force_sigma_min)

                with gr.Group() as manual_scheduler_group:
                    custom_sigmas_manual_schedule = gr.Textbox(label="Custom Sigmas Manual Schedule", value=self.custom_sigmas_manual_schedule)

                with gr.Group() as graph_group:
                    color = gr.Dropdown(["blue", "red", "green", "cyan", "magenta", "yellow", "black"], label="Graph Color", value=self.color)
                    print_as_list = gr.Checkbox(label="Print as List", value=self.print_as_list)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode in ["Merge sigmas by average", "Merge sigmas gradually", "Merge many sigmas by average"])),
                    gr.Group.update(visible=(mode == "Multiply sigmas")),
                    gr.Group.update(visible=(mode == "Split and concatenate sigmas")),
                    gr.Group.update(visible=(mode in ["The Golden Scheduler", "Gaussian Tail Scheduler", "Aligned Scheduler"])),
                    gr.Group.update(visible=(mode == "Manual scheduler")),
                    gr.Group.update(visible=(mode == "Graph sigmas"))
                )

            mode.change(
                update_visibility,
                inputs=[mode],
                outputs=[merge_group, multiply_group, concat_group, scheduler_group, manual_scheduler_group, graph_group]
            )

            enabled.change(
                lambda x: self.update_enabled(x),
                inputs=[enabled]
            )

        return (enabled, mode, proportion_1, factor, sigmas_1_until, rescale_sum, steps, sgm, model_type, force_sigma_min, custom_sigmas_manual_schedule, color, print_as_list)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 13:
            self.enabled, self.mode, self.proportion_1, self.factor, self.sigmas_1_until, self.rescale_sum, self.steps, self.sgm, self.model_type, self.force_sigma_min, self.custom_sigmas_manual_schedule, self.color, self.print_as_list = args[:13]
        else:
            print("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        model = p.sd_model.forge_objects.unet
        model_sampling = model.model.model_sampling

        original_sigmas = model_sampling.sigmas.clone()
        original_log_sigmas = model_sampling.log_sigmas.clone() if hasattr(model_sampling, 'log_sigmas') else None

        try:
            if self.mode == "Merge sigmas by average":
                new_sigmas = sigmas_merge().simple_output(original_sigmas, original_sigmas, self.proportion_1)[0]
            elif self.mode == "Merge sigmas gradually":
                new_sigmas = sigmas_gradual_merge().simple_output(original_sigmas, original_sigmas, self.proportion_1)[0]
            elif self.mode == "Merge many sigmas by average":
                new_sigmas = multi_sigmas_average().simple_output(original_sigmas)[0]
            elif self.mode == "Multiply sigmas":
                new_sigmas = sigmas_mult().simple_output(original_sigmas, self.factor)[0]
            elif self.mode == "Split and concatenate sigmas":
                new_sigmas = sigmas_concat().simple_output(original_sigmas, original_sigmas, self.sigmas_1_until, self.rescale_sum)[0]
            elif self.mode == "The Golden Scheduler":
                new_sigmas = the_golden_scheduler().simple_output(model, self.steps, self.sgm)[0]
            elif self.mode == "Gaussian Tail Scheduler":
                new_sigmas = GaussianTailScheduler().simple_output(model, self.steps)[0]
            elif self.mode == "Aligned Scheduler":
                new_sigmas = aligned_scheduler().simple_output(model, self.steps, self.model_type, self.force_sigma_min)[0]
            elif self.mode == "Manual scheduler":
                new_sigmas = manual_scheduler().simple_output(model, self.custom_sigmas_manual_schedule, self.steps, self.sgm)[0]
            elif self.mode == "Get sigmas as float":
                sigma_float = get_sigma_float().simple_output(original_sigmas, model)[0]
                print(f"Sigma as float: {sigma_float}")
                return
            elif self.mode == "Graph sigmas":
                sigmas_to_graph().simple_output(original_sigmas, self.color, self.print_as_list)
                return
            elif self.mode == "Output min/max sigmas":
                min_sigma, max_sigma = sigmas_min_max_out_node().simple_output(model)
                print(f"Min sigma: {min_sigma}, Max sigma: {max_sigma}")
                return

            import torch
            # Ensure new_sigmas has the same device as original_sigmas
            new_sigmas = new_sigmas.to(original_sigmas.device)

            # If the size of new_sigmas is different, interpolate to match the original size
            if new_sigmas.size(0) != original_sigmas.size(0):
                print(f"Interpolating new sigmas from size {new_sigmas.size(0)} to {original_sigmas.size(0)}")
                new_sigmas = torch.nn.functional.interpolate(new_sigmas.unsqueeze(0).unsqueeze(0), 
                                                            size=original_sigmas.size(0), 
                                                            mode='linear', 
                                                            align_corners=False).squeeze()

            # Update the sigmas in the model_sampling object
            model_sampling.set_sigmas(new_sigmas)

            # If log_sigmas exists, update it as well
            if original_log_sigmas is not None:
                model_sampling.register_buffer('log_sigmas', new_sigmas.log().float())

            print(f"Original sigmas: {original_sigmas[:5]} ... {original_sigmas[-5:]}")
            print(f"New sigmas: {model_sampling.sigmas[:5]} ... {model_sampling.sigmas[-5:]}")
            print(f"Difference: {torch.mean(torch.abs(original_sigmas - model_sampling.sigmas)).item()}")

            # Visualize the sigmas
            self.plot_sigmas(original_sigmas, model_sampling.sigmas)

        except Exception as e:
            print(f"Error in {self.mode}: {str(e)}")
            import traceback
            traceback.print_exc()

        p.extra_generation_params.update({
            "sigmas_merge_enabled": self.enabled,
            "sigmas_merge_mode": self.mode,
            "proportion_1": self.proportion_1 if self.mode in ["Merge sigmas by average", "Merge sigmas gradually"] else None,
            "factor": self.factor if self.mode == "Multiply sigmas" else None,
            "sigmas_1_until": self.sigmas_1_until if self.mode == "Split and concatenate sigmas" else None,
            "rescale_sum": self.rescale_sum if self.mode == "Split and concatenate sigmas" else None,
            "steps": self.steps if self.mode in ["The Golden Scheduler", "Gaussian Tail Scheduler", "Aligned Scheduler", "Manual scheduler"] else None,
            "sgm": self.sgm if self.mode in ["The Golden Scheduler", "Manual scheduler"] else None,
            "model_type": self.model_type if self.mode == "Aligned Scheduler" else None,
            "force_sigma_min": self.force_sigma_min if self.mode == "Aligned Scheduler" else None,
            "custom_sigmas_manual_schedule": self.custom_sigmas_manual_schedule if self.mode == "Manual scheduler" else None,
            "color": self.color if self.mode == "Graph sigmas" else None,
            "print_as_list": self.print_as_list if self.mode == "Graph sigmas" else None,
        })

        print(f"Sigmas Merge: Enabled: {self.enabled}, Mode: {self.mode}")
