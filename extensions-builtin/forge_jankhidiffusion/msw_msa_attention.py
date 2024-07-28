import torch

from .utils import *


class ApplyMSWMSAAttention:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_blocks": ("STRING", {"default": "1,2"}),
                "middle_blocks": ("STRING", {"default": ""}),
                "output_blocks": ("STRING", {"default": "9,10,11"}),
                "time_mode": (["percent", "timestep", "sigma"],),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 999.0}),
                "model": ("MODEL",),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    # reference: https://github.com/microsoft/Swin-Transformer
    # Window functions adapted from https://github.com/megvii-research/HiDiffusion
    @staticmethod
    def window_partition(x, window_size, shift_size, height, width) -> torch.Tensor:
        batch, _features, channels = x.shape
        x = x.view(batch, height, width, channels)
        if not isinstance(shift_size, (list, tuple)):
            shift_size = (shift_size, shift_size)
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        x = x.view(
            batch,
            height // window_size[0],
            window_size[0],
            width // window_size[1],
            window_size[1],
            channels,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0], window_size[1], channels)
        )
        return windows.view(-1, window_size[0] * window_size[1], channels)

    @staticmethod
    def window_reverse(windows, window_size, shift_size, height, width) -> torch.Tensor:
        batch, features, channels = windows.shape
        windows = windows.view(-1, window_size[0], window_size[1], channels)
        batch = int(
            windows.shape[0] / (height * width / window_size[0] / window_size[1]),
        )
        x = windows.view(
            batch,
            height // window_size[0],
            width // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
        if not isinstance(shift_size, (list, tuple)):
            shift_size = (shift_size, shift_size)
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        return x.view(batch, height * width, channels)

    @staticmethod
    def get_window_args(n, orig_shape, shift) -> tuple:
        _batch, features, _channels = n.shape
        orig_height, orig_width = orig_shape[-2:]

        downsample_ratio = int(
            ((orig_height * orig_width) // features) ** 0.5,
        )
        height, width = (
            orig_height // downsample_ratio,
            orig_width // downsample_ratio,
        )
        window_size = (height // 2, width // 2)

        if shift == 0:
            shift_size = (0, 0)
        elif shift == 1:
            shift_size = (window_size[0] // 4, window_size[1] // 4)
        elif shift == 2:
            shift_size = (window_size[0] // 4 * 2, window_size[1] // 4 * 2)
        else:
            shift_size = (window_size[0] // 4 * 3, window_size[1] // 4 * 3)
        return (window_size, shift_size, height, width)

    def patch(self, model, input_blocks, middle_blocks, output_blocks, time_mode, start_time, end_time):
        use_blocks = parse_blocks("input", input_blocks)
        use_blocks |= parse_blocks("middle", middle_blocks)
        use_blocks |= parse_blocks("output", output_blocks)

        model = model.clone()
        ms = model.model_sampling
        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)

        def attn1_patch(q, k, v, extra_options):
            if extra_options.get("block") not in use_blocks or not check_time(extra_options, start_sigma, end_sigma):
                return q, k, v

            orig_shape = extra_options["original_shape"]
            shift = int(torch.rand(1, device="cpu").item() * 4)
            window_args = self.get_window_args(q, orig_shape, shift)

            try:
                return (self.window_partition(q, *window_args),) * 3
            except RuntimeError as exc:
                errstr = f"MSW-MSA attention error: Incompatible model patches or bad resolution. Try using resolutions that are multiples of 32 or 64. Original exception: {exc}"
                raise RuntimeError(errstr) from exc

        def attn1_output_patch(n, extra_options):
            if extra_options.get("block") not in use_blocks or not check_time(extra_options, start_sigma, end_sigma):
                return n

            orig_shape = extra_options["original_shape"]
            shift = int(torch.rand(1, device="cpu").item() * 4)
            window_args = self.get_window_args(n, orig_shape, shift)

            return self.window_reverse(n, *window_args)

        model.set_model_attn1_patch(attn1_patch)
        model.set_model_attn1_output_patch(attn1_output_patch)

        return (model,)


class ApplyMSWMSAAttentionSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["SD15", "SDXL"],),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    def go(self, model_type, model):
        time_range = (0.2, 1.0)
        if model_type == "SD15":
            blocks = ("1,2", "", "11,10,9")
        elif model_type == "SDXL":
            blocks = ("4,5", "", "5,4")
        else:
            raise ValueError("Unknown model type")

        print(f"** ApplyMSWMSAAttentionSimple: Using preset {model_type}: in/mid/out blocks [{' / '.join(b if b else 'none' for b in blocks)}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}")
        
        return ApplyMSWMSAAttention().patch(model, *blocks, "percent", *time_range)


__all__ = ("ApplyMSWMSAAttention", "ApplyMSWMSAAttentionSimple")
