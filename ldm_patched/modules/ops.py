# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


import torch
import ldm_patched.modules.model_management
import contextlib

from modules_forge import stream

# Existing stash and use_patched_ops
stash = {}

@contextlib.contextmanager
def use_patched_ops(operations):
    op_names = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'GroupNorm', 'LayerNorm', 'ConvTranspose1d', 'ConvTranspose2d']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return

def cast_to_input(weight, input, non_blocking=False):
    return weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)


def cast_bias_weight(s, input):
    weight, bias, signal = None, None, None
    non_blocking = ldm_patched.modules.model_management.device_should_use_non_blocking(input.device)
    
    if stream.using_stream:
        with stream.stream_context()(stream.mover_stream):
            if s.bias is not None:
                bias = cast_to_input(s.bias, input, non_blocking=non_blocking)
                if hasattr(s, 'bias_function') and s.bias_function is not None:
                    bias = s.bias_function(bias)
            weight = cast_to_input(s.weight, input, non_blocking=non_blocking)
            if hasattr(s, 'weight_function') and s.weight_function is not None:
                weight = s.weight_function(weight)
            signal = stream.mover_stream.record_event()
    else:
        if s.bias is not None:
            bias = s.bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
            if hasattr(s, 'bias_function') and s.bias_function is not None:
                bias = s.bias_function(bias)
        weight = s.weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
        if hasattr(s, 'weight_function') and s.weight_function is not None:
            weight = s.weight_function(weight)
    
    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if not stream.using_stream or signal is None:
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        stash[id(finished_signal)] = (weight, bias, finished_signal)

    garbage = []
    for k, (w, b, s) in stash.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del stash[k]
    return


def cleanup_cache():
    if not stream.using_stream:
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()
    return

class CastWeightBiasOp:
    ldm_patched_cast_weights = False
    weight_function = None
    bias_function = None

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)
            
    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, signal = cast_bias_weight(self, input)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            else:
                return torch.nn.functional.layer_norm(input, self.normalized_shape, None, None, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.conv_transpose2d(
                    input, weight, bias, self.stride, self.padding,
                    output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.conv_transpose1d(
                    input, weight, bias, self.stride, self.padding,
                    output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 1:
            return s.Conv1d(*args, **kwargs)
        elif dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True
    class Conv1d(disable_weight_init.Conv1d):
        ldm_patched_cast_weights = True
    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True
    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True
    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True
    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True
    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        ldm_patched_cast_weights = True
    class Embedding(disable_weight_init.Embedding):
        ldm_patched_cast_weights = True
    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        ldm_patched_cast_weights = True
