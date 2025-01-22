# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


import torch
import copy
import inspect
import logging
import uuid

import ldm_patched.modules.utils
import ldm_patched.modules.model_management
from ldm_patched.modules.types import UnetWrapperFunction
import collections
import ldm_patched.float

extra_weight_calculators = {}


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength):
    dora_scale = ldm_patched.modules.model_management.cast_to_device(dora_scale, weight.device, torch.float32)
    lora_diff *= alpha
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = (
        weight_calc.transpose(0, 1)
        .reshape(weight_calc.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
        .transpose(0, 1)
    )

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight

def string_to_seed(s):
    """Convert a string into a seed integer by using a simple hash function."""
    if hasattr(s, 'encode'):
        s = s.encode()
    seed = 0
    for byte in s:
        if isinstance(byte, str):
            byte = ord(byte)
        seed = ((seed * 33) + byte) & 0xFFFFFFFF
    return seed


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options

def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = ldm_patched.modules.model_management.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone):
        if not self.is_clone(clone):
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_unet_function_wrapper(self, unet_wrapper_function: UnetWrapperFunction):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function

    def set_model_vae_regulation(self, vae_regulation):
        self.model_options["model_vae_regulation"] = vae_regulation

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return ldm_patched.modules.utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()
    
    import contextlib
    @contextlib.contextmanager
    def use_ejected(self, skip_and_inject_on_exit_only=False):
        was_injected = False
        prev_skip_injection = getattr(self, 'skip_injection', False)
        
        try:
            if skip_and_inject_on_exit_only:
                self.skip_injection = True
                
            if getattr(self, 'is_injected', False):
                if hasattr(self, 'eject_model'):
                    self.eject_model()
                was_injected = True
                
            yield
            
        finally:
            if skip_and_inject_on_exit_only:
                self.skip_injection = prev_skip_injection
                if hasattr(self, 'inject_model'):
                    self.inject_model()
                    
            if was_injected and not getattr(self, 'skip_injection', False):
                if hasattr(self, 'inject_model'):
                    self.inject_model()
                    
            self.skip_injection = prev_skip_injection

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            p = set()
            model_sd = self.model.state_dict()
            
            # Create mapping for keys with and without diffusion_model prefix
            key_mapping = {}
            needs_prefix = not any(k.startswith("diffusion_model.") for k in model_sd.keys())
            
            for k in model_sd.keys():
                # Map both with and without prefix
                if needs_prefix:
                    key_mapping[f"diffusion_model.{k}"] = k
                    key_mapping[k] = k
                else:
                    key_mapping[k] = k
                    if k.startswith("diffusion_model."):
                        key_mapping[k[len("diffusion_model."):]] = k

            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                # Try to find the key in our mapping
                actual_key = key_mapping.get(key)
                
                if actual_key is not None and actual_key in model_sd:
                    p.add(k)
                    current_patches = self.patches.get(actual_key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    self.patches[actual_key] = current_patches
                    # print(f"Successfully applied patch for key: {key} -> {actual_key}")
                else:
                    # print(f"Failed to find matching key in model: {key}")
                    pass

            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_key_patches(self, filter_prefix=None):
        ldm_patched.modules.model_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd
    
    def restore_original_model(self):
        patch_keys = []  # Initialize patch_keys at the start
        
        if hasattr(self.model, 'diffusion_model'):  # Check if it's UNet
            if hasattr(self.model.diffusion_model, '_orig_mod'):
                patch_keys = list(self.object_patches_backup.keys())
                for k in patch_keys:
                    if "diffusion_model." in k:
                        ldm_patched.modules.utils.set_attr(self.model.diffusion_model._orig_mod, k.replace('diffusion_model.', ''), self.object_patches_backup[k])
        elif hasattr(self.model, '_orig_mod'):  # Handle CLIP and other models
            patch_keys = list(self.object_patches_backup.keys())
            for k in patch_keys:
                ldm_patched.modules.utils.set_attr(self.model._orig_mod, k, self.object_patches_backup[k])
                
        return patch_keys

    def recompile_model(self, patch_keys=None):
        if patch_keys is None:
            return
            
        # Handle UNet compilation
        if hasattr(self.model, 'diffusion_model') and hasattr(self.model.diffusion_model, "compile_settings"):
            compile_settings = self.model.diffusion_model.compile_settings
            for k in patch_keys:
                if "diffusion_model." in k:
                    key = k.replace('diffusion_model.', '')
                    attributes = key.split('.')
                    block = self.model.diffusion_model._orig_mod
                    for attr in attributes[:-1]:
                        if attr.isdigit():
                            block = block[int(attr)]
                        else:
                            block = getattr(block, attr)

                    compiled_block = torch.compile(
                        block,
                        **compile_settings
                    )
                    # Set the compiled block back
                    parent = self.model.diffusion_model._orig_mod
                    for attr in attributes[:-1]:
                        if attr.isdigit():
                            parent = parent[int(attr)]
                        else:
                            parent = getattr(parent, attr)
                    setattr(parent, attributes[-1], compiled_block)
        
        # Handle CLIP and other models
        elif hasattr(self.model, "compile_settings"):
            compile_settings = self.model.compile_settings
            for k in patch_keys:
                attributes = k.split('.')
                block = self.model._orig_mod
                for attr in attributes[:-1]:
                    if attr.isdigit():
                        block = block[int(attr)]
                    else:
                        block = getattr(block, attr)
                
                compiled_block = torch.compile(
                    block,
                    **compile_settings
                )
                # Set the compiled block back
                parent = self.model._orig_mod
                for attr in attributes[:-1]:
                    if attr.isdigit():
                        parent = parent[int(attr)]
                    else:
                        parent = getattr(parent, attr)
                setattr(parent, attributes[-1], compiled_block)


    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return

        inplace_update = self.weight_inplace_update or inplace_update

        # Handle keys with _orig_mod
        if key.startswith('diffusion_model._orig_mod.'):
            model_key = key[len('diffusion_model._orig_mod.'):]
            actual_model = self.model.diffusion_model._orig_mod if hasattr(self.model, 'diffusion_model') else self.model._orig_mod
        else:
            model_key = key
            actual_model = self.model

        weight = ldm_patched.modules.utils.get_attr(actual_model, model_key)

        if key not in self.backup:
            # Store a copy of the weight tensor
            weight_copy = weight.to(device=self.offload_device, copy=inplace_update)
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                weight_copy, inplace_update)

        if device_to is not None:
            temp_weight = ldm_patched.modules.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)

        out_weight = self.calculate_weight(self.patches[key], temp_weight, key)
        out_weight = ldm_patched.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))

        if inplace_update:
            ldm_patched.modules.utils.copy_to_param(actual_model, model_key, out_weight)
        else:
            ldm_patched.modules.utils.set_attr_param(actual_model, model_key, out_weight)

    def patch_model(self, device_to=None, patch_weights=True):
        # First restore original model if needed
        patch_keys = self.restore_original_model()

        for k in self.object_patches:
            old = ldm_patched.modules.utils.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            if hasattr(self.model, '_orig_mod'):
                model_sd = self.model._orig_mod.state_dict()
            else:
                model_sd = self.model.state_dict()

            for key in self.patches:
                if key not in model_sd:
                    print(f"Warning: could not patch. key doesn't exist in model: {key}")
                    continue
                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        # Recompile model parts if needed
        self.recompile_model(patch_keys)
        return self.model

    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        self.patch_model(device_to, patch_weights=False)

        logging.info("loading in lowvram mode {}".format(lowvram_model_memory/(1024 * 1024)))
        class LowVramPatch:
            def __init__(self, key, model_patcher):
                self.key = key
                self.model_patcher = model_patcher
            def __call__(self, weight):
                return self.model_patcher.calculate_weight(self.model_patcher.patches[self.key], weight, self.key)

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = ldm_patched.modules.model_management.module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += ldm_patched.modules.model_management.module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            strength = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key), )

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if strength != 0.0:
                    if w1.shape != weight.shape:
                        logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += strength * ldm_patched.modules.model_management.cast_to_device(w1, weight.device, weight.dtype)
            elif patch_type == "lora": #lora/locon
                mat1 = ldm_patched.modules.model_management.cast_to_device(v[0], weight.device, torch.float32)
                mat2 = ldm_patched.modules.model_management.cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                if v[2] is not None:
                    alpha = v[2] / mat2.shape[0]
                else:
                    alpha = 1.0

                if v[3] is not None:
                    #locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = ldm_patched.modules.model_management.cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1_a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w1_b, weight.device, torch.float32))
                else:
                    w1 = ldm_patched.modules.model_management.cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32),
                                          ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32),
                                          ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32))
                else:
                    w2 = ldm_patched.modules.model_management.cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha = v[2] / dim
                else:
                    alpha = 1.0

                try:
                    lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha = v[2] / w1b.shape[0]
                else:
                    alpha = 1.0

                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None: #cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t1, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32))

                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32))
                else:
                    m1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32))
                    m2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32))

                try:
                    lora_diff = (m1 * m2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha = v[4] / v[0].shape[0]
                else:
                    alpha = 1.0

                dora_scale = v[5]

                a1 = ldm_patched.modules.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, torch.float32)
                a2 = ldm_patched.modules.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, torch.float32)
                b1 = ldm_patched.modules.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, torch.float32)
                b2 = ldm_patched.modules.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, torch.float32)

                try:
                    lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logging.error("ERROR {} {} {}".format(patch_type, key, e))
            else:
                logging.warning("patch type not recognized {} {}".format(patch_type, key))

        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            if self.model_lowvram:
                for m in self.model.modules():
                    if hasattr(m, "prev_ldm_patched_cast_weights"):
                        m.ldm_patched_cast_weights = m.prev_ldm_patched_cast_weights
                        del m.prev_ldm_patched_cast_weights
                    m.weight_function = None
                    m.bias_function = None

                self.model_lowvram = False
                self.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            if self.weight_inplace_update:
                for k in keys:
                    # Get the actual weight from the Dimension namedtuple 
                    bk = self.backup[k]
                    weight = bk.weight if hasattr(bk, 'weight') else bk
                    
                    # Handle _orig_mod prefix
                    if k.startswith('diffusion_model._orig_mod.'):
                        model_key = k[len('diffusion_model._orig_mod.'):]
                        actual_model = self.model.diffusion_model._orig_mod if hasattr(self.model.diffusion_model, '_orig_mod') else self.model.diffusion_model
                        ldm_patched.modules.utils.copy_to_param(actual_model, model_key, weight)
                    else:
                        ldm_patched.modules.utils.copy_to_param(self.model, k, weight)
            else:
                for k in keys:
                    # Get the actual weight from the Dimension namedtuple
                    bk = self.backup[k]
                    weight = bk.weight if hasattr(bk, 'weight') else bk

                    # Handle _orig_mod prefix
                    if k.startswith('diffusion_model._orig_mod.'):
                        model_key = k[len('diffusion_model._orig_mod.'):]
                        actual_model = self.model.diffusion_model._orig_mod if hasattr(self.model.diffusion_model, '_orig_mod') else self.model.diffusion_model
                        ldm_patched.modules.utils.set_attr_param(actual_model, model_key, weight)
                    else:
                        ldm_patched.modules.utils.set_attr_param(self.model, k, weight)

            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            ldm_patched.modules.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()