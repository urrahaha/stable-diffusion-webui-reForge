import torch
import contextlib

from ldm_patched.modules import model_management
from ldm_patched.modules import model_detection

from ldm_patched.modules.sd import VAE, CLIP, load_model_weights
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.clip_vision

from omegaconf import OmegaConf
from modules.sd_models_config import find_checkpoint_config
from modules.shared import cmd_opts
from modules import sd_hijack
from modules.sd_models_xl import extend_sdxl
from ldm.util import instantiate_from_config
from modules_forge import forge_clip
from modules_forge.unet_patcher import UnetPatcher
from ldm_patched.modules.model_base import model_sampling, ModelType, SD3
import logging
import types

import open_clip
from transformers import CLIPTextModel, CLIPTokenizer


class FakeObject:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.visual = None
        return

    def eval(self, *args, **kwargs):
        return self

    def parameters(self, *args, **kwargs):
        return []


class ForgeSD:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeSD(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


@contextlib.contextmanager
def no_clip():
    backup_openclip = open_clip.create_model_and_transforms
    backup_CLIPTextModel = CLIPTextModel.from_pretrained
    backup_CLIPTokenizer = CLIPTokenizer.from_pretrained

    try:
        open_clip.create_model_and_transforms = lambda *args, **kwargs: (FakeObject(), None, None)
        CLIPTextModel.from_pretrained = lambda *args, **kwargs: FakeObject()
        CLIPTokenizer.from_pretrained = lambda *args, **kwargs: FakeObject()
        yield

    finally:
        open_clip.create_model_and_transforms = backup_openclip
        CLIPTextModel.from_pretrained = backup_CLIPTextModel
        CLIPTokenizer.from_pretrained = backup_CLIPTokenizer
    return

# In forge_loader.py

from ldm_patched.modules import utils

def load_checkpoint_guess_config(state_dict, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True):
    sd_keys = state_dict.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(state_dict)
    parameters = utils.calculate_parameters(state_dict, diffusion_model_prefix)
    weight_dtype = ldm_patched.modules.utils.weight_dtype(state_dict, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(state_dict, diffusion_model_prefix)
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type")

    unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=[weight_dtype] + model_config.supported_inference_dtypes)
    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None and output_clipvision:
        clipvision = ldm_patched.modules.clip_vision.load_clipvision_from_sd(state_dict, model_config.clip_vision_prefix, True)

    if output_model:
        initial_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        offload_device = model_management.unet_offload_device()
        
        if isinstance(model_config, SD3):
            model = SD3(model_config, model_type=ModelType.EPS, device=initial_load_device)
        else:
            model = model_config.get_model(state_dict, diffusion_model_prefix, device=initial_load_device)
        
        model.load_model_weights(state_dict, diffusion_model_prefix)

    if output_vae:
        vae_sd = utils.state_dict_prefix_replace(state_dict, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target(state_dict=state_dict)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(state_dict)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))
                if len(u) > 0:
                    logging.debug("clip unexpected: {}".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = state_dict.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = UnetPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device(), current_device=initial_load_device)
        if initial_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            model_management.load_model_gpu(model_patcher)

    return ForgeSD(model_patcher, clip, vae, clipvision)


@torch.no_grad()
def load_model_for_a1111(timer, checkpoint_info=None, state_dict=None):
    is_sd3 = 'model.diffusion_model.x_embedder.proj.weight' in state_dict
    timer.record("forge solving config")
    
    if not is_sd3:
        a1111_config_filename = find_checkpoint_config(state_dict, checkpoint_info)
        a1111_config = OmegaConf.load(a1111_config_filename)
        if hasattr(a1111_config.model.params, 'network_config'):
            a1111_config.model.params.network_config.target = 'modules_forge.forge_loader.FakeObject'
        if hasattr(a1111_config.model.params, 'unet_config'):
            a1111_config.model.params.unet_config.target = 'modules_forge.forge_loader.FakeObject'
        if hasattr(a1111_config.model.params, 'first_stage_config'):
            a1111_config.model.params.first_stage_config.target = 'modules_forge.forge_loader.FakeObject'
        with no_clip():
            sd_model = instantiate_from_config(a1111_config.model)
    else:
        sd_model = torch.nn.Module()  # Create a base module for SD3
    
    timer.record("forge instantiate config")
    
    forge_objects = load_checkpoint_guess_config(
        state_dict,
        output_vae=True,
        output_clip=True,
        output_clipvision=True,
        embedding_directory=cmd_opts.embeddings_dir,
        output_model=True
    )
    
    if is_sd3:
        class SD3(torch.nn.Module):
            def __init__(self, unet_model, vae_model, clip_model, forge_objects):
                super().__init__()
                self.model = unet_model
                self.first_stage_model = vae_model
                self.cond_stage_model = clip_model
                self.forge_objects = forge_objects
                self.diffusion_model = self.model.diffusion_model
                self.cond_stage_key = "edit"
                self.parameterization = "v"
                self.conditioning_key = "crossattn"
                
                # Add these lines
                self.log_sigmas = self.model.log_sigmas if hasattr(self.model, 'log_sigmas') else None
                self.sigmas = self.model.sigmas if hasattr(self.model, 'sigmas') else None

            def get_learned_conditioning(self, text_prompts):
                if isinstance(text_prompts, list):
                    return [self.cond_stage_model(prompt) for prompt in text_prompts]
                else:
                    return self.cond_stage_model(text_prompts)

            def get_first_stage_encoding(self, x):
                return x

            def create_denoiser(self):
                return self.model

        # Create the SD3 model
        sd_model = SD3(
            unet_model=forge_objects.unet.model,
            vae_model=forge_objects.vae.first_stage_model,
            clip_model=forge_clip.CLIP_SD3(forge_objects.clip, sd_hijack.model_hijack),
            forge_objects=forge_objects
        )

        # Add these lines after moving the model to the device
        # if sd_model.log_sigmas is None:
        #     sd_model.log_sigmas = torch.linspace(sd_model.model.log_sigma_min, sd_model.model.log_sigma_max, sd_model.model.num_timesteps).to(device)
        # if sd_model.sigmas is None:
        #     sd_model.sigmas = torch.exp(sd_model.log_sigmas)

        # Move the model to the appropriate device
        device = next(sd_model.model.parameters()).device
        sd_model.model = sd_model.model.to(device)
        sd_model.first_stage_model = sd_model.first_stage_model.to(device)
        sd_model.cond_stage_model = sd_model.cond_stage_model.to(device)

        sd_model.register_schedule = lambda *args, **kwargs: None
        sd_model.register_buffer = lambda *args, **kwargs: None
        sd_model.model_sampling = model_sampling(sd_model.model.model_config, ModelType.V_PREDICTION)

        # Set additional attributes
        sd_model.is_sd3 = True
        sd_model.latent_channels = 16
    else:
        sd_model.first_stage_model = forge_objects.vae.first_stage_model
        sd_model.model.diffusion_model = forge_objects.unet.model.diffusion_model
    
    sd_model.forge_objects = forge_objects
    sd_model.forge_objects_original = forge_objects.shallow_copy()
    sd_model.forge_objects_after_applying_lora = forge_objects.shallow_copy()
    timer.record("forge load real models")
    
    conditioner = getattr(sd_model, 'conditioner', None)
    if is_sd3:
        sd_model.cond_stage_model = forge_clip.CLIP_SD3(forge_objects.clip, sd_hijack.model_hijack)
    elif conditioner:
        text_cond_models = []
        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__
            if typename == 'FrozenCLIPEmbedder':  # SDXL Clip L
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
                    model_embeddings.token_embedding, sd_hijack.model_hijack)
                embedder = forge_clip.CLIP_SD_XL_L(embedder, sd_hijack.model_hijack)
                conditioner.embedders[i] = embedder
                text_cond_models.append(embedder)
            elif typename == 'FrozenOpenCLIPEmbedder2':  # SDXL Clip G
                embedder.tokenizer = forge_objects.clip.tokenizer.clip_g.tokenizer
                embedder.transformer = forge_objects.clip.cond_stage_model.clip_g.transformer
                embedder.text_projection = forge_objects.clip.cond_stage_model.clip_g.text_projection
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
                    model_embeddings.token_embedding, sd_hijack.model_hijack, textual_inversion_key='clip_g')
                embedder = forge_clip.CLIP_SD_XL_G(embedder, sd_hijack.model_hijack)
                conditioner.embedders[i] = embedder
                text_cond_models.append(embedder)
        if len(text_cond_models) == 1:
            sd_model.cond_stage_model = text_cond_models[0]
        else:
            sd_model.cond_stage_model = conditioner
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenCLIPEmbedder':  # SD15 Clip
        sd_model.cond_stage_model.tokenizer = forge_objects.clip.tokenizer.clip_l.tokenizer
        sd_model.cond_stage_model.transformer = forge_objects.clip.cond_stage_model.clip_l.transformer
        model_embeddings = sd_model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
            model_embeddings.token_embedding, sd_hijack.model_hijack)
        sd_model.cond_stage_model = forge_clip.CLIP_SD_15_L(sd_model.cond_stage_model, sd_hijack.model_hijack)
    elif type(sd_model.cond_stage_model).__name__ == 'FrozenOpenCLIPEmbedder':  # SD21 Clip
        sd_model.cond_stage_model.tokenizer = forge_objects.clip.tokenizer.clip_h.tokenizer
        sd_model.cond_stage_model.transformer = forge_objects.clip.cond_stage_model.clip_h.transformer
        model_embeddings = sd_model.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = sd_hijack.EmbeddingsWithFixes(
            model_embeddings.token_embedding, sd_hijack.model_hijack)
        sd_model.cond_stage_model = forge_clip.CLIP_SD_21_H(sd_model.cond_stage_model, sd_hijack.model_hijack)
    else:
        raise NotImplementedError('Bad Clip Class Name:' + type(sd_model.cond_stage_model).__name__)

    timer.record("forge set components")
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")
    
    if getattr(sd_model, 'parameterization', None) == 'v':
        sd_model.forge_objects.unet.model.model_sampling = model_sampling(sd_model.forge_objects.unet.model.model_config, ModelType.V_PREDICTION)
    
    sd_model.is_sd3 = is_sd3
    sd_model.latent_channels = 16 if is_sd3 else 4
    sd_model.is_sdxl = conditioner is not None and not is_sd3
    sd_model.is_sdxl_inpaint = sd_model.is_sdxl and forge_objects.unet.model.diffusion_model.in_channels == 9
    sd_model.is_sd2 = not sd_model.is_sdxl and not is_sd3 and hasattr(sd_model.cond_stage_model, 'model')
    sd_model.is_sd1 = not sd_model.is_sdxl and not sd_model.is_sd2 and not is_sd3
    sd_model.is_ssd = sd_model.is_sdxl and 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in sd_model.state_dict().keys()
    
    if sd_model.is_sdxl:
        extend_sdxl(sd_model)
    
    sd_model.sd_model_hash = sd_model_hash
    sd_model.sd_model_checkpoint = checkpoint_info.filename
    sd_model.sd_checkpoint_info = checkpoint_info

    @torch.inference_mode()
    def patched_decode_first_stage(x):
        if sd_model.is_sd3:
            sample = x
        else:
            sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_out(x)
        sample = sd_model.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    @torch.inference_mode()
    def patched_encode_first_stage(x):
        sample = sd_model.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        if not sd_model.is_sd3:
            sample = sd_model.forge_objects.unet.model.model_config.latent_format.process_in(sample)
        return sample.to(x)

    sd_model.ema_scope = lambda *args, **kwargs: contextlib.nullcontext()
    sd_model.get_first_stage_encoding = lambda x: x
    sd_model.decode_first_stage = patched_decode_first_stage
    sd_model.encode_first_stage = patched_encode_first_stage
    sd_model.clip = sd_model.cond_stage_model
    sd_model.tiling_enabled = False
    timer.record("forge finalize")
    sd_model.current_lora_hash = str([])
    return sd_model
