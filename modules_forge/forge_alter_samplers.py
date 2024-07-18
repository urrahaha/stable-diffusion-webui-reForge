import torch
from modules import sd_samplers_kdiffusion, sd_samplers_common

from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import calculate_sigmas_scheduler
from ldm_patched.k_diffusion import deis
import ldm_patched.modules.model_patcher


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, scheduler_name):
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.unet = sd_model.forge_objects.unet
        self.model = sd_model
        if sampler_name == 'ddpm':
            sampler_function = k_diffusion_sampling.sample_ddpm
        elif sampler_name == 'heunpp2':
            sampler_function = k_diffusion_sampling.sample_heunpp2
        elif sampler_name == 'ipndm':
            sampler_function = k_diffusion_sampling.sample_ipndm
        elif sampler_name == 'ipndm_v':
            sampler_function = k_diffusion_sampling.sample_ipndm_v
        elif sampler_name == 'deis':
            sampler_function = k_diffusion_sampling.sample_deis
        elif sampler_name == 'euler_cfg_pp':
            sampler_function = k_diffusion_sampling.sample_euler_cfg_pp
        elif sampler_name == 'euler_ancestral_cfg_pp':
            sampler_function = k_diffusion_sampling.sample_euler_ancestral_cfg_pp
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        super().__init__(sampler_function, sd_model, None)

    def get_sigmas(self, p, steps):
        if self.scheduler_name == 'turbo':
            timesteps = torch.flip(torch.arange(1, steps + 1) * float(1000.0 / steps) - 1, (0,)).round().long().clip(0, 999)
            sigmas = self.unet.model.model_sampling.sigma(timesteps)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        else:
            sigmas = calculate_sigmas_scheduler(self.unet.model, self.scheduler_name, steps, is_sdxl=getattr(self.model, "is_sdxl", False))
        return sigmas.to(self.unet.load_device)


def build_constructor(sampler_name, scheduler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name, scheduler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm', scheduler_name='normal'), ['ddpm'], {}),
    sd_samplers_common.SamplerData('HeunPP2', build_constructor(sampler_name='heunpp2', scheduler_name='normal'), ['heunpp2'], {}),
    sd_samplers_common.SamplerData('IPNDM', build_constructor(sampler_name='ipndm', scheduler_name='normal'), ['ipndm'], {}),
    sd_samplers_common.SamplerData('IPNDM_V', build_constructor(sampler_name='ipndm_v', scheduler_name='normal'), ['ipndm_v'], {}),
    sd_samplers_common.SamplerData('DEIS', build_constructor(sampler_name='deis', scheduler_name='normal'), ['deis'], {}),
    # sd_samplers_common.SamplerData('Euler CFG++', build_constructor(sampler_name='euler_cfg_pp', scheduler_name='normal'), ['euler_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('Euler Ancestral CFG++', build_constructor(sampler_name='euler_ancestral_cfg_pp', scheduler_name='normal'), ['euler_ancestral_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DDPM Karras', build_constructor(sampler_name='ddpm', scheduler_name='karras'), ['ddpm_karras'], {}),
    # sd_samplers_common.SamplerData('Euler AYS', build_constructor(sampler_name='euler', scheduler_name='ays'), ['euler_ays'], {}),
    # sd_samplers_common.SamplerData('Euler A Turbo', build_constructor(sampler_name='euler_ancestral', scheduler_name='turbo'), ['euler_ancestral_turbo'], {}),
    # sd_samplers_common.SamplerData('Euler A AYS', build_constructor(sampler_name='euler_ancestral', scheduler_name='ays'), ['euler_ancestral_ays'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M Turbo', build_constructor(sampler_name='dpmpp_2m', scheduler_name='turbo'), ['dpmpp_2m_turbo'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M AYS', build_constructor(sampler_name='dpmpp_2m', scheduler_name='ays'), ['dpmpp_2m_ays'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE Turbo', build_constructor(sampler_name='dpmpp_2m_sde', scheduler_name='turbo'), ['dpmpp_2m_sde_turbo'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE AYS', build_constructor(sampler_name='dpmpp_2m_sde', scheduler_name='ays'), ['dpmpp_2m_sde_ays'], {}),
    # sd_samplers_common.SamplerData('LCM Karras', build_constructor(sampler_name='lcm', scheduler_name='karras'), ['lcm_karras'], {}),
    # sd_samplers_common.SamplerData('Euler SGMUniform', build_constructor(sampler_name='euler', scheduler_name='sgm_uniform'), ['euler_sgm_uniform'], {}),
    # sd_samplers_common.SamplerData('Euler A SGMUniform', build_constructor(sampler_name='euler_ancestral', scheduler_name='sgm_uniform'), ['euler_ancestral_sgm_uniform'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SGMUniform', build_constructor(sampler_name='dpmpp_2m', scheduler_name='sgm_uniform'), ['dpmpp_2m_sgm_uniform'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE SGMUniform', build_constructor(sampler_name='dpmpp_2m_sde', scheduler_name='sgm_uniform'), ['dpmpp_2m_sde_sgm_uniform'], {}),
]
