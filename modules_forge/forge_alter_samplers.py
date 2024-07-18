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
        
        sampler_functions = {
            'ddpm': k_diffusion_sampling.sample_ddpm,
            'heunpp2': k_diffusion_sampling.sample_heunpp2,
            'ipndm': k_diffusion_sampling.sample_ipndm,
            'ipndm_v': k_diffusion_sampling.sample_ipndm_v,
            'deis': k_diffusion_sampling.sample_deis,
            'euler_cfg_pp': k_diffusion_sampling.sample_euler_cfg_pp,
            'euler_ancestral_cfg_pp': k_diffusion_sampling.sample_euler_ancestral_cfg_pp,
            'dpmpp_2s_ancestral_cfg_pp': k_diffusion_sampling.sample_dpmpp_2s_ancestral_cfg_pp,
            'dpmpp_sde_cfg_pp': k_diffusion_sampling.sample_dpmpp_sde_cfg_pp,
            'dpmpp_2m_cfg_pp': k_diffusion_sampling.sample_dpmpp_2m_cfg_pp,
            # 'dpmpp_2m_sde_cfg_pp': k_diffusion_sampling.sample_dpmpp_2m_sde_cfg_pp,
            # 'dpmpp_3m_sde_cfg_pp': k_diffusion_sampling.sample_dpmpp_3m_sde_cfg_pp,
            # 'ddpm_cfg_pp': k_diffusion_sampling.sample_ddpm_cfg_pp,
            # 'lcm_cfg_pp': k_diffusion_sampling.sample_lcm_cfg_pp,
            # 'heunpp2_cfg_pp': k_diffusion_sampling.sample_heunpp2_cfg_pp,
            # 'ipndm_cfg_pp': k_diffusion_sampling.sample_ipndm_cfg_pp,
            # 'ipndm_v_cfg_pp': k_diffusion_sampling.sample_ipndm_v_cfg_pp,
            # 'deis_cfg_pp': k_diffusion_sampling.sample_deis_cfg_pp,
        }
        
        sampler_function = sampler_functions.get(sampler_name)
        if sampler_function is None:
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
    sd_samplers_common.SamplerData('Euler CFG++', build_constructor(sampler_name='euler_cfg_pp', scheduler_name='normal'), ['euler_cfg_pp'], {}),
    sd_samplers_common.SamplerData('Euler Ancestral CFG++', build_constructor(sampler_name='euler_ancestral_cfg_pp', scheduler_name='normal'), ['euler_ancestral_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral CFG++', build_constructor(sampler_name='dpmpp_2s_ancestral_cfg_pp', scheduler_name='normal'), ['dpmpp_2s_ancestral_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ SDE CFG++', build_constructor(sampler_name='dpmpp_sde_cfg_pp', scheduler_name='normal'), ['dpmpp_sde_cfg_pp'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M CFG++', build_constructor(sampler_name='dpmpp_2m_cfg_pp', scheduler_name='normal'), ['dpmpp_2m_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE CFG++', build_constructor(sampler_name='dpmpp_2m_sde_cfg_pp', scheduler_name='normal'), ['dpmpp_2m_sde_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DPM++ 3M SDE CFG++', build_constructor(sampler_name='dpmpp_3m_sde_cfg_pp', scheduler_name='normal'), ['dpmpp_3m_sde_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DDPM CFG++', build_constructor(sampler_name='ddpm_cfg_pp', scheduler_name='normal'), ['ddpm_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('LCM CFG++', build_constructor(sampler_name='lcm_cfg_pp', scheduler_name='normal'), ['lcm_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('HeunPP2 CFG++', build_constructor(sampler_name='heunpp2_cfg_pp', scheduler_name='normal'), ['heunpp2_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('IPNDM CFG++', build_constructor(sampler_name='ipndm_cfg_pp', scheduler_name='normal'), ['ipndm_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('IPNDM_V CFG++', build_constructor(sampler_name='ipndm_v_cfg_pp', scheduler_name='normal'), ['ipndm_v_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DEIS CFG++', build_constructor(sampler_name='deis_cfg_pp', scheduler_name='normal'), ['deis_cfg_pp'], {}),
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
