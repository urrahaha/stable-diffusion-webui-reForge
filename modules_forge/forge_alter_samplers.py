import torch
from modules import sd_samplers_kdiffusion, sd_samplers_common

from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import calculate_sigmas
from modules import shared


ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}
ALL_SOLVERS = list(ADAPTIVE_SOLVERS | FIXED_SOLVERS)
ALL_SOLVERS.sort()

class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, scheduler_name, solver=None, rtol=None, atol=None):
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.unet = sd_model.forge_objects.unet
        self.model = sd_model
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
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
            'ode_bosh3': self.sample_ode_bosh3,
            'ode_fehlberg2': self.sample_ode_fehlberg2,
            'ode_adaptive_heun': self.sample_ode_adaptive_heun,
            'ode_dopri5': self.sample_ode_dopri5,
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

    def sample_ode_bosh3(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="bosh3", 
                                               rtol=10**shared.opts.ode_bosh3_rtol, 
                                               atol=10**shared.opts.ode_bosh3_atol, 
                                               max_steps=shared.opts.ode_bosh3_max_steps)

    def sample_ode_fehlberg2(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="fehlberg2", 
                                               rtol=10**shared.opts.ode_fehlberg2_rtol, 
                                               atol=10**shared.opts.ode_fehlberg2_atol, 
                                               max_steps=shared.opts.ode_fehlberg2_max_steps)

    def sample_ode_adaptive_heun(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="adaptive_heun", 
                                               rtol=10**shared.opts.ode_adaptive_heun_rtol, 
                                               atol=10**shared.opts.ode_adaptive_heun_atol, 
                                               max_steps=shared.opts.ode_adaptive_heun_max_steps)

    def sample_ode_dopri5(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver="dopri5", 
                                               rtol=10**shared.opts.ode_dopri5_rtol, 
                                               atol=10**shared.opts.ode_dopri5_atol, 
                                               max_steps=shared.opts.ode_dopri5_max_steps)
    
    def sample_ode_custom(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        solver = shared.opts.ode_custom_solver
        rtol = 10**shared.opts.ode_custom_rtol if solver in ADAPTIVE_SOLVERS else None
        atol = 10**shared.opts.ode_custom_atol if solver in ADAPTIVE_SOLVERS else None
        max_steps = shared.opts.ode_custom_max_steps
        
        return k_diffusion_sampling.sample_ode(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                               solver=solver, rtol=rtol, atol=atol, max_steps=max_steps)



    def get_sigmas(self, p, steps):
        if self.scheduler_name == 'turbo':
            timesteps = torch.flip(torch.arange(1, steps + 1) * float(1000.0 / steps) - 1, (0,)).round().long().clip(0, 999)
            sigmas = self.unet.model.model_sampling.sigma(timesteps)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        else:
            sigmas = calculate_sigmas(self.unet.model.model_sampling, self.scheduler_name, steps)
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
    sd_samplers_common.SamplerData('ODE (Bosh3)', build_constructor(sampler_name='ode_bosh3', scheduler_name='normal'), ['ode_bosh3'], {}),
    sd_samplers_common.SamplerData('ODE (Fehlberg2)', build_constructor(sampler_name='ode_fehlberg2', scheduler_name='normal'), ['ode_fehlberg2'], {}),
    sd_samplers_common.SamplerData('ODE (Adaptive Heun)', build_constructor(sampler_name='ode_adaptive_heun', scheduler_name='normal'), ['ode_adaptive_heun'], {}),
    sd_samplers_common.SamplerData('ODE (Dopri5)', build_constructor(sampler_name='ode_dopri5', scheduler_name='normal'), ['ode_dopri5'], {}),
    sd_samplers_common.SamplerData('ODE Custom', build_constructor(sampler_name='ode_custom', scheduler_name='normal'), ['ode_custom'], {}),
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
