import torch
import torch.nn as nn

from models.denoising_mlp import SimpleMLPAdaLN

from .gaussian_diffusion import create_diffusion


class GaussDiffLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(GaussDiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.train_diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(
            timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps,
                          (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg_scale=1.0, **kwargs):
        # diffusion loss sampling
        if not cfg_scale == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg_scale)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent
