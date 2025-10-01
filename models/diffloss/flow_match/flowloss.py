import torch
import torch.nn as nn

from models.denoising_mlp import SimpleMLPAdaLN

from .flow_match import ODE, training_losses


class FlowMatchLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(FlowMatchLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        self.num_sampling_steps = int(num_sampling_steps)

    def forward(self, target, z, mask=None):
        loss = training_losses(self.net, target, model_kwargs=dict(c=z))
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, time_shifting_factor=1.0, cfg_scale=1.0, **kwargs):
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

        samples = ODE(self.num_sampling_steps, time_shifting_factor=time_shifting_factor).sample(
            noise, sample_fn, **model_kwargs
        )[-1]

        return samples
