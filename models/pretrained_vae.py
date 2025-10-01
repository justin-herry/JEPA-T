import os

import torch.nn as nn

VAE = {
    "KL16_MAR": {
        "vae_embed_dim": 16,
        "ckpt_path": "vae/kl16.ckpt",
        "kwargs": dict(embed_dim=16, ch_mult=(1, 1, 2, 2, 4)),
        "scaling_factor": 0.2325,
        "shift": 0.0,
        "vae_stride": 16
    }
}


def get_pretrained_vae(args):
    vae_dict = VAE[args.vae]
    if args.use_cached and not args.online_eval and not args.evaluate:
        vae = nn.Identity()
    else:
        # Used for loading KL16 from MAR.
        ckpt_path = os.path.join(
            args.local_diffusers_model_root, vae_dict["ckpt_path"])
        from models.vae import AutoencoderKL
        vae = AutoencoderKL(ckpt_path=ckpt_path, **
                            vae_dict["kwargs"]).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    args.vae_embed_dim = vae_dict["vae_embed_dim"]
    args.scaling_factor = vae_dict['scaling_factor']
    args.shift = vae_dict["shift"]
    args.vae_stride = vae_dict["vae_stride"]

    return vae
