import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from models.block import Block, LayerNorm
from models.diffloss import FlowMatchLoss, GaussDiffLoss
from models.jepaloss import JepaLoss
import clip


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class JEPAT(nn.Module):
    no_weight_decay_set = {
        'diffloss', 'jepaloss', 'buffer', "head", "class_emb", "fake_latent", "z_proj",
        "encoder_pos_embed_learned", "decoder_embed", "mask_token", "decoder_pos_embed_learned",
        "clip"
    }

    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4, norm_layer=LayerNorm,
                 qk_norm: bool = False,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,

                 diffloss: str = "gauss",  # gauss, flow, none
                 jepaloss: str = "jepa",  # jepa, none
                 grad_checkpointing: bool = False,
                 **ignored_kwargs,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.buffer_size = buffer_size
        self.grad_checkpointing = grad_checkpointing

        self.cross_attention = nn.MultiheadAttention(
        embed_dim=decoder_embed_dim,  
        num_heads=8,  
        batch_first=True,
        dropout=attn_dropout
        )

        # --------------------------------------------------------------------------
        # CLIP for Class Embedding
        self.num_classes = class_num
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cuda')
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # CLIP output is 512, project to encoder_embed_dim
        self.class_emb = nn.Sequential(
            nn.Linear(512, encoder_embed_dim),
            nn.LayerNorm(encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim)
        )
        
        self.label_drop_prob = label_drop_prob
        self.fuse_proj = nn.Linear(self.decoder_embed_dim + self.encoder_embed_dim,
                           self.decoder_embed_dim, bias=True)
        # Fake class embedding for cfg_scale's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # Define knowledge buffer here and append it in  encoder.
        self.buffer = nn.Parameter(torch.zeros(
            1, self.buffer_size, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT encoder specifics (used for classification and diffusion.)
        self.z_proj = nn.Linear(self.token_embed_dim,
                                encoder_embed_dim, bias=True)
        self.z_proj_ln = norm_layer(encoder_embed_dim)

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim,
                  encoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  qk_norm=qk_norm,
                  proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Positional Embedding for decoder
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT decoder specifics (no requires for classification task.)
        self.decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim,
                  decoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  qk_norm=qk_norm,
                  proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # --------------------------------------------------------------------------
        # Positional Embedding for encoder
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim))

        # NOTE: initialize weights before diffusion loss!
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss (only requires for diffusion learning.)

        self.diffloss = {
            "gauss": GaussDiffLoss,
            "flow": FlowMatchLoss,
            "none": nn.Identity,
        }[diffloss](
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self.jepaloss = {
            "jepa": JepaLoss,
            "none": nn.Identity,
        }[jepaloss](
            decoder_embed_dim, encoder_embed_dim, norm_layer, beta=2.0)

    def count_params(self):
        def count_param(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params = count_param(self)
        encoder_params = count_param(self.encoder_blocks)
        decoder_params = count_param(self.decoder_blocks)
        denoising_params = count_param(self.diffloss)
        return {
            "# total": total_params,
            "# encoder": encoder_params,
            "# decoder": decoder_params,
            "# denoising": denoising_params
        }

    def initialize_weights(self):
        # apply default init first.
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # parameters
        # Initialize class_emb projection layers
        for m in self.class_emb.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                    
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, "encoder_pos_embed_learned"):
            torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        if hasattr(self, "decoder_pos_embed_learned"):
            torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # init for head weights
        if hasattr(self, "head"):
            torch.nn.init.trunc_normal_(self.head.weight, std=0.01)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        p = self.patch_size

        x = F.unfold(x, kernel_size=(p, p), stride=p)
        x = x.transpose(1, 2)

        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.transpose(1, 2).reshape(bsz, c * p * p, h_ * w_)
        x = F.fold(x, output_size=(h_ * p, w_ * p),
                   kernel_size=(p, p), stride=p)

        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = max(0, int(np.ceil(seq_len * mask_rate)))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_encoder(self, x, mask, class_embedding):
        """
        Returns: 
            feature: batch size, seq_len, dims
        """
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = x + self.encoder_pos_embed_learned

        # append buffer
        buffer = self.buffer + class_embedding.unsqueeze(1)
        x = torch.cat([buffer.type_as(x), x], dim=1)

        mask_with_buffer = torch.cat(
            [torch.zeros(bsz, self.buffer_size, device=x.device), mask], dim=1)

        # dropping
        indices = (1-mask_with_buffer.float()).nonzero(as_tuple=True)
        x = x[indices].reshape(bsz, -1, embed_dim)

        tokens = x.shape[1]
        freqs_cis = None

        # encoder position embedding
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for blk in self.encoder_blocks:
                # pad None to support checkpoint
                x = checkpoint(blk, x, None, None, freqs_cis)
        else:
            for blk in self.encoder_blocks:
                x = blk(x, freqs_cis=freqs_cis)
        x = self.encoder_norm(x)

        return x

    def forward_decoder(self, x, mask, class_embedding=None):
        bsz = len(x)
        x = self.decoder_embed(x)

        # If class_embedding is provided, add it to the decoder input
        if class_embedding is not None:
        # Expand class_embedding to match the sequence length
            class_embedding = class_embedding.unsqueeze(1).expand(bsz, x.size(1), -1)
            x = x + class_embedding

        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        x_after_pad = self.mask_token.repeat(
            bsz, self.buffer_size + self.seq_len, 1).type_as(x).clone()
        indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        x_after_pad[indices] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    
        x_after_pad[:, self.buffer_size:] += self.decoder_pos_embed_learned
        freqs_cis = None

        # decoder position embedding
        x = x_after_pad

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for blk in self.decoder_blocks:
                # pad None to support checkpoint
                x = checkpoint(blk, x, None, None, freqs_cis)
        else:
            for blk in self.decoder_blocks:
                x = blk(x, freqs_cis=freqs_cis)
        x = self.decoder_norm(x)

        # drop buffer size.
        x = x[:, self.buffer_size:] #[32,320,768]
        return x

    def diffusion_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(
            bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def jepa_loss(self, x, ema_x, mask):
        """
        ema_x: b, n, c
        x: b, n, c
        mask: b, n
        """
        return self.jepaloss(x, ema_x, mask)

    def encode_labels_with_clip(self, labels):
        """
        Encode labels using CLIP model
        labels can be either:
        - torch.Tensor of integers (class indices)
        - list of strings (text descriptions)
        """
        with torch.no_grad():
            if isinstance(labels, torch.Tensor):
                # If labels are integers, convert to text
                text_labels = [f"a photo of class {int(label)}" for label in labels]
                text_tokens = clip.tokenize(text_labels).cuda()
                clip_features = self.clip_model.encode_text(text_tokens).float()
            elif isinstance(labels, list) and isinstance(labels[0], str):
                # If labels are already text
                text_tokens = clip.tokenize(labels).cuda()
                clip_features = self.clip_model.encode_text(text_tokens).float()
            else:
                # Fallback: assume labels are tensors that can be converted to int
                text_labels = [f"a photo of class {int(label)}" for label in labels]
                text_tokens = clip.tokenize(text_labels).cuda()
                clip_features = self.clip_model.encode_text(text_tokens).float()
        
        return clip_features

    def get_class_embedding(self, labels):
        bsz = len(labels) if isinstance(labels, list) else labels.size(0)

        # Encode labels using CLIP
        clip_features = self.encode_labels_with_clip(labels)
        
        # Project CLIP features to encoder_embed_dim
        class_embedding = self.class_emb(clip_features)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(
                -1).type_as(class_embedding).cuda()
            class_embedding = drop_latent_mask * self.fake_latent + \
                (1 - drop_latent_mask) * class_embedding
        return class_embedding

    @torch.no_grad()
    def forward_ema_encoder(self, imgs, labels):
        class_embedding = self.get_class_embedding(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        # mae encoder
        x = self.forward_encoder(x, mask, class_embedding)
        return x[:, self.buffer_size:]  # drop the buffer size for ema.

    def forward(self, imgs, labels, ema_x=None):
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        class_embedding = self.get_class_embedding(labels)

        # mae encoder
        x = self.forward_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_decoder(x, mask, class_embedding)
        # Cross attn
        class_embedding = class_embedding.unsqueeze(1).expand(-1, z.size(1), -1)
        z_attended, _ = self.cross_attention(query=z, key=class_embedding, value=class_embedding)
        z = z + z_attended 
        z = torch.cat([z, class_embedding], dim=-1)
        z = self.fuse_proj(z) 

        # diffloss
        if not isinstance(self.diffloss, nn.Identity):
            diffloss = self.diffusion_loss(z=z, target=gt_latents, mask=mask)
        else:
            diffloss = torch.zeros(1).type_as(z)

        # l1loss
        if not isinstance(self.jepaloss, nn.Identity):
            assert ema_x is not None, "ema_x must be passed if jepa loss is required."
            jepa_loss = self.jepa_loss(z, ema_x, mask)
        else:
            jepa_loss = torch.zeros(1).type_as(z)

        return diffloss, jepa_loss

    def sample_tokens(self, bsz, num_iter=64, cfg_scale=1.0, cfg_schedule="linear", labels=None, temperature=1.0, time_shifting_factor=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and cfg_scale
            if labels is not None:
                # Use CLIP to encode labels
                clip_features = self.encode_labels_with_clip(labels)
                class_embedding = self.class_emb(clip_features)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg_scale == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat(
                    [class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_decoder(x, mask, class_embedding)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor(
                [np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))  # type: ignore

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(
                    mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg_scale == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg_scale schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg_scale - 1) * (self.seq_len -
                                                  mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg_scale
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(
                z, temperature=temperature, cfg_scale=cfg_iter, time_shifting_factor=time_shifting_factor)
            if not cfg_scale == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(
                    2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(
                as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def JEPAT_base(**kwargs):
    # total: 212,017,440
    # encoder: 85,056,000
    # decoder: 85,056,000
    # denoising: 35,749,920
    model = JEPAT(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, diffloss_d=6, diffloss_w=1024,
        **kwargs)
    return model


def JEPAT_large(**kwargs):
    # total: 687,421,984
    # encoder: 302,312,448
    # decoder: 302,312,448
    # denoising: 72,230,432
    model = JEPAT(
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=8, diffloss_w=1280,
        **kwargs)
    return model


def JEPAT_huge(**kwargs):
    # total: 1,426,730,784
    # encoder: 629,683,200
    # decoder: 629,683,200
    # denoising: 151,206,944
    model = JEPAT(
        encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=12, diffloss_w=1536,
        **kwargs)
    return model
