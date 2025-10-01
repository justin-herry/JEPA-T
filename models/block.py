from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                output = output.to(self.weight.dtype)
            output = output * self.weight
        else:
            output = output.to(x.dtype)

        return output


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, eps=1e-6)

    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            return super().forward(*args, **kwargs)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer=RMSNorm):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        dtype = x.dtype
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        # batch size, seq_len, num heads, head_dim
        q, k, v = qkv.unbind(0)

        # batch size, num heads, seq_len, head dim
        q, k = self.q_norm(q).to(dtype), self.k_norm(k).to(dtype)
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        norm_layer=LayerNorm,
        mlp_layer=Mlp,
        attn_norm_layer=RMSNorm
    ):
        super().__init__()
        attn_norm_layer = attn_norm_layer or norm_layer

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=attn_norm_layer,
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask=None, freqs_cis=None, *args, **kwargs):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
