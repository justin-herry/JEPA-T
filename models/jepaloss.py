import torch.nn as nn
from timm.layers import Mlp


class JepaLoss(nn.Module):
    """JEAP Loss.
    """

    def __init__(self, in_features, out_features, norm_layer=None, beta: float = 2.0):
        super(JepaLoss, self).__init__()
        self.net = Mlp(
            in_features=in_features,
            hidden_features=in_features * 4,
            out_features=out_features,
            norm_layer=norm_layer)

        self.loss = nn.SmoothL1Loss(reduction="none", beta=beta)

    def forward(self, x, ema_x, mask=None):
        feat = self.net(x)
        loss = self.loss(feat, ema_x.detach())  # no grad to ema_x
        if mask is not None:
            return (loss.mean(dim=-1) * mask).sum() / mask.sum()
        return loss.mean()
