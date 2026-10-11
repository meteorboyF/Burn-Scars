"""Model families.

unet        : segmentation_models_pytorch.Unet, ResNet-34 encoder, RANDOM init
              (encoder_weights=None), 6 input bands, 2 classes. ("from scratch")
prithvi     : Prithvi-EO-2.0-300M BASE pretrained encoder (never the BurnScars
              fine-tuned one, which saw IBM-train labels), FROZEN; features from
              transformer blocks 5/11/17/23 (the released config's indices) are
              extracted once, cached (fp16), and a light decoder is trained on them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRITHVI_LAYERS = (5, 11, 17, 23)
PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def build_unet(encoder="resnet34", in_channels=6, classes=2):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_channels, classes=classes)


def build_prithvi_encoder(pretrained: bool, ckpt_path=None):
    """Frozen Prithvi-EO-2.0-300M encoder. pretrained=False only for smoke tests."""
    from terratorch.registry import BACKBONE_REGISTRY
    kw = {"pretrained": pretrained, "bands": PRITHVI_BANDS}
    if ckpt_path:
        kw["ckpt_path"] = str(ckpt_path)
    enc = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", **kw)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def prithvi_features(encoder, x: torch.Tensor, layers=PRITHVI_LAYERS) -> torch.Tensor:
    """x: (B,6,H,W) normalised -> (B, len(layers), D, H/16, W/16). Drops the cls token."""
    outs = encoder(x)
    b, n, d = outs[layers[0]].shape
    side = int(round((n - 1) ** 0.5))
    feats = [outs[i][:, 1:, :].transpose(1, 2).reshape(b, d, side, side) for i in layers]
    return torch.stack(feats, 1)


class LightDecoder(nn.Module):
    """Light decoder on cached frozen features: per-layer 1x1 projection, fuse,
    then 4 x (upsample x2 + conv-BN-ReLU) from 1/16 to full resolution."""

    def __init__(self, n_layers=4, dim=1024, proj=128, widths=(256, 128, 64, 32), classes=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(dim, proj, 1) for _ in range(n_layers)])
        self.fuse = nn.Sequential(nn.Conv2d(proj * n_layers, widths[0], 3, padding=1),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        chans = [widths[0], *widths]
        self.up = nn.ModuleList([nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                                               nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True))
                                 for i in range(len(widths))])
        self.head = nn.Conv2d(widths[-1], classes, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:  # f: (B, L, D, h, w)
        z = torch.cat([p(f[:, i].float()) for i, p in enumerate(self.proj)], 1)
        z = self.fuse(z)
        for blk in self.up:
            z = blk(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(z)


def n_params(m: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
