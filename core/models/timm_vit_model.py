import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


TIMM_VIT_CONFIGS = {
    'ViT-B_16-pt': 'vit_base_patch16_224',
    'ViT-L_16-pt': 'vit_large_patch16_224',
}


class TimmViTWrapper(nn.Module):
    """Upsamples 32x32 input to 96x96, applies ImageNet normalization,
    then passes to a pretrained timm ViT.
    img_size=96 triggers positional encoding interpolation (196 → 36 tokens).
    Using register_buffer so DataParallel copies mean/std to each GPU replica."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return self.vit(x)


def create_timm_vit(name, num_classes):
    timm_name = TIMM_VIT_CONFIGS[name]
    # img_size=96 causes timm to interpolate pretrained positional encodings
    # from 14x14 (196 tokens at 224px) to 6x6 (36 tokens at 96px).
    vit = timm.create_model(timm_name, pretrained=True, num_classes=num_classes, img_size=96)
    return TimmViTWrapper(vit)
