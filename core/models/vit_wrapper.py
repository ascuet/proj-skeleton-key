import torch.nn as nn
import torch.nn.functional as F


class ViTWrapper(nn.Module):
    """Upsamples input to 224x224 before passing to VisionTransformer.
    CIFAR-10/100 images are 32x32; ViT-B/16 needs 224x224 for a meaningful
    14x14 patch grid instead of a 2x2 grid.
    """
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit(x)

    def forward2(self, x, labels=None):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        if hasattr(self.vit, 'forward2'):
            return self.vit.forward2(x, labels=labels)
        return self.vit(x)
