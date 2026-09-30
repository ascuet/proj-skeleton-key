import torch.nn as nn
import torch.nn.functional as F

from core.models.TransformerModels import VisionTransformer, CONFIGS as VIT_CONFIGS

VIT_MODELS_2 = {'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'}


class ViT224Wrapper2(nn.Module):
    """Attack-space Design 2.

    The trainer/attack receives native CIFAR tensors (32x32).  Resizing happens
    inside the differentiable model forward, so adversarial perturbations are
    constrained in the original 32x32 input space while ViT sees 224x224.
    """

    def __init__(self, vit, input_size=224):
        super().__init__()
        self.vit = vit
        self.input_size = int(input_size)

    def _resize(self, x):
        target = (self.input_size, self.input_size)
        if x.shape[-2:] != target:
            x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        return self.vit(self._resize(x))

    def forward2(self, x, labels=None):
        x = self._resize(x)
        if hasattr(self.vit, 'forward2'):
            return self.vit.forward2(x, labels=labels)
        return self.vit(x)


def create_vit_2(name: str, num_classes: int, input_size: int = 224):
    if name not in VIT_MODELS_2:
        raise ValueError(f'Unsupported ViT model for Design 2: {name}')
    if int(input_size) != 224:
        raise ValueError('This Design-2 ViT pipeline is standardized on 224x224 model input.')

    vit = VisionTransformer(
        config=VIT_CONFIGS[name],
        img_size=224,
        num_classes=num_classes,
        zero_head=False,
        vis=False,
    )
    return ViT224Wrapper2(vit, input_size=224)
