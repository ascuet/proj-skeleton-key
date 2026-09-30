import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambavision import create_model as mamba_create_model


class MambaVisionWrapper(nn.Module):
    """Resize inputs to the Mamba resolution and apply ImageNet normalization."""

    def __init__(self, mamba: nn.Module, input_size: int = 224):
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        self.mamba = mamba
        self.input_size = int(input_size)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"MambaVision expects [N,C,H,W], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 3:
            raise ValueError(f"MambaVision expects 3 channels, got {x.shape[1]}")

        # Keep adversarial attacks in the original data space (e.g. 32x32
        # CIFAR). The resize is differentiable, so gradients flow back to the
        # original attack tensor.
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        x = (x - self.mean) / self.std
        return self.mamba(x)


MAMBA_CONFIGS = {
    "mamba_vision_T",
    "mamba_vision_S",
    "mamba_vision_B",
    "mamba_vision_L",
    "mamba_vision_L2",
}


def create_mamba(
    name: str,
    num_classes: int,
    input_size: int = 224,
    pretrained: bool = False,
):
    """Create a wrapped MambaVision model.

    ``input_size`` is the resolution consumed by the MambaVision backbone;
    attacks can still operate on the original dataloader resolution because
    resizing happens inside ``MambaVisionWrapper``.
    """
    if name not in MAMBA_CONFIGS:
        raise ValueError(
            f"Unsupported MambaVision model {name!r}; expected one of {sorted(MAMBA_CONFIGS)}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")

    # Needed by some MambaVision/PyTorch 2.6 checkpoint-loading paths.
    torch.serialization.add_safe_globals([argparse.Namespace])

    mamba = mamba_create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return MambaVisionWrapper(mamba, input_size=input_size)
