from __future__ import annotations

import argparse

import csv
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from collections import Counter
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (
    _this_dir,
    os.path.join(_this_dir, "core"),
    os.path.join(_this_dir, "core", "models"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.model import sew_resnet
from core.models.mamba_model import create_mamba as create_mamba_2
from Utilities import DataManagerPytorch as DMP

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SkeletonKeyAttack:

    def __init__(
        self,
        models: Sequence[nn.Module],
        target_idx: int = 0,
        N: int = 100,
        eps_max: float = 8 / 255,
        alpha: float | None = None,
        beta: float | None = None,
        loss_fn: LossFn | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if len(models) < 2:
            raise ValueError("At least one target and one non-target model are required.")
        if not (0 <= target_idx < len(models)):
            raise ValueError("target_idx is out of range.")
        if N <= 0:
            raise ValueError("N must be positive.")
        if eps_max <= 0:
            raise ValueError("eps_max must be positive.")

        self.models = list(models)
        self.target_idx = int(target_idx)
        self.non_target_indices = [i for i in range(len(self.models)) if i != self.target_idx]
        self.N = int(N)
        self.eps_max = float(eps_max)
        self.device = torch.device(device)

        default_step = self.eps_max / (2.0 * self.N)
        self.alpha = float(default_step if alpha is None else alpha)
        self.beta = float(default_step if beta is None else beta)
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive.")

        self.loss_fn = loss_fn or self._cross_entropy_loss

        for model in self.models:
            model.to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y)

    def _forward_batched_logits(self, model_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Forward an entire attack batch, preserving model architecture/state."""
        model = self.models[model_idx]
        if isinstance(model, SNNLogitsWrapper):
            functional.reset_net(model.model)
            logits = model.model(x)
            if logits.ndim == 3:
                logits = logits.mean(dim=0)
            return logits
        return model(x)

    def _batched_input_gradient(
        self,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        model_idx: int,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_in = x_adv.detach().clone().requires_grad_(True)
        logits = self._forward_batched_logits(model_idx, x_in)
        per_sample_loss = F.cross_entropy(logits, y, reduction="none")
        total_loss = per_sample_loss.sum()
        grad_x = torch.autograd.grad(
            total_loss,
            x_in,
            retain_graph=False,
            create_graph=False,
        )[0]
        return grad_x.detach(), logits.detach() if return_logits else None

    @torch.no_grad()
    def _target_correct_class_probability_from_logits(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return probs.gather(1, y.view(-1, 1)).squeeze(1)

    def _projection(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = torch.clamp(x_adv - x, min=-self.eps_max, max=self.eps_max)
        return torch.clamp(x + delta, min=0.0, max=1.0)

    def attack(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).long()

        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2:] != (32, 32):
            raise ValueError(f"Expected CIFAR input [B,3,32,32], got {tuple(x.shape)}")
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("SkeletonKeyAttack expects inputs in [0,1].")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape [B] matching x batch size.")

        x_adv = x.detach().clone()

        for step in range(self.N):
            f_blend = torch.zeros_like(x_adv)
            for model_idx in self.non_target_indices:
                grad_v, _ = self._batched_input_gradient(x_adv, y, model_idx)
                f_blend += torch.sign(grad_v)

            target_grad, target_logits = self._batched_input_gradient(
                x_adv, y, self.target_idx, return_logits=True
            )
            target_prob = self._target_correct_class_probability_from_logits(target_logits, y)
            indicator = (target_prob < 0.5).to(dtype=x_adv.dtype).view(-1, 1, 1, 1)
            f_target = torch.sign(target_grad) * indicator

            x_adv = x_adv + self.alpha * f_blend - self.beta * f_target
            x_adv = self._projection(x, x_adv).detach()

            if verbose and ((step + 1) == 1 or (step + 1) % 10 == 0 or (step + 1) == self.N):
                delta = (x_adv - x).abs().flatten(1).max(dim=1).values
                print(
                    f"      Attack step {step + 1}/{self.N} "
                    f"max_Linf={delta.max().item():.8f}"
                )

        return x_adv


class ResizeNormalizeWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_size: int,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = False,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_size = int(target_size)
        self.mode = mode
        self.align_corners = align_corners if mode in ("bilinear", "bicubic") else None
        self.antialias = bool(antialias)
        if (mean is None) != (std is None):
            raise ValueError("mean and std must either both be provided or both be omitted.")
        if mean is not None and std is not None:
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("mean and std must each contain 3 channel values.")
            if any(s == 0 for s in std):
                raise ValueError("std values must be non-zero.")
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        else:
            self.mean = None
            self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.target_size, self.target_size):
            kwargs = {"size": (self.target_size, self.target_size), "mode": self.mode}
            if self.mode in ("bilinear", "bicubic"):
                kwargs["align_corners"] = self.align_corners
            kwargs["antialias"] = self.antialias
            x = F.interpolate(x, **kwargs)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return self.model(x)


class SNNLogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self.model)
        out = self.model(x)
        if out.ndim == 3:
            out = out.mean(dim=0)
        return out


def build_snn(num_classes: int = 10) -> nn.Module:
    return sew_resnet.multi_step_sew_resnet18(
        T=5,
        num_classes=num_classes,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepParametricLIFNode,
        surrogate_function=surrogate.ATan(),
    )


CHECKPOINT_PATHS = {
    "vit": "checkpoint/Vit_L_16.pt",
    "cait": "checkpoint/Cait_xxs.pt",
    "mamba": "checkpoint/MambaVision_T.pt",
    "wrn": "checkpoint/Wrn_28_10.pt",
    "dm_r18": "checkpoint/snn_sew_resnet.pt",
    "fat_r18": "checkpoint/snn_resnet18_cifar10_5_219_7315.pth",
}

RESIZE_TARGET_SIZE = {"vit": 224, "cait": 224, "mamba": 224}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _CaiTLayerScale(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, depth: int) -> None:
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        self.scale = nn.Parameter(torch.full((1, 1, dim), init_eps))
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) * self.scale


class _CaiTPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _CaiTFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CaiTTalkingHeadsAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 48, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        context_tokens = x if context is None else torch.cat((x, context), dim=1)
        q = self.to_q(x)
        k, v = self.to_kv(context_tokens).chunk(2, dim=-1)
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, context_tokens.shape[1], h, -1).transpose(1, 2)
        v = v.view(b, context_tokens.shape[1], h, -1).transpose(1, 2)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        dots = torch.einsum("bhij,hg->bgij", dots, self.mix_heads_pre_attn)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = torch.einsum("bhij,hg->bgij", attn, self.mix_heads_post_attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * k.shape[-1])
        return self.to_out(out)


class _CaiTTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0, layer_dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = float(layer_dropout)
        for ind in range(depth):
            self.layers.append(
                nn.ModuleList([
                    _CaiTLayerScale(
                        dim,
                        _CaiTPreNorm(dim, _CaiTTalkingHeadsAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        depth=ind + 1,
                    ),
                    _CaiTLayerScale(
                        dim,
                        _CaiTPreNorm(dim, _CaiTFeedForward(dim, mlp_dim, dropout=dropout)),
                        depth=ind + 1,
                    ),
                ])
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        layers = list(self.layers)
        for attn, ff in layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CaiT32CIFAR(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, num_classes: int = 10,
                 dim: int = 192, depth: int = 12, cls_depth: int = 2, heads: int = 4,
                 mlp_dim: int = 384, dim_head: int = 48, dropout: float = 0.0,
                 emb_dropout: float = 0.0, layer_dropout: float = 0.0) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(nn.Identity(), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.patch_transformer = _CaiTTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = _CaiTTransformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        p = self.patch_size
        if c != 3 or h != self.image_size or w != self.image_size:
            raise ValueError(f"CaiT32CIFAR expects [B,3,{self.image_size},{self.image_size}], got {tuple(img.shape)}")
        x = img.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(b, (h // p) * (w // p), p * p * c)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._patchify(img)
        x = self.to_patch_embedding[1](x)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.patch_transformer(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = self.cls_transformer(cls_tokens, context=x)
        return self.mlp_head(x[:, 0])


class _WRNBlock(nn.Module):
    """Exact WRN-28-10 block used by the training checkpoint."""
    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=False)
        self.conv_0 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=0, bias=False
        )
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=False)
        self.conv_1 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1,
                stride=stride, padding=0, bias=False
            )
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))

        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')

        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _WRNBlockGroup(nn.Module):
    def __init__(self, num_blocks: int, in_planes: int, out_planes: int,
                 stride: int, activation_fn=nn.ReLU):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                _WRNBlock(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    activation_fn=activation_fn,
                )
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    """WRN-28-10-SiLU with the checkpoint's CIFAR-10 normalization."""
    def __init__(self, num_classes: int = 10, depth: int = 28,
                 width: int = 10, activation_fn=nn.ReLU,
                 mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2471, 0.2435, 0.2616), padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        # Register normalization constants as buffers so they move with the
        # model and are never cached as inference tensors during an earlier
        # no_grad/inference-mode evaluation.  This is essential because the
        # same WRN instance is later used for input-gradient computation.
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, num_input_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, num_input_channels, 1, 1),
            persistent=False,
        )
        self.padding = padding

        num_channels = [16, 16 * width, 32 * width, 64 * width]
        if (depth - 4) % 6 != 0:
            raise ValueError('Invalid WRN depth.')
        num_blocks = (depth - 4) // 6

        self.init_conv = nn.Conv2d(
            num_input_channels, num_channels[0], kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.layer = nn.Sequential(
            _WRNBlockGroup(num_blocks, num_channels[0], num_channels[1], 1, activation_fn),
            _WRNBlockGroup(num_blocks, num_channels[1], num_channels[2], 2, activation_fn),
            _WRNBlockGroup(num_blocks, num_channels[2], num_channels[3], 2, activation_fn),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=False)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        # Use registered buffers directly.  Do not cache .to(device) tensors
        # created under no_grad/inference_mode: those tensors can later trigger
        # "Inference tensors cannot be saved for backward" during APGD/Skeleton
        # Key input-gradient computation.
        out = (x - self.mean) / self.std

        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_this_dir, path)


def _require_checkpoint(path: str) -> str:
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved!r}")
    return resolved


def _load_checkpoint(path: str, device: torch.device) -> object:
    checkpoint_path = _require_checkpoint(path)
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _looks_like_state_dict(obj: object) -> bool:
    return isinstance(obj, Mapping) and all(isinstance(k, str) for k in obj.keys()) and all(torch.is_tensor(v) for v in obj.values())


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if _looks_like_state_dict(checkpoint):
        return dict(checkpoint)  # type: ignore[arg-type]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    preferred_keys = ("model_state_dict", "state_dict", "unaveraged_model_state_dict", "model", "net", "network")
    for key in preferred_keys:
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return dict(value)  # type: ignore[arg-type]
    for _, value in checkpoint.items():
        if _looks_like_state_dict(value):
            return dict(value)  # type: ignore[arg-type]
    raise KeyError("Could not find a tensor state_dict in checkpoint.")


def _clean_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break
        cleaned[new_key] = value
    return cleaned


def _adapt_state_dict_to_model(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    cleaned = _clean_state_dict_keys(state_dict)
    target_keys = set(model.state_dict().keys())
    target_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    used_source_keys: set[str] = set()

    for source_key, value in cleaned.items():
        if source_key in target_keys and tuple(value.shape) == tuple(target_state[source_key].shape):
            adapted[source_key] = value
            used_source_keys.add(source_key)

    for target_key in target_keys:
        if target_key in adapted:
            continue
        candidates = [s for s in cleaned if s not in used_source_keys and s.endswith("." + target_key)]
        if len(candidates) == 1 and tuple(cleaned[candidates[0]].shape) == tuple(target_state[target_key].shape):
            adapted[target_key] = cleaned[candidates[0]]
            used_source_keys.add(candidates[0])

    remaining_targets = [k for k in target_state if k not in adapted]
    remaining_sources = [k for k in cleaned if k not in used_source_keys]
    if remaining_targets and len(remaining_targets) == len(remaining_sources):
        if all(tuple(target_state[t].shape) == tuple(cleaned[s].shape) for t, s in zip(remaining_targets, remaining_sources)):
            for t, s in zip(remaining_targets, remaining_sources):
                adapted[t] = cleaned[s]
                used_source_keys.add(s)

    ignored = [k for k in cleaned if k not in used_source_keys]
    return adapted, ignored


def _load_weights_strict(model: nn.Module, state_dict: Mapping[str, torch.Tensor], model_name: str) -> None:
    adapted, ignored = _adapt_state_dict_to_model(model, state_dict)
    result = model.load_state_dict(adapted, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{model_name} checkpoint could not be mapped. "
            f"Missing={result.missing_keys[:20]}, Unexpected={result.unexpected_keys[:20]}"
        )
    print(f"[{model_name}] Loaded {len(adapted)} tensors ({len(ignored)} checkpoint-only tensors ignored).")


def load_mamba(device: torch.device, num_classes: int = 10) -> nn.Module:
    model = create_mamba_2(
        name="mamba_vision_T",
        num_classes=num_classes,
        input_size=224,
        pretrained=False,
    )
    state = _extract_state_dict(_load_checkpoint(CHECKPOINT_PATHS["mamba"], device))
    target = set(model.state_dict().keys())
    if any(k.startswith("mamba.") for k in target) and not any(k.startswith("mamba.") for k in state):
        state = {"mamba." + k: v for k, v in state.items()}
    _load_weights_strict(model, state, "MambaVision-T")
    model = model.to(device).eval()
    wrapper = ResizeNormalizeWrapper(
        model,
        target_size=224,
        mean=None,
        std=None,
    ).to(device).eval()
    wrapper._is_mamba_wrapper = True
    return wrapper


def load_snn(device: torch.device, num_classes: int, kind: str) -> nn.Module:
    model = build_snn(num_classes)
    checkpoint_key = "dm_r18" if kind == "DM_R18" else "fat_r18"
    state_blob = _load_checkpoint(CHECKPOINT_PATHS[checkpoint_key], device)
    state = _extract_state_dict(state_blob)
    cleaned = _clean_state_dict_keys(state)
    normalized = {}
    for key, value in cleaned.items():
        if key.startswith("0."):
            key = key[2:]
        normalized[key] = value
    target_state = model.state_dict()
    compatible = {}
    wrong_shapes = []
    for key, value in normalized.items():
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            compatible[key] = value
        elif key in target_state:
            wrong_shapes.append((key, tuple(value.shape), tuple(target_state[key].shape)))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected or wrong_shapes:
        raise RuntimeError(
            f"{kind} checkpoint mismatch. Missing={real_missing}; Unexpected={unexpected}; WrongShapes={wrong_shapes}"
        )
    print(f"[{kind}] Loaded {len(compatible)} tensors; ignored {len(normalized) - len(compatible)} checkpoint-only tensors.")
    return SNNLogitsWrapper(model).to(device).eval()


def load_vit(device: torch.device, num_classes: int = 10) -> nn.Module:
    # from TransformerModels import VisionTransformer, CONFIGS
    from core.models.TransformerModels import VisionTransformer, CONFIGS
    config = CONFIGS["ViT-L_16"]
    vit = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes)
    state = _extract_state_dict(_load_checkpoint(CHECKPOINT_PATHS["vit"], device))
    _load_weights_strict(vit, state, "ViT-L/16")
    vit = vit.to(device).eval()
    return ResizeNormalizeWrapper(vit, target_size=224, mean=None, std=None).to(device).eval()


def load_cait(device: torch.device, num_classes: int = 10) -> nn.Module:
    cait = CaiT32CIFAR(image_size=32, patch_size=4, num_classes=num_classes, dim=192, depth=12, cls_depth=2, heads=4, mlp_dim=384, dim_head=48)
    state = _extract_state_dict(_load_checkpoint(CHECKPOINT_PATHS["cait"], device))
    _load_weights_strict(cait, state, "CaiT-XXS24-CIFAR")
    return cait.to(device).eval()


def load_wrn(device: torch.device, num_classes: int = 10) -> nn.Module:
    wrn = WideResNet(
        num_classes=num_classes,
        depth=28,
        width=10,
        activation_fn=nn.SiLU,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2471, 0.2435, 0.2616),
    )
    state = _extract_state_dict(_load_checkpoint(CHECKPOINT_PATHS["wrn"], device))
    _load_weights_strict(wrn, state, "WideResNet-28-10")
    return wrn.to(device).eval()


def _predict(models: Sequence[nn.Module], x: torch.Tensor) -> list[list[int]]:
    with torch.no_grad():
        outputs = [model(x).argmax(dim=1).tolist() for model in models]
    return [list(row) for row in zip(*outputs)]


MODEL_NAMES = ["ViT", "CAIT", "WRN", "MAMBA", "DM_R18", "FAT_R18"]
DISPLAY_NAMES = {
    "ViT": "ViT-L/16",
    "CAIT": "CaiT-XXS24",
    "WRN": "WRN-28-10",
    "MAMBA": "MambaVision-T",
    "DM_R18": "DM-R18",
    "FAT_R18": "FAT-R18",
}
TARGET_ALIASES = {
    "0": "ViT", "vit": "ViT", "vit-l/16": "ViT", "vit_l_16": "ViT",
    "1": "CAIT", "cait": "CAIT", "cait-xxs24": "CAIT",
    "2": "WRN", "wrn": "WRN", "wrn-28-10": "WRN", "wrn_28_10": "WRN",
    "3": "MAMBA", "mamba": "MAMBA", "mambavision-t": "MAMBA", "mamba_vision_t": "MAMBA",
    "4": "DM_R18", "dm_r18": "DM_R18", "dm-r18": "DM_R18", "dm": "DM_R18",
    "5": "FAT_R18", "fat_r18": "FAT_R18", "fat-r18": "FAT_R18", "fat": "FAT_R18",
}


class Tee:
    """Mirror stdout/stderr to both terminal and a per-target log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_all_models(device: torch.device, num_classes: int):
    print("Loading ViT-L/16...")
    model_vit = load_vit(device, num_classes)
    print("Loading CaiT-XXS24...")
    model_cait = load_cait(device, num_classes)
    print("Loading WRN-28-10...")
    model_wrn = load_wrn(device, num_classes)
    print("Loading MambaVision-T...")
    model_mamba = load_mamba(device, num_classes)
    print("Loading DM-R18...")
    model_dm = load_snn(device, num_classes, "DM_R18")
    print("Loading FAT-R18...")
    model_fat = load_snn(device, num_classes, "FAT_R18")
    return [model_vit, model_cait, model_wrn, model_mamba, model_dm, model_fat]


def make_subset_payload(x_clean, x_adv, y, success, target_correct, non_target_fooled,
                        predictions, linf, sample_indices, mask):
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    return {
        "sample_indices": sample_indices[idx].cpu(),
        "x_clean": x_clean[idx].cpu(),
        "x_adv": x_adv[idx].cpu(),
        "y": y[idx].cpu(),
        "success": success[idx].cpu(),
        "target_correct": target_correct[idx].cpu(),
        "non_target_fooled": non_target_fooled[idx].cpu(),
        "predictions": predictions[idx].cpu(),
        "linf": linf[idx].cpu(),
    }


def _load_all_validation_samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the full CIFAR-10 validation/test set in [0,1] 32x32 format."""
    loader = DMP.GetCIFAR10Validation(imgSize=32, batchSize=256)
    x_parts = []
    y_parts = []
    idx_parts = []
    offset = 0
    for x_batch, y_batch in loader:
        n = len(y_batch)
        x_parts.append(x_batch.cpu())
        y_parts.append(y_batch.long().cpu())
        idx_parts.append(torch.arange(offset, offset + n, dtype=torch.long))
        offset += n
    return (
        torch.cat(x_parts, dim=0),
        torch.cat(y_parts, dim=0),
        torch.cat(idx_parts, dim=0),
    )


def _create_or_load_common_xstart(
    seed_root: str,
    num_per_class: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Create one random balanced xStart for this seed and reuse it for every target run."""
    os.makedirs(seed_root, exist_ok=True)
    common_path = os.path.join(seed_root, "common_xStart.pt")
    manifest_path = os.path.join(seed_root, "common_xStart_manifest.json")
    total_requested = num_per_class * num_classes

    if os.path.isfile(common_path):
        blob = torch.load(common_path, map_location="cpu", weights_only=False)
        required = {"x", "y", "sample_indices", "num_per_class", "num_classes", "seed"}
        missing = required.difference(blob.keys()) if isinstance(blob, Mapping) else required
        if missing:
            raise RuntimeError(f"Existing {common_path} is missing fields: {sorted(missing)}")
        if int(blob["num_per_class"]) != num_per_class or int(blob["num_classes"]) != num_classes:
            raise RuntimeError(
                f"Existing common_xStart was created for {blob['num_per_class']} per class; "
                f"requested {num_per_class}. Remove {common_path} to intentionally regenerate it."
            )
        if int(blob["seed"]) != seed:
            raise RuntimeError(
                f"Existing common_xStart was created with seed {blob['seed']}, "
                f"but requested seed {seed}. Keep the existing seed for consistency, "
                f"or remove {common_path} to intentionally regenerate the shared set."
            )
        x = blob["x"].cpu()
        y = blob["y"].long().cpu()
        sample_indices = blob["sample_indices"].long().cpu()
        if len(y) != total_requested:
            raise RuntimeError(f"Existing common_xStart has {len(y)} samples, expected {total_requested}.")
        return x, y, sample_indices, common_path

    x_all, y_all, idx_all = _load_all_validation_samples()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    selected_indices = []
    for cls in range(num_classes):
        cls_idx = torch.nonzero(y_all.eq(cls), as_tuple=False).flatten()
        if len(cls_idx) < num_per_class:
            raise RuntimeError(
                f"Class {cls} has only {len(cls_idx)} validation/test samples; "
                f"cannot select {num_per_class}."
            )
        perm = torch.randperm(len(cls_idx), generator=generator)
        selected_indices.append(cls_idx[perm[:num_per_class]])

    selected_indices = torch.cat(selected_indices, dim=0)
    order = torch.randperm(len(selected_indices), generator=generator)
    selected_indices = selected_indices[order]

    x = x_all[selected_indices].contiguous()
    y = y_all[selected_indices].contiguous()
    sample_indices = idx_all[selected_indices].contiguous()

    class_counts = [int((y == cls).sum().item()) for cls in range(num_classes)]
    payload = {
        "x": x,
        "y": y,
        "sample_indices": sample_indices,
        "num_per_class": num_per_class,
        "num_classes": num_classes,
        "total_samples": total_requested,
        "seed": seed,
        "source": "DMP.GetCIFAR10Validation",
        "class_counts": class_counts,
    }
    torch.save(payload, common_path)

    manifest = {
        "file": common_path,
        "source": "DMP.GetCIFAR10Validation",
        "seed": seed,
        "num_classes": num_classes,
        "num_per_class": num_per_class,
        "total_samples": total_requested,
        "class_counts": class_counts,
        "selection_rule": "randomly select num_per_class samples independently within each class, then randomize overall order",
        "target_correctness_filter": False,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return x, y, sample_indices, common_path


PERSISTENT_ROOT = os.path.join(_this_dir, "generated_adversarial_examples", "skeleton_key", "cifar10")


def _atomic_torch_save(obj, path: str) -> None:
    """Write a torch file atomically as far as the mounted filesystem allows."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _batch_filename(start: int, end_exclusive: int) -> str:
    return f"batch_{start:06d}_{end_exclusive - 1:06d}.pt"


def _list_completed_batches(batch_dir: str, total_samples: int, batch_size: int) -> list[tuple[int, int, str]]:
    """Return contiguous completed batch ranges. A later orphan batch is not trusted for resume."""
    completed = []
    start = 0
    while start < total_samples:
        end = min(start + batch_size, total_samples)
        path = os.path.join(batch_dir, _batch_filename(start, end))
        if not os.path.isfile(path):
            break
        completed.append((start, end, path))
        start = end
    return completed


def _load_completed_batch_records(batch_dir: str, total_samples: int, batch_size: int):
    batches = _list_completed_batches(batch_dir, total_samples, batch_size)
    if not batches:
        return [], 0
    payloads = []
    next_sample = 0
    for start, end, path in batches:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "start", "end", "sample_indices", "x_adv", "y", "predictions",
            "success", "target_correct", "non_target_fooled", "linf"
        }
        if not required.issubset(blob.keys()):
            raise RuntimeError(f"Incomplete batch checkpoint: {path}")
        if int(blob["start"]) != start or int(blob["end"]) != end:
            raise RuntimeError(f"Batch checkpoint range mismatch: {path}")
        if len(blob["y"]) != end - start:
            raise RuntimeError(f"Batch checkpoint sample-count mismatch: {path}")
        payloads.append(blob)
        next_sample = end
    return payloads, next_sample


def _save_partial_progress(
    progress_path: str,
    *,
    target_name: str,
    seed: int,
    total_samples: int,
    batch_size: int,
    next_sample: int,
    working_count: int,
    max_linf: float,
    mean_linf_sum: float,
    processed_count: int,
) -> None:
    progress = {
        "status": "running",
        "target_model": target_name,
        "seed": seed,
        "total_samples": total_samples,
        "batch_size": batch_size,
        "next_sample": next_sample,
        "last_completed_sample": next_sample - 1,
        "processed_samples": processed_count,
        "working_keys": working_count,
        "running_yield": working_count / processed_count if processed_count else 0.0,
        "max_linf": max_linf,
        "mean_linf_so_far": mean_linf_sum / processed_count if processed_count else 0.0,
        "updated_at": datetime.now().isoformat(),
    }
    _atomic_json_save(progress, progress_path)


def _write_predictions_json(
    path: str,
    *,
    target_name: str,
    seed: int,
    num_per_class: int,
    y_all: torch.Tensor,
    sample_indices: torch.Tensor,
    initial_predictions: torch.Tensor,
    predictions_all: torch.Tensor,
    success_all: torch.Tensor,
    target_correct_all: torch.Tensor,
    non_target_fooled_all: torch.Tensor,
    linf_all: torch.Tensor,
) -> None:
    records = []
    clean_rows = initial_predictions.tolist()
    adv_rows = predictions_all.tolist()
    for i in range(len(y_all)):
        clean_row = {
            DISPLAY_NAMES[MODEL_NAMES[m]]: int(clean_rows[i][m])
            for m in range(len(MODEL_NAMES))
        }
        adv_row = {
            DISPLAY_NAMES[MODEL_NAMES[m]]: int(adv_rows[i][m])
            for m in range(len(MODEL_NAMES))
        }
        records.append({
            "sample": i,
            "dataset_index": int(sample_indices[i]),
            "true_label": int(y_all[i]),
            "clean_predictions": clean_row,
            "adversarial_predictions": adv_row,
            "success": bool(success_all[i]),
            "target_model": DISPLAY_NAMES[target_name],
            "target_correct": bool(target_correct_all[i]),
            "all_non_targets_fooled": bool(non_target_fooled_all[i]),
            "linf": float(linf_all[i]),
        })
    _atomic_json_save({
        "target_model": DISPLAY_NAMES[target_name],
        "model_order": [DISPLAY_NAMES[n] for n in MODEL_NAMES],
        "seed": seed,
        "num_per_class": num_per_class,
        "total_samples": len(y_all),
        "records": records,
    }, path)


def run_resumable(args) -> None:
    target_name = args.target
    target_idx = MODEL_NAMES.index(target_name)
    num_classes = 10
    total_requested = args.num_per_class * num_classes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Request/allocate a GPU before running.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {args.gpu_id}. Visible CUDA devices: 0..{torch.cuda.device_count()-1}"
        )
    device = torch.device(f"cuda:{args.gpu_id}")

    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for this experiment.")

    seed_root = os.path.join(args.output_root, f"seed_{args.seed}")
    target_dir = os.path.join(seed_root, target_name)
    batch_dir = os.path.join(target_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)

    log_path = os.path.join(target_dir, "skeleton_key.log")
    metrics_path = os.path.join(target_dir, "metrics.json")
    csv_path = os.path.join(target_dir, "results.csv")
    predictions_path = os.path.join(target_dir, "predictions.json")
    clean_path = os.path.join(target_dir, "clean.pt")
    clean_predictions_path = os.path.join(target_dir, "clean_predictions.pt")
    all_keys_path = os.path.join(target_dir, "all_skeleton_keys.pt")
    working_path = os.path.join(target_dir, "working_keys.pt")
    non_working_path = os.path.join(target_dir, "non_working_keys.pt")
    progress_path = os.path.join(target_dir, "progress.json")
    completion_path = os.path.join(target_dir, "COMPLETED.json")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    started_at = datetime.now().isoformat()
    start_time = time.time()

    try:
        print("=" * 80)
        print("SKELETON KEY - RESUMABLE BATCHED ALL-6 DEFENSE EXPERIMENT")
        print("=" * 80)
        print(f"Target model       : {DISPLAY_NAMES[target_name]} ({target_name}, id={target_idx})")
        print("Non-target models  : " + ", ".join(DISPLAY_NAMES[n] for n in MODEL_NAMES if n != target_name))
        print(f"Device             : {device}")
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
        print(f"Samples per class  : {args.num_per_class}")
        print(f"Total samples      : {total_requested}")
        print(f"Shared xStart seed : {args.seed}")
        print(f"Batch size         : {args.batch_size}")
        print(f"Steps              : {args.steps}")
        print(f"Epsilon            : {args.epsilon:.8f}")
        print("xStart selection   : RANDOM, class-balanced, no correctness filter")
        print("Gradients          : PER-SAMPLE (summed per-sample CE; NOT batch-averaged)")
        print("Persistence        : batch-level checkpoints")
        print(f"Persistent root    : {args.output_root}")
        print(f"Target directory   : {target_dir}")

        if os.path.isfile(completion_path) and not args.restart_target:
            print(f"Target already completed. Reusing existing results: {completion_path}")
            return

        if args.restart_target:
            print("RESTART_TARGET=True: removing previous target checkpoints/results.")
            for name in os.listdir(target_dir):
                path = os.path.join(target_dir, name)
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                elif name != os.path.basename(log_path):
                    os.remove(path)
            os.makedirs(batch_dir, exist_ok=True)

        # Load all six models. Architecture/checkpoint definitions remain unchanged.
        models = load_all_models(device, num_classes)
        target_model = models[target_idx]

        # One shared xStart for all six target runs.
        x_all, y_all, sample_indices, common_path = _create_or_load_common_xstart(
            seed_root, args.num_per_class, num_classes, args.seed
        )
        class_counts = [int((y_all == cls).sum().item()) for cls in range(num_classes)]
        if len(y_all) != total_requested:
            raise RuntimeError(f"Expected {total_requested} selected samples, got {len(y_all)}")
        if any(c != args.num_per_class for c in class_counts):
            raise RuntimeError(f"Class balance mismatch: {class_counts}")
        print(f"Selected samples   : {len(y_all)}")
        print(f"Class counts       : {class_counts}")
        print(f"Common xStart      : {common_path}")

        if os.path.isfile(clean_predictions_path):
            initial_blob = torch.load(clean_predictions_path, map_location="cpu", weights_only=False)
            initial_predictions = initial_blob["initial_predictions"].long().cpu()
            clean_accuracy_by_model = initial_blob["clean_accuracy_by_model"]
            print("Loaded clean predictions from persistent Drive state.")
        else:
            initial_predictions = torch.empty((len(y_all), len(MODEL_NAMES)), dtype=torch.long)
            clean_accuracy_by_model = {}
            with torch.inference_mode():
                for idx, model_name in enumerate(MODEL_NAMES):
                    correct = 0
                    for start in range(0, len(y_all), args.batch_size):
                        end = min(start + args.batch_size, len(y_all))
                        pred = models[idx](x_all[start:end].to(device, non_blocking=True)).argmax(dim=1).cpu()
                        initial_predictions[start:end, idx] = pred
                        correct += int(pred.eq(y_all[start:end]).sum().item())
                    clean_accuracy_by_model[model_name] = correct / len(y_all)
                    print(f"Clean accuracy - {DISPLAY_NAMES[model_name]}: {100.0 * clean_accuracy_by_model[model_name]:.2f}%")
            _atomic_torch_save({
                "initial_predictions": initial_predictions,
                "clean_accuracy_by_model": clean_accuracy_by_model,
            }, clean_predictions_path)

        print("Clean predictions for all six models are persisted and will be included in the final CSV/JSON.")
        _atomic_torch_save({
            "x": x_all,
            "y": y_all,
            "sample_indices": sample_indices,
            "target_model": target_name,
            "target_display_name": DISPLAY_NAMES[target_name],
            "class_counts": class_counts,
            "common_xStart": True,
            "common_xStart_path": common_path,
            "seed": args.seed,
            "target_correctness_filter": False,
        }, clean_path)

        x_all_device = x_all.to(device, non_blocking=True)
        attacker = SkeletonKeyAttack(
            models=models,
            target_idx=target_idx,
            N=args.steps,
            eps_max=args.epsilon,
            device=device,
        )

        completed_payloads, resume_start = _load_completed_batch_records(
            batch_dir, len(y_all), args.batch_size
        )
        if completed_payloads:
            print(f"Resuming from persistent state: {len(completed_payloads)} batches already complete.")
            print(f"Next sample to process: {resume_start}")
        else:
            print("No completed attack batches found. Starting from sample 0.")

        working_count = sum(int(b["success"].sum().item()) for b in completed_payloads)
        processed_count = resume_start
        max_linf = max(
            [float(b["linf"].max().item()) for b in completed_payloads],
            default=0.0,
        )
        mean_linf_sum = sum(float(b["linf"].sum().item()) for b in completed_payloads)
        _save_partial_progress(
            progress_path,
            target_name=target_name,
            seed=args.seed,
            total_samples=len(y_all),
            batch_size=args.batch_size,
            next_sample=resume_start,
            working_count=working_count,
            max_linf=max_linf,
            mean_linf_sum=mean_linf_sum,
            processed_count=processed_count,
        )

        for start in range(resume_start, len(y_all), args.batch_size):
            end = min(start + args.batch_size, len(y_all))
            batch_path = os.path.join(batch_dir, _batch_filename(start, end))
            if os.path.isfile(batch_path):
                print(f"Skipping already-saved batch {start + 1}-{end}.")
                continue

            x_batch = x_all_device[start:end]
            y_batch_cpu = y_all[start:end]
            print(f"\nProcessing batch {start + 1}-{end} / {len(y_all)}")

            x_adv = attacker.attack(x_batch, y_batch_cpu, verbose=True)
            preds_list = _predict(models, x_adv)
            preds = torch.tensor(preds_list, dtype=torch.long)
            target_correct = preds[:, target_idx].eq(y_batch_cpu)
            non_target_cols = [j for j in range(len(MODEL_NAMES)) if j != target_idx]
            non_target_fooled = torch.ones(len(y_batch_cpu), dtype=torch.bool)
            for j in non_target_cols:
                non_target_fooled &= preds[:, j].ne(y_batch_cpu)
            success = target_correct & non_target_fooled
            linf = (x_adv.detach() - x_batch).abs().flatten(1).max(dim=1).values.cpu()

            batch_payload = {
                "start": start,
                "end": end,
                "sample_indices": sample_indices[start:end].cpu(),
                "x_adv": x_adv.detach().cpu(),
                "y": y_batch_cpu.cpu(),
                "predictions": preds.cpu(),
                "success": success.cpu(),
                "target_correct": target_correct.cpu(),
                "non_target_fooled": non_target_fooled.cpu(),
                "linf": linf.cpu(),
            }
            _atomic_torch_save(batch_payload, batch_path)

            working_count += int(success.sum().item())
            processed_count = end
            max_linf = max(max_linf, float(linf.max().item()))
            mean_linf_sum += float(linf.sum().item())
            _save_partial_progress(
                progress_path,
                target_name=target_name,
                seed=args.seed,
                total_samples=len(y_all),
                batch_size=args.batch_size,
                next_sample=end,
                working_count=working_count,
                max_linf=max_linf,
                mean_linf_sum=mean_linf_sum,
                processed_count=processed_count,
            )

            running_yield = working_count / processed_count
            print(
                f"Batch checkpoint saved: {batch_path}\n"
                f"Batch working keys: {int(success.sum().item())}; "
                f"cumulative working={working_count}/{processed_count}; "
                f"running_yield={running_yield:.4f} ({100.0 * running_yield:.2f}%)"
            )

            del x_adv, preds_list, preds, target_correct, non_target_fooled, success, linf

        all_batches, final_next = _load_completed_batch_records(batch_dir, len(y_all), args.batch_size)
        if final_next != len(y_all):
            raise RuntimeError(
                f"Experiment is not complete: saved through sample {final_next}, expected {len(y_all)}."
            )

        x_adv_all = torch.cat([b["x_adv"] for b in all_batches], dim=0)
        predictions_all = torch.cat([b["predictions"] for b in all_batches], dim=0)
        success_all = torch.cat([b["success"] for b in all_batches], dim=0)
        target_correct_all = torch.cat([b["target_correct"] for b in all_batches], dim=0)
        non_target_fooled_all = torch.cat([b["non_target_fooled"] for b in all_batches], dim=0)
        linf_all = torch.cat([b["linf"] for b in all_batches], dim=0)

        working_count = int(success_all.sum().item())
        non_working_count = int((~success_all).sum().item())
        yield_rate = working_count / len(y_all)
        finished_at = datetime.now().isoformat()
        elapsed_seconds = time.time() - start_time

        all_payload = {
            "sample_indices": sample_indices,
            "x_clean": x_all,
            "x_adv": x_adv_all,
            "y": y_all,
            "success": success_all,
            "target_correct": target_correct_all,
            "non_target_fooled": non_target_fooled_all,
            "predictions": predictions_all,
            "linf": linf_all,
            "target_model": target_name,
            "model_order": MODEL_NAMES,
            "common_xStart_path": common_path,
            "seed_output_root": seed_root,
            "selection_seed": args.seed,
        }
        _atomic_torch_save(all_payload, all_keys_path)
        _atomic_torch_save(
            make_subset_payload(
                x_all, x_adv_all, y_all, success_all, target_correct_all,
                non_target_fooled_all, predictions_all, linf_all, sample_indices, success_all
            ), working_path,
        )
        _atomic_torch_save(
            make_subset_payload(
                x_all, x_adv_all, y_all, success_all, target_correct_all,
                non_target_fooled_all, predictions_all, linf_all, sample_indices, ~success_all
            ), non_working_path,
        )

        with open(csv_path + ".tmp", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["sample", "dataset_index", "true_label"]
                + [f"clean_pred_{name}" for name in MODEL_NAMES]
                + [f"adv_pred_{name}" for name in MODEL_NAMES]
                + ["target_correct", "non_target_fooled", "success", "linf"]
            )
            for i in range(len(y_all)):
                writer.writerow(
                    [i, int(sample_indices[i]), int(y_all[i])]
                    + initial_predictions[i].tolist()
                    + predictions_all[i].tolist()
                    + [
                        bool(target_correct_all[i]),
                        bool(non_target_fooled_all[i]),
                        bool(success_all[i]),
                        float(linf_all[i]),
                    ]
                )
        os.replace(csv_path + ".tmp", csv_path)

        _write_predictions_json(
            predictions_path,
            target_name=target_name,
            seed=args.seed,
            num_per_class=args.num_per_class,
            y_all=y_all,
            sample_indices=sample_indices,
            initial_predictions=initial_predictions,
            predictions_all=predictions_all,
            success_all=success_all,
            target_correct_all=target_correct_all,
            non_target_fooled_all=non_target_fooled_all,
            linf_all=linf_all,
        )

        metrics = {
            "target_model": target_name,
            "target_display_name": DISPLAY_NAMES[target_name],
            "target_id": target_idx,
            "model_order": MODEL_NAMES,
            "non_target_models": [n for n in MODEL_NAMES if n != target_name],
            "num_models": 6,
            "num_classes": num_classes,
            "num_per_class": args.num_per_class,
            "total_samples": int(len(y_all)),
            "class_counts": class_counts,
            "common_xStart": True,
            "common_xStart_path": common_path,
            "seed_output_root": seed_root,
            "selection_seed": args.seed,
            "random_selection_rule": f"randomly select {args.num_per_class} samples per class; same xStart reused for all six targets",
            "target_correctness_filter": False,
            "clean_accuracy_common_xStart": clean_accuracy_by_model,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "epsilon": args.epsilon,
            "alpha": attacker.alpha,
            "beta": attacker.beta,
            "algorithm": "older Skeleton Key formulation: f_blend=sum sign(non-target CE gradients); f_target=sign(target CE gradient)*I(p_target(y)<0.5); x_adv=x_adv+alpha*f_blend-beta*f_target",
            "gradient_rule": "per-sample gradients obtained from summed per-sample CE; no batch-average loss",
            "success_definition": "target model correct AND all five non-target models incorrect",
            "working_keys": working_count,
            "non_working_keys": non_working_count,
            "yield_rate": yield_rate,
            "yield_rate_percent": 100.0 * yield_rate,
            "target_correct_after_attack": int(target_correct_all.sum().item()),
            "all_non_targets_fooled": int(non_target_fooled_all.sum().item()),
            "max_linf": float(linf_all.max().item()),
            "mean_linf": float(linf_all.mean().item()),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "resume_enabled": True,
            "persistent_batch_dir": batch_dir,
            "files": {
                "clean": clean_path,
                "all_skeleton_keys": all_keys_path,
                "working_keys": working_path,
                "non_working_keys": non_working_path,
                "csv": csv_path,
                "predictions": predictions_path,
                "log": log_path,
                "metrics": metrics_path,
                "common_xStart": common_path,
                "progress": progress_path,
                "batches": batch_dir,
            },
        }
        _atomic_json_save(metrics, metrics_path)
        _atomic_json_save({
            "status": "completed",
            "target_model": target_name,
            "seed": args.seed,
            "total_samples": len(y_all),
            "working_keys": working_count,
            "yield_rate": yield_rate,
            "completed_at": finished_at,
        }, completion_path)
        _atomic_json_save({
            **json.load(open(progress_path)),
            "status": "completed",
            "next_sample": len(y_all),
            "last_completed_sample": len(y_all) - 1,
            "processed_samples": len(y_all),
            "working_keys": working_count,
            "running_yield": yield_rate,
            "completed_at": finished_at,
        }, progress_path)

        print("\n" + "=" * 80)
        print("FINAL SKELETON KEY RESULTS")
        print("=" * 80)
        print(f"Target model       : {DISPLAY_NAMES[target_name]}")
        print(f"Total samples      : {len(y_all)}")
        print(f"Working keys       : {working_count}")
        print(f"Non-working keys   : {non_working_count}")
        print(f"Yield rate         : {yield_rate:.6f} ({100.0 * yield_rate:.2f}%)")
        print(f"Max L_inf          : {linf_all.max().item():.8f}")
        print(f"Persistent batches : {batch_dir}")
        print(f"Metrics            : {metrics_path}")
        print(f"Predictions        : {predictions_path}")
        print(f"Log                : {log_path}")
        print("Experiment completed and checkpointed.")

    except Exception:
        print("\nERROR: Skeleton Key experiment failed.")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def parse_target(value: str) -> str:
    """
    Parse target model from either numeric ID or model-name alias.

    Target mapping:
        0 = ViT-L/16
        1 = CaiT-XXS24
        2 = WRN-28-10
        3 = MambaVision-T
        4 = DM-R18
        5 = FAT-R18
    """
    aliases = {
        # ViT
        "0": "ViT",
        "vit": "ViT",
        "vit-l/16": "ViT",
        "vit_l_16": "ViT",

        # CaiT
        "1": "CAIT",
        "cait": "CAIT",
        "cait-xxs24": "CAIT",

        # WRN
        "2": "WRN",
        "wrn": "WRN",
        "wrn-28-10": "WRN",
        "wrn_28_10": "WRN",

        # Mamba
        "3": "MAMBA",
        "mamba": "MAMBA",
        "mambavision-t": "MAMBA",
        "mamba_vision_t": "MAMBA",

        # DM-R18
        "4": "DM_R18",
        "dm": "DM_R18",
        "dm-r18": "DM_R18",
        "dm_r18": "DM_R18",

        # FAT-R18
        "5": "FAT_R18",
        "fat": "FAT_R18",
        "fat-r18": "FAT_R18",
        "fat_r18": "FAT_R18",
    }

    raw = str(value).strip()

    if raw in MODEL_NAMES:
        return raw

    key = raw.lower()

    if key in aliases:
        return aliases[key]

    valid = (
        "0=ViT, 1=CAIT, 2=WRN, "
        "3=MAMBA, 4=DM_R18, 5=FAT_R18"
    )

    raise argparse.ArgumentTypeError(
        f"Unknown target {value!r}. Use one of: {valid}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable batched Skeleton Key experiment on one GPU using all six defense models."
    )
    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        help="Target model: 0=ViT, 1=CAIT, 2=WRN, 3=MAMBA, 4=DM_R18, 5=FAT_R18",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="Visible CUDA device index to use (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the shared random class-balanced xStart.")
    parser.add_argument("--num-per-class", type=int, default=200,
                        help="Random validation samples per CIFAR-10 class (default: 200 => 2000 total).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Attack batch size. Larger is faster if GPU memory permits.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Skeleton Key attack iterations.")
    parser.add_argument("--epsilon", type=float, default=0.031,
                        help="Maximum Linf perturbation. Default: 0.031.")
    parser.add_argument("--output-root", type=str, default=PERSISTENT_ROOT,
                        help="Persistent output root.")
    parser.add_argument("--restart-target", action="store_true",
                        help="Delete the target's previous checkpoints/results and start over intentionally.")
    args = parser.parse_args()
    if args.num_per_class <= 0:
        parser.error("--num-per-class must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("PRE-FLIGHT CHECK")
    print("=" * 80)
    print(f"Script directory : {_this_dir}")
    print(f"Target           : {DISPLAY_NAMES[args.target]} ({args.target})")
    print(f"GPU ID           : {args.gpu_id}")
    print(f"Seed             : {args.seed}")
    print(f"Samples/class    : {args.num_per_class}")
    print(f"Total samples    : {args.num_per_class * 10}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Steps            : {args.steps}")
    print(f"Epsilon          : {args.epsilon}")
    print(f"Output root      : {args.output_root}")

    required_project_files = [
        os.path.join(_this_dir, "Utilities", "DataManagerPytorch.py"),
        os.path.join(_this_dir, "core", "models", "TransformerModels.py"),
        os.path.join(_this_dir, "core", "models", "mamba_model.py"),
    ]
    for path in required_project_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required project file missing: {path}")

    for key, rel_path in CHECKPOINT_PATHS.items():
        resolved = _resolve_path(rel_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Checkpoint missing for {key}: {resolved}")

    if args.output_root == "":
        raise ValueError("--output-root cannot be empty")

    os.makedirs(args.output_root, exist_ok=True)
    run_resumable(args)


if __name__ == "__main__":
    main()
