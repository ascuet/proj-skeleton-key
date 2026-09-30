#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import AttackWrappersAPGD_CE_batch as APGD
from Utilities import DataManagerPytorch as DMP
import importlib
WRN = importlib.import_module("core.models.wideresnetwithswish")
CAIT = importlib.import_module("core.models.cait_model")
from core.models.vit_model import create_vit_2
from core.models.mamba_model import create_mamba

from spikingjelly.clock_driven import neuron, surrogate, functional
from spikingjelly.clock_driven.model import sew_resnet

NUM_CLASSES = 10
DATASET = "cifar10"
DEFAULT_TOTAL_SAMPLES = 1000
DEFAULT_NUM_PER_CLASS = 100
DEFAULT_EPSILON = 0.031
DEFAULT_STEPS = 100
DEFAULT_ETA_START_FACTOR = 2.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_GPU_IDS = [0, 1, 2, 3, 4, 5]

CHECKPOINTS = {
    "WRN": PROJECT_ROOT / "checkpoint" / "Wrn_28_10.pt",
    "CAIT": PROJECT_ROOT / "checkpoint" / "Cait_xxs.pt",
    "VIT": PROJECT_ROOT / "checkpoint" / "Vit_L_16.pt",
    "MAMBA": PROJECT_ROOT / "checkpoint" / "MambaVision_T.pt",
    "DM_R18": PROJECT_ROOT / "checkpoint" / "snn_sew_resnet.pt",
    "FAT_R18": PROJECT_ROOT / "checkpoint" / "snn_resnet18_cifar10_5_219_7315.pth",
}

DISPLAY_NAMES = {
    "WRN": "WRN-28-10-SiLU",
    "CAIT": "CaiT-XXS-12",
    "VIT": "ViT-L/16",
    "MAMBA": "MambaVision-T",
    "DM_R18": "DM-R18",
    "FAT_R18": "FAT-R18",
}

MODEL_ORDER = ["WRN", "CAIT", "VIT", "MAMBA", "DM_R18", "FAT_R18"]


@dataclass
class Tee:
    terminal: Any
    file: Any

    def write(self, data: str) -> None:
        self.terminal.write(data)
        self.terminal.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Six-GPU APGD-CE generation on random CIFAR-10 training samples."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed controlling the common xStart selection (default: 42).")
    parser.add_argument("--num-per-class", type=int, default=DEFAULT_NUM_PER_CLASS,
                        help="Random samples per CIFAR-10 class (default: 100).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="APGD batch size to try first (default: 32).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="APGD iterations (default: 100).")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help="L_inf epsilon (default: 0.031).")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated physical/logical GPU IDs, one model per GPU.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing model outputs. Common xStart is still verified by seed/count.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normal_autograd_clone(tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(False):
        return tensor.detach().clone()


def parse_gpu_ids(spec: str) -> List[int]:
    ids = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(ids) != 6:
        raise ValueError(f"Exactly 6 GPU IDs are required; got {ids}")
    if len(set(ids)) != 6:
        raise ValueError(
            f"GPU IDs must be six distinct devices for the six-model experiment; got {ids}"
        )
    return ids


def ensure_checkpoints() -> None:
    missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))


def extract_state_dict(blob: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(blob, dict):
        return blob
    for key in ("model_state_dict", "unaveraged_model_state_dict", "state_dict", "model"):
        if key in blob:
            print(f"  Using checkpoint['{key}']", flush=True)
            return blob[key]
    return blob


def strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
        out[k] = value
    return out


def load_checkpoint(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checked(model: nn.Module, state: Dict[str, torch.Tensor], name: str,
                 zero_prefix: bool = False) -> nn.Module:
    state = strip_common_prefixes(state)
    if zero_prefix:
        state = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    real_unexpected = list(unexpected)
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"{name} checkpoint mismatch. Missing={real_missing[:20]} Unexpected={real_unexpected[:20]}"
        )
    return model


def load_wrn(device: torch.device) -> nn.Module:
    model = WRN.WideResNet(
        depth=28,
        width=10,
        num_classes=NUM_CLASSES,
        activation_fn=nn.SiLU,
    )
    state = extract_state_dict(load_checkpoint(CHECKPOINTS["WRN"]))
    model = load_checked(model, state, "WRN", zero_prefix=True)
    return nn.Sequential(model).to(device).eval()


def load_cait(device: torch.device) -> nn.Module:
    cfg = CAIT.CAIT_CONFIGS["CaiT-XXS-12"]
    backbone = CAIT.CaiT(
        image_size=(32, 32),
        patch_size=cfg["patch_size"],
        num_classes=NUM_CLASSES,
        dim=cfg["dim"],
        depth=cfg["depth"],
        cls_depth=cfg["cls_depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
        dim_head=cfg["dim_head"],
    )
    model = nn.Sequential(backbone)
    state = extract_state_dict(load_checkpoint(CHECKPOINTS["CAIT"]))
    state = strip_common_prefixes(state)
    state = {(k if k.startswith("0.") else "0." + k): v for k, v in state.items()}
    model = load_checked(model, state, "CAIT")
    return model.to(device).eval()


def load_vit(device: torch.device) -> nn.Module:
    vit_wrapper = create_vit_2(
        name="ViT-L_16",
        num_classes=NUM_CLASSES,
        input_size=224,
    )
    model = nn.Sequential(vit_wrapper)
    state = extract_state_dict(load_checkpoint(CHECKPOINTS["VIT"]))
    state = strip_common_prefixes(state)

    target_keys = set(model.state_dict().keys())
    if state and not any(k in target_keys for k in state):
        candidate = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
        if sum(k in target_keys for k in candidate) > 0:
            state = candidate

    model = load_checked(model, state, "VIT")
    return model.to(device).eval()


def load_mamba(device: torch.device) -> nn.Module:
    mamba_wrapper = create_mamba(
        name="mamba_vision_T",
        num_classes=NUM_CLASSES,
        input_size=224,
        pretrained=False,
    )

    checkpoint = load_checkpoint(CHECKPOINTS["MAMBA"])
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Mamba checkpoint must be a dict, got {type(checkpoint)}")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        print("  Mamba checkpoint format: model_state_dict", flush=True)
    elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state = checkpoint
        print("  Mamba checkpoint format: plain state_dict", flush=True)
    else:
        candidates = {k: v for k, v in checkpoint.items() if isinstance(v, dict)}
        if not candidates:
            raise RuntimeError(f"Unrecognised Mamba checkpoint format: {list(checkpoint.keys())[:20]}")
        selected = next(iter(candidates))
        state = candidates[selected]
        print(f"  Mamba checkpoint format: dict key {selected!r}", flush=True)

    if not isinstance(state, dict) or not state:
        raise RuntimeError("Mamba checkpoint state_dict is empty or invalid.")

    first_key = next(iter(state))
    for prefix in ("module.0.", "module.", ""):
        if prefix == "" or first_key.startswith(prefix):
            if prefix:
                state = {k[len(prefix):]: v for k, v in state.items()}
            print(f"  Mamba checkpoint prefix removed: {prefix!r}", flush=True)
            break

    live_keys = set(mamba_wrapper.state_dict().keys())
    if not any(k in live_keys for k in state):
        candidate = {
            (k if k.startswith("mamba.") else "mamba." + k): v
            for k, v in state.items()
        }
        if sum(k in live_keys for k in candidate) > 0:
            state = candidate
            print('  Added "mamba." prefix (raw model -> MambaVisionWrapper)', flush=True)

    missing, unexpected = mamba_wrapper.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    real_unexpected = list(unexpected)
    if real_missing or real_unexpected:
        raise RuntimeError(
            "MambaVision-T checkpoint mismatch. "
            f"Missing={real_missing[:20]} Unexpected={real_unexpected[:20]}"
        )

    print(f"[MambaVision-T] Loaded {len(state)} tensors (0 checkpoint-only tensors ignored).", flush=True)
    return nn.Sequential(mamba_wrapper).to(device).eval()


class SNNMeanLogits(nn.Module):
    def __init__(self, snn: nn.Module) -> None:
        super().__init__()
        self.snn = snn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self.snn)
        output = self.snn(x)
        if output.ndim == 3:
            output = output.mean(dim=0)
        elif output.ndim != 2:
            raise RuntimeError(f"Unexpected SNN output shape: {tuple(output.shape)}")
        return output


def build_snn() -> nn.Module:
    return sew_resnet.multi_step_sew_resnet18(
        T=5,
        num_classes=NUM_CLASSES,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepParametricLIFNode,
        surrogate_function=surrogate.ATan(),
    )


def load_snn(device: torch.device, kind: str) -> nn.Module:
    model = build_snn()
    checkpoint = load_checkpoint(CHECKPOINTS[kind])
    state = extract_state_dict(checkpoint)
    state = strip_common_prefixes(state)
    normalized = {}
    for key, value in state.items():
        if key.startswith("0."):
            key = key[2:]
        normalized[key] = value

    target = model.state_dict()
    compatible = {
        k: v for k, v in normalized.items()
        if k in target and tuple(v.shape) == tuple(target[k].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"{kind} checkpoint mismatch. Missing={real_missing[:20]} Unexpected={list(unexpected)[:20]}"
        )
    return SNNMeanLogits(model).to(device).eval()


def load_model(name: str, device: torch.device) -> nn.Module:
    if name == "WRN":
        return load_wrn(device)
    if name == "CAIT":
        return load_cait(device)
    if name == "VIT":
        return load_vit(device)
    if name == "MAMBA":
        return load_mamba(device)
    if name in ("DM_R18", "FAT_R18"):
        return load_snn(device, name)
    raise KeyError(name)

def create_or_load_common_xstart(seed: int, num_per_class: int,
                                  output_root: Path,
                                  overwrite: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path]:
    if num_per_class <= 0:
        raise ValueError("--num-per-class must be positive")

    total = num_per_class * NUM_CLASSES
    common_path = output_root / "common_xStart.pt"
    manifest_path = output_root / "common_xStart_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)

    if common_path.exists() and not overwrite:
        payload = torch.load(common_path, map_location="cpu", weights_only=False)
        required = {"x_clean", "y", "sample_indices", "seed", "num_per_class", "class_counts"}
        missing = required - set(payload.keys())
        if missing:
            raise RuntimeError(f"Existing common xStart missing fields: {sorted(missing)}")
        if int(payload["seed"]) != seed or int(payload["num_per_class"]) != num_per_class:
            raise RuntimeError(
                f"Existing common_xStart was created for seed={payload['seed']} and "
                f"num_per_class={payload['num_per_class']}; requested seed={seed}, num_per_class={num_per_class}. "
                f"Use another seed or --overwrite intentionally."
            )
        x_clean = normal_autograd_clone(payload["x_clean"].cpu())
        y = normal_autograd_clone(payload["y"].long().cpu())
        sample_indices = normal_autograd_clone(payload["sample_indices"].long().cpu())
        expected_counts = [num_per_class] * NUM_CLASSES
        actual_counts = torch.bincount(y, minlength=NUM_CLASSES).tolist()
        if x_clean.shape[0] != total or actual_counts != expected_counts:
            raise RuntimeError(
                f"Existing common xStart is inconsistent: total={x_clean.shape[0]}, "
                f"class_counts={actual_counts}, expected={expected_counts}"
            )
        print(f"[COMMON] Reusing {common_path}")
        print(f"[COMMON] Verified class counts: {actual_counts}")
        return x_clean, y, sample_indices, common_path

    set_global_seed(seed)
    train_loader = DMP.GetCIFAR10Training(imgSize=32, batchSize=256)
    dataset = train_loader.dataset
    n = len(dataset)

    # Extract labels without using model predictions.
    labels = torch.tensor([int(dataset[i][1]) for i in range(n)], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)

    selected_indices: List[int] = []
    for c in range(NUM_CLASSES):
        class_indices = torch.nonzero(labels == c, as_tuple=False).flatten()
        if class_indices.numel() < num_per_class:
            raise RuntimeError(
                f"Class {c} has only {class_indices.numel()} samples; cannot select {num_per_class}."
            )
        perm = torch.randperm(class_indices.numel(), generator=generator)
        chosen = class_indices[perm[:num_per_class]].tolist()
        selected_indices.extend(int(i) for i in chosen)

    order_gen = torch.Generator().manual_seed(seed + 1)
    order = torch.randperm(len(selected_indices), generator=order_gen).tolist()
    selected_indices = [selected_indices[i] for i in order]

    x_list: List[torch.Tensor] = []
    y_list: List[int] = []
    for idx in selected_indices:
        x_i, y_i = dataset[idx]
        x_list.append(x_i.detach().cpu())
        y_list.append(int(y_i))

    x_clean = normal_autograd_clone(torch.stack(x_list))
    y = normal_autograd_clone(torch.tensor(y_list, dtype=torch.long))
    sample_indices = normal_autograd_clone(torch.tensor(selected_indices, dtype=torch.long))
    class_counts = torch.bincount(y, minlength=NUM_CLASSES).tolist()

    if x_clean.shape[0] != total or class_counts != [num_per_class] * NUM_CLASSES:
        raise RuntimeError(
            f"Common xStart selection error: total={x_clean.shape[0]}, class_counts={class_counts}"
        )

    payload = {
        "x_clean": x_clean,
        "y": y,
        "sample_indices": sample_indices,
        "seed": seed,
        "num_per_class": num_per_class,
        "total_samples": total,
        "class_counts": class_counts,
        "dataset": "CIFAR-10 training set",
        "selection": "random class-balanced; NO correctly-classified filter",
        "input_range": "[0,1] raw CIFAR-10 tensors",
    }
    torch.save(payload, common_path)

    manifest = dict(payload)
    manifest.pop("x_clean", None)
    manifest["x_clean_shape"] = list(x_clean.shape)
    manifest["y"] = y.tolist()
    manifest["sample_indices"] = sample_indices.tolist()
    manifest["class_counts"] = [int(v) for v in class_counts]
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[COMMON] Created {common_path}")
    print(f"[COMMON] Selected samples : {len(y)}")
    print(f"[COMMON] Class counts      : {class_counts}")
    return x_clean, y, sample_indices, common_path


def predict_tensor(model: nn.Module, x: torch.Tensor, device: torch.device,
                   batch_size: int = 256) -> torch.Tensor:
    ds = TensorDataset(x, torch.zeros(x.shape[0], dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu())
    return torch.cat(preds)


def build_attack_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    return DMP.TensorToDataLoader(
        x, y,
        transforms=None,
        batchSize=batch_size,
        randomizer=None,
    )


def run_apgd_with_fallback(model: nn.Module, name: str, x_clean: torch.Tensor,
                           y: torch.Tensor, device: torch.device,
                           epsilon: float, steps: int, requested_bs: int) -> Tuple[torch.Tensor, int]:
    eta_start = DEFAULT_ETA_START_FACTOR * epsilon
    candidates = []
    b = requested_bs
    while b >= 1:
        candidates.append(b)
        b //= 2
    # preserve a few common values if requested_bs is not a power of two
    for b in (32, 16, 8, 4, 2, 1):
        if b <= requested_bs and b not in candidates:
            candidates.append(b)
    candidates = sorted(set(candidates), reverse=True)

    x_clean_attack = normal_autograd_clone(x_clean.cpu())
    y_attack = normal_autograd_clone(y.cpu()).long()

    for attack_bs in candidates:
        print(f"[{name}] Trying APGD batch size {attack_bs}", flush=True)
        loader = build_attack_loader(x_clean_attack, y_attack, attack_bs)
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            with torch.inference_mode(False), torch.enable_grad():
                adv_loader = APGD.APGDNativePytorch_CE(
                    device=device,
                    dataLoader=loader,
                    model=model,
                    modelPlus=type("ModelPlus", (), {
                        "modelName": name,
                        "formatDataLoader": lambda self, dl: dl,
                    })(),
                    epsilonMax=epsilon,
                    etaStart=eta_start,
                    numSteps=steps,
                    clipMin=0.0,
                    clipMax=1.0,
                    targeted=False,
                    random_start=False,
                )
            x_adv, y_adv = DMP.DataLoaderToTensor(adv_loader)
            y_adv = y_adv.long().cpu()
            if not torch.equal(y_adv, y.cpu()):
                raise RuntimeError(f"{name}: labels changed during APGD generation")
            print(f"[{name}] APGD succeeded with batch size {attack_bs}", flush=True)
            return x_adv.cpu(), attack_bs
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda error" not in message:
                raise
            print(f"[{name}] CUDA memory failure at batch size {attack_bs}; trying smaller.", flush=True)
            del loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raise RuntimeError(f"{name}: APGD failed at all batch sizes down to 1.")


def save_results(name: str, model: nn.Module, x_clean: torch.Tensor, x_adv: torch.Tensor,
                 y: torch.Tensor, sample_indices: torch.Tensor, device: torch.device,
                 output_dir: Path, seed: int, num_per_class: int,
                 epsilon: float, steps: int, used_bs: int) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean.pt"
    adv_path = output_dir / f"{DISPLAY_NAMES[name].replace('/', '_')}_eps_0p031_steps_{steps}_n_{len(y)}.pt"
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "results.csv"

    torch.save(
        {
            "x_clean": x_clean,
            "y": y,
            "sample_indices": sample_indices,
            "seed": seed,
            "num_per_class": num_per_class,
            "total_samples": len(y),
            "dataset": "CIFAR-10 training set",
            "selection": "random class-balanced; no correctness filter",
            "model": name,
        },
        clean_path,
    )

    clean_pred = predict_tensor(model, x_clean, device)
    adv_pred = predict_tensor(model, x_adv, device)

    clean_correct = clean_pred.eq(y.cpu())
    adv_correct = adv_pred.eq(y.cpu())
    clean_correct_count = int(clean_correct.sum().item())
    adv_correct_count = int(adv_correct.sum().item())

    clean_acc = clean_correct.float().mean().item()
    robust_acc = adv_correct.float().mean().item()
    attack_success_all = 1.0 - robust_acc

    if clean_correct_count > 0:
        conditional_success = float(
            (clean_correct & ~adv_correct).sum().item() / clean_correct_count
        )
        conditional_robust = float(
            (clean_correct & adv_correct).sum().item() / clean_correct_count
        )
    else:
        conditional_success = None
        conditional_robust = None

    diff = (x_adv - x_clean).abs().flatten(1)
    linf_per_sample = diff.max(dim=1).values
    max_linf = float(linf_per_sample.max().item())
    mean_linf = float(linf_per_sample.mean().item())

    if max_linf > epsilon + 1e-6:
        raise RuntimeError(f"{name}: L_inf violation: {max_linf} > {epsilon}")

    torch.save(
        {
            "x_clean": x_clean,
            "x_adv": x_adv,
            "y": y,
            "sample_indices": sample_indices,
            "model": name,
            "display_name": DISPLAY_NAMES[name],
            "checkpoint": str(CHECKPOINTS[name]),
            "dataset": "CIFAR-10 training set",
            "selection": "random class-balanced; no correctness filter",
            "seed": seed,
            "num_per_class": num_per_class,
            "num_samples": len(y),
            "batch_size": used_bs,
            "attack": "APGD-CE",
            "epsilon": epsilon,
            "steps": steps,
            "eta_start": DEFAULT_ETA_START_FACTOR * epsilon,
            "clean_accuracy": clean_acc,
            "clean_correct_count": clean_correct_count,
            "robust_accuracy": robust_acc,
            "robust_correct_count": adv_correct_count,
            "attack_success_rate": attack_success_all,
            "conditional_attack_success_rate_on_clean_correct": conditional_success,
            "conditional_robust_accuracy_on_clean_correct": conditional_robust,
            "max_linf": max_linf,
            "mean_linf": mean_linf,
        },
        adv_path,
    )

    rows = []
    for i in range(len(y)):
        rows.append([
            i,
            int(sample_indices[i]),
            int(y[i]),
            int(clean_pred[i]),
            int(adv_pred[i]),
            bool(clean_correct[i]),
            bool(adv_correct[i]),
            float(linf_per_sample[i]),
        ])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "position", "dataset_index", "true_label",
            "clean_prediction", "adv_prediction",
            "clean_correct", "adv_correct", "linf",
        ])
        writer.writerows(rows)

    metrics = {
        "model": name,
        "display_name": DISPLAY_NAMES[name],
        "seed": seed,
        "dataset": "CIFAR-10 training set",
        "selection": "random class-balanced; no correctly-classified filter",
        "num_per_class": num_per_class,
        "total_samples": len(y),
        "class_counts": torch.bincount(y.cpu(), minlength=NUM_CLASSES).tolist(),
        "attack": "APGD-CE",
        "batch_size": used_bs,
        "steps": steps,
        "epsilon": epsilon,
        "eta_start": DEFAULT_ETA_START_FACTOR * epsilon,
        "clean_correct_count": clean_correct_count,
        "clean_accuracy": clean_acc,
        "robust_correct_count": adv_correct_count,
        "robust_accuracy": robust_acc,
        "attack_success_rate": attack_success_all,
        "conditional_attack_success_rate_on_clean_correct": conditional_success,
        "conditional_robust_accuracy_on_clean_correct": conditional_robust,
        "max_linf": max_linf,
        "mean_linf": mean_linf,
        "files": {
            "clean": str(clean_path),
            "adversarial": str(adv_path),
            "metrics": str(metrics_path),
            "csv": str(csv_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def worker(model_name: str, gpu_id: int, args_dict: Dict[str, Any],
           x_clean: torch.Tensor, y: torch.Tensor, sample_indices: torch.Tensor,
           output_root: str) -> None:
    args = argparse.Namespace(**args_dict)
    target_dir = Path(output_root) / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "generation.log"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    start_time = time.time()
    try:
        print("\n" + "=" * 80)
        print(f"{DISPLAY_NAMES[model_name]} APGD-CE WORKER")
        print("=" * 80)
        print(f"GPU ID             : {gpu_id}")
        print(f"Seed               : {args.seed}")
        print(f"Samples            : {len(y)}")
        print(f"Samples/class      : {args.num_per_class}")
        print(f"Batch size         : {args.batch_size}")
        print(f"Steps              : {args.steps}")
        print(f"Epsilon            : {args.epsilon}")
        print(f"Output directory   : {target_dir}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Device             : {device}")
        print(f"GPU                : {torch.cuda.get_device_name(gpu_id)}")

        set_global_seed(args.seed + 1000 + gpu_id)
        print(f"Loading {DISPLAY_NAMES[model_name]}...")
        model = load_model(model_name, device)
        print(f"[{DISPLAY_NAMES[model_name]}] Model loaded successfully.")

        x_clean = normal_autograd_clone(x_clean.cpu())
        y = normal_autograd_clone(y.cpu()).long()
        sample_indices = normal_autograd_clone(sample_indices.cpu()).long()

        clean_pred = predict_tensor(model, x_clean, device)
        clean_acc = float(clean_pred.eq(y.cpu()).float().mean().item())
        print(f"Clean accuracy on common random xStart: {100.0 * clean_acc:.2f}%")

        print("Generating APGD-CE adversarial examples...")
        x_adv, used_bs = run_apgd_with_fallback(
            model=model,
            name=model_name,
            x_clean=x_clean,
            y=y,
            device=device,
            epsilon=args.epsilon,
            steps=args.steps,
            requested_bs=args.batch_size,
        )

        metrics = save_results(
            name=model_name,
            model=model,
            x_clean=x_clean.cpu(),
            x_adv=x_adv.cpu(),
            y=y.cpu(),
            sample_indices=sample_indices.cpu(),
            device=device,
            output_dir=target_dir,
            seed=args.seed,
            num_per_class=args.num_per_class,
            epsilon=args.epsilon,
            steps=args.steps,
            used_bs=used_bs,
        )

        metrics["elapsed_seconds"] = time.time() - start_time
        metrics["started_at"] = datetime.fromtimestamp(start_time).isoformat()
        metrics["finished_at"] = datetime.now().isoformat()
        (target_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        print("\n" + "=" * 80)
        print(f"FINAL RESULTS - {DISPLAY_NAMES[model_name]}")
        print("=" * 80)
        print(f"Clean accuracy                  : {100.0 * metrics['clean_accuracy']:.2f}%")
        print(f"Robust accuracy                 : {100.0 * metrics['robust_accuracy']:.2f}%")
        print(f"Attack success (all 1000)       : {100.0 * metrics['attack_success_rate']:.2f}%")
        if metrics["conditional_attack_success_rate_on_clean_correct"] is not None:
            print(
                "Attack success (clean-correct)  : "
                f"{100.0 * metrics['conditional_attack_success_rate_on_clean_correct']:.2f}%"
            )
        print(f"Max L_inf                        : {metrics['max_linf']:.8f}")
        print(f"Mean L_inf                       : {metrics['mean_linf']:.8f}")
        print(f"Effective APGD batch size       : {used_bs}")
        print(f"Metrics                          : {target_dir / 'metrics.json'}")
        print(f"Adversarial dataset              : {target_dir / next(p.name for p in target_dir.glob('*eps_0p031*.pt'))}")
        print(f"Elapsed seconds                  : {metrics['elapsed_seconds']:.2f}")

    except Exception as exc:
        print("\nERROR: worker failed")
        traceback.print_exc()
        error_payload = {
            "model": model_name,
            "gpu_id": gpu_id,
            "seed": args.seed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now().isoformat(),
        }
        (target_dir / "error.json").write_text(json.dumps(error_payload, indent=2))
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    ensure_checkpoints()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    total_samples = args.num_per_class * NUM_CLASSES
    if total_samples != DEFAULT_TOTAL_SAMPLES:
        print(
            f"WARNING: num_per_class={args.num_per_class} gives total={total_samples}; "
            "the requested standard experiment uses 1000 total samples."
        )

    output_root = (
        PROJECT_ROOT
        / "generated_adversarial_examples"
        / "apgd-ce"
        / "random-training"
        / DATASET
        / f"seed_{args.seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    master_log_path = output_root / "generation_master.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    master_log = open(master_log_path, "a", buffering=1)
    sys.stdout = Tee(original_stdout, master_log)
    sys.stderr = Tee(original_stderr, master_log)

    try:
        print("=" * 80)
        print("APGD-CE RANDOM CIFAR-10 TRAINING DATA - SIX GPU EXPERIMENT")
        print("=" * 80)
        print(f"Project root       : {PROJECT_ROOT}")
        print(f"Seed               : {args.seed}")
        print(f"Samples/class      : {args.num_per_class}")
        print(f"Total samples      : {total_samples}")
        print(f"Batch size         : {args.batch_size}")
        print(f"Steps              : {args.steps}")
        print(f"Epsilon            : {args.epsilon}")
        print(f"GPU IDs            : {gpu_ids}")
        print(f"Output root        : {output_root}")
        print("Selection          : RANDOM, class-balanced, NO correctness filter")
        print("Dataset            : CIFAR-10 TRAINING set")
        print("Input              : raw [0,1] CIFAR-10 tensors")
        print()

        x_clean, y, sample_indices, common_path = create_or_load_common_xstart(
            seed=args.seed,
            num_per_class=args.num_per_class,
            output_root=output_root,
            overwrite=args.overwrite,
        )

        print(f"[COMMON] Common xStart path: {common_path}")
        print(f"[COMMON] x_clean shape     : {tuple(x_clean.shape)}")
        print(f"[COMMON] y shape           : {tuple(y.shape)}")
        print()
        print("Launching six model workers...")

        args_dict = vars(args).copy()
        processes: List[mp.Process] = []
        for model_name, gpu_id in zip(MODEL_ORDER, gpu_ids):
            p = mp.Process(
                target=worker,
                args=(model_name, gpu_id, args_dict, x_clean, y, sample_indices, str(output_root)),
                name=f"APGD-{model_name}-GPU{gpu_id}",
            )
            p.start()
            processes.append(p)
            print(f"Started {model_name} on GPU {gpu_id}: PID={p.pid}")

        exit_codes = {}
        for model_name, proc in zip(MODEL_ORDER, processes):
            proc.join()
            exit_codes[model_name] = proc.exitcode
            print(f"Finished {model_name}: exit_code={proc.exitcode}")

        summary: Dict[str, Any] = {
            "experiment": "APGD-CE on random CIFAR-10 training samples",
            "seed": args.seed,
            "num_per_class": args.num_per_class,
            "total_samples": total_samples,
            "selection": "random class-balanced; no correctly-classified filter",
            "dataset": "CIFAR-10 training set",
            "epsilon": args.epsilon,
            "steps": args.steps,
            "requested_batch_size": args.batch_size,
            "gpu_ids": gpu_ids,
            "common_xStart": str(common_path),
            "workers": {},
            "master_log": str(master_log_path),
            "finished_at": datetime.now().isoformat(),
        }

        for model_name in MODEL_ORDER:
            metrics_path = output_root / model_name / "metrics.json"
            if metrics_path.exists():
                try:
                    summary["workers"][model_name] = json.loads(metrics_path.read_text())
                except Exception:
                    summary["workers"][model_name] = {"metrics_read_error": True}
            else:
                summary["workers"][model_name] = {
                    "exit_code": exit_codes[model_name],
                    "status": "failed or incomplete",
                }

        summary_path = output_root / "all_models_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print("\n" + "=" * 80)
        print("SIX GPU EXPERIMENT COMPLETE")
        print("=" * 80)
        for name in MODEL_ORDER:
            print(f"{name:<8} exit_code={exit_codes[name]}")
        print(f"Common xStart   : {common_path}")
        print(f"Master log      : {master_log_path}")
        print(f"Summary JSON    : {summary_path}")

        if any(code != 0 for code in exit_codes.values()):
            raise SystemExit(1)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        master_log.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
