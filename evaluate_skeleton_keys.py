#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_NAMES = ["WRN", "CAIT", "VIT", "MAMBA", "DM_R18", "FAT_R18"]
DISPLAY_NAMES = {
    "WRN": "WRN-28-10",
    "CAIT": "CaiT-XXS24",
    "VIT": "ViT-L/16",
    "MAMBA": "MambaVision-T",
    "DM_R18": "DM-R18",
    "FAT_R18": "FAT-R18",
}
TARGET_DIR_NAMES = {
    "VIT": "ViT",
    "CAIT": "CAIT",
    "WRN": "WRN",
    "MAMBA": "MAMBA",
    "DM_R18": "DM_R18",
    "FAT_R18": "FAT_R18",
}


def default_skeleton_root(dataset: str) -> str:
    return os.path.join(PROJECT_ROOT, "generated_adversarial_examples", "skeleton_key", dataset)


def default_game_results(dataset: str) -> str:
    return os.path.join(
        PROJECT_ROOT, "ResultsForGaME", dataset, "GaME-results-mixed-6.json"
    )


def default_output_root(dataset: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "ResultsForGaME",
        dataset,
        "skeleton_key_repeated_probability",
    )


def parse_probabilities(text: str) -> np.ndarray:
    try:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Probabilities must be comma-separated numbers, e.g. "
            "0.1,0.2,0.2,0.1,0.2,0.2"
        ) from exc

    p = np.asarray(values, dtype=np.float64)
    if p.shape != (6,):
        raise argparse.ArgumentTypeError(
            f"Expected exactly 6 probabilities, got {len(values)}"
        )
    if not np.all(np.isfinite(p)):
        raise argparse.ArgumentTypeError("Probabilities must all be finite")
    if np.any(p < 0):
        raise argparse.ArgumentTypeError("Probabilities cannot be negative")
    total = float(p.sum())
    if total <= 0:
        raise argparse.ArgumentTypeError("Probability sum must be > 0")
    p = p / total
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the first working Skeleton Key from each of six target groups "
            "and randomly select the defending model for every repetition."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "tiny-imagenet"],
        help="Dataset associated with the Skeleton Key directory and outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Skeleton-Key generation seed, e.g. 42.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        required=True,
        help="N repetitions for EACH of the six first working Skeleton Keys.",
    )
    parser.add_argument(
        "--roll-seed",
        type=int,
        default=None,
        help="RNG seed for defender-model selection. Defaults to --seed.",
    )
    parser.add_argument(
        "--probability-mode",
        choices=["game", "equal", "custom"],
        default="game",
        help=(
            "Defender probabilities: 'game' reads GaME P_D from JSON, "
            "'equal' uses 1/6 each, 'custom' uses --probabilities."
        ),
    )
    parser.add_argument(
        "--probabilities",
        type=parse_probabilities,
        default=None,
        help=(
            "Six comma-separated probabilities in order "
            "WRN,CAIT,VIT,MAMBA,DM_R18,FAT_R18."
        ),
    )
    parser.add_argument(
        "--game-results",
        default=None,
        help="GaME JSON containing defender_strategy_P_D. Defaults by dataset.",
    )
    parser.add_argument(
        "--skeleton-root",
        default=None,
        help="Root containing seed_<seed>/<target>/working_keys.pt.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for experiment outputs. Defaults by dataset.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for the selected model.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def _load_torch_payload(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def canonicalize_target(name: str) -> str:
    mapping = {
        "ViT": "VIT",
        "VIT": "VIT",
        "CAIT": "CAIT",
        "WRN": "WRN",
        "MAMBA": "MAMBA",
        "DM_R18": "DM_R18",
        "FAT_R18": "FAT_R18",
    }
    if name not in mapping:
        raise ValueError(f"Unknown target model name: {name!r}")
    return mapping[name]


def load_first_working_keys(seed: int, skeleton_root: str):
    seed_root = os.path.join(skeleton_root, f"seed_{seed}")
    records: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []

    for model_name in MODEL_NAMES:
        target_dir = TARGET_DIR_NAMES[model_name]
        path = os.path.join(seed_root, target_dir, "working_keys.pt")

        if not os.path.exists(path):
            missing.append(path)
            continue

        payload = _load_torch_payload(path)
        for key in ("x_adv", "x_clean", "y"):
            if key not in payload:
                raise KeyError(f"{path} is missing required field '{key}'")

        x_adv = payload["x_adv"].detach().cpu()
        x_clean = payload["x_clean"].detach().cpu()
        y = payload["y"].detach().cpu().long()

        if x_adv.ndim != 4 or x_clean.ndim != 4:
            raise ValueError(f"Unexpected tensor dimensions in {path}")
        if len(x_adv) == 0:
            raise ValueError(f"No working keys found in {path}")
        if len(x_adv) != len(x_clean) or len(x_adv) != len(y):
            raise ValueError(f"Length mismatch in {path}")

        if "success" in payload:
            success = payload["success"].detach().cpu().bool()
            if not bool(success.all().item()):
                raise ValueError(
                    f"{path} is named working_keys.pt but contains a non-working key"
                )

        sample_indices = payload.get("sample_indices")
        if sample_indices is None:
            sample_indices = torch.arange(len(y), dtype=torch.long)
        else:
            sample_indices = sample_indices.detach().cpu().long()

        if len(sample_indices) != len(y):
            raise ValueError(f"sample_indices length mismatch in {path}")

        target_field = payload.get("target_model", target_dir)
        canonical_target = canonicalize_target(str(target_field))
        expected = canonicalize_target(target_dir)
        if canonical_target != expected:
            print(
                f"WARNING: {path} payload target={target_field!r} "
                f"but directory target={target_dir!r}"
            )

        # FIRST working key means tensor position 0 in the saved working_keys.pt.
        records[model_name] = {
            "target_model": canonical_target,
            "sample_index": int(sample_indices[0].item()),
            "x_adv": x_adv[0].clone(),
            "x_clean": x_clean[0].clone(),
            "y": int(y[0].item()),
            "source_file": os.path.abspath(path),
            "source_position": 0,
            "num_available_working_keys": int(len(y)),
            "original_predictions": (
                payload["predictions"][0].detach().cpu().long().tolist()
                if "predictions" in payload
                else None
            ),
        }

    if missing:
        raise FileNotFoundError(
            "Missing working-key files:\n" + "\n".join(missing)
        )
    if len(records) != 6:
        raise RuntimeError(f"Expected six first working keys, found {len(records)}")

    return records, seed_root


# -----------------------------------------------------------------------------
# Probability configuration
# -----------------------------------------------------------------------------

def load_game_probabilities(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GaME results not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "defender_strategy_P_D" not in data:
        raise KeyError("GaME JSON does not contain defender_strategy_P_D")
    return parse_probabilities(
        ",".join(str(v) for v in data["defender_strategy_P_D"])
    )


def get_probabilities(args: argparse.Namespace) -> Tuple[np.ndarray, str, Optional[str]]:
    if args.probability_mode == "equal":
        return np.full(6, 1.0 / 6.0, dtype=np.float64), "equal", None

    if args.probability_mode == "custom":
        if args.probabilities is None:
            raise ValueError("--probability-mode custom requires --probabilities")
        return args.probabilities.copy(), "custom", None

    game_path = args.game_results or default_game_results(args.dataset)
    return load_game_probabilities(game_path), "game", os.path.abspath(game_path)


# -----------------------------------------------------------------------------
# Direct model loading (no game_mixed_6.py dependency)
# -----------------------------------------------------------------------------

def add_project_root_to_path() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(blob):
    if not isinstance(blob, dict):
        return blob
    for key in ("model_state_dict", "unaveraged_model_state_dict", "state_dict", "model"):
        if key in blob:
            return blob[key]
    return blob


def strip_common_prefixes(state):
    out = {}
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


CHECKPOINTS = {
    "WRN": os.path.join(PROJECT_ROOT, "checkpoint", "Wrn_28_10.pt"),
    "CAIT": os.path.join(PROJECT_ROOT, "checkpoint", "Cait_xxs.pt"),
    "VIT": os.path.join(PROJECT_ROOT, "checkpoint", "Vit_L_16.pt"),
    "MAMBA": os.path.join(PROJECT_ROOT, "checkpoint", "MambaVision_T.pt"),
    "DM_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_sew_resnet.pt"),
    "FAT_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_resnet18_cifar10_5_219_7315.pth"),
}


def load_checked(model: nn.Module, state, name: str, zero_prefix: bool = False):
    state = strip_common_prefixes(state)
    if zero_prefix:
        state = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint mismatch. Missing={real_missing[:20]} "
            f"Unexpected={list(unexpected)[:20]}"
        )
    return model


def load_wrn(device: torch.device):
    add_project_root_to_path()
    WRN = __import__("core.models.wideresnetwithswish", fromlist=["WideResNet"])
    model = WRN.WideResNet(depth=28, width=10, num_classes=10, activation_fn=nn.SiLU)
    state = extract_state_dict(load_checkpoint(CHECKPOINTS["WRN"]))
    return nn.Sequential(load_checked(model, state, "WRN", zero_prefix=True)).to(device).eval()


def load_cait(device: torch.device):
    add_project_root_to_path()
    CAIT = __import__("core.models.cait_model", fromlist=["CaiT", "CAIT_CONFIGS"])
    cfg = CAIT.CAIT_CONFIGS["CaiT-XXS-12"]
    backbone = CAIT.CaiT(
        image_size=(32, 32),
        patch_size=cfg["patch_size"],
        num_classes=10,
        dim=cfg["dim"],
        depth=cfg["depth"],
        cls_depth=cfg["cls_depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
        dim_head=cfg["dim_head"],
    )
    model = nn.Sequential(backbone)
    state = strip_common_prefixes(extract_state_dict(load_checkpoint(CHECKPOINTS["CAIT"])))
    state = {(k if k.startswith("0.") else "0." + k): v for k, v in state.items()}
    return load_checked(model, state, "CAIT").to(device).eval()


def load_vit(device: torch.device):
    add_project_root_to_path()
    from core.models.vit_model import create_vit_2
    wrapper = create_vit_2(name="ViT-L_16", num_classes=10, input_size=224)
    state = strip_common_prefixes(extract_state_dict(load_checkpoint(CHECKPOINTS["VIT"])))
    target_keys = set(wrapper.state_dict().keys())
    if state and not any(k in target_keys for k in state):
        candidate = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
        if sum(k in target_keys for k in candidate) > 0:
            state = candidate
    return nn.Sequential(wrapper).to(device).eval(), "vit_wrapper"


class ViTInputWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x)


def load_vit_final(device: torch.device):
    model, _ = load_vit(device)
    return ViTInputWrapper(model[0]).to(device).eval()


def load_mamba(device: torch.device):
    """Load MambaVision-T while handling checkpoint prefixes safely.

    The checkpoint used by this project can contain keys such as:
        0.mean
        0.std
        0.mamba.patch_embed...

    The live create_mamba() wrapper expects:
        mean
        std
        mamba.patch_embed...

    Therefore the leading ``0.`` is removed before loading. We choose the
    candidate key mapping that maximizes overlap with the live model keys,
    and then require all non-buffer parameters/buffers expected by the model
    to be present with no unexpected checkpoint keys.
    """
    add_project_root_to_path()
    from core.models.mamba_model import create_mamba

    wrapper = create_mamba(
        name="mamba_vision_T",
        num_classes=10,
        input_size=224,
        pretrained=False,
    )

    checkpoint = load_checkpoint(CHECKPOINTS["MAMBA"])
    raw_state = extract_state_dict(checkpoint)
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError("Mamba checkpoint state_dict is empty or invalid.")

    raw_state = strip_common_prefixes(raw_state)
    live_keys = set(wrapper.state_dict().keys())

    # Generate a few plausible namespace normalizations.
    candidates = []
    candidates.append(("as-is", dict(raw_state)))

    # Case seen in the checkpoint: 0.mean / 0.std / 0.mamba.*
    no_zero = {
        (k[2:] if k.startswith("0.") else k): v
        for k, v in raw_state.items()
    }
    candidates.append(("remove-leading-0", no_zero))

    # Case where only the Mamba submodule needs the mamba. namespace.
    add_mamba = {
        (k if k.startswith("mamba.") else "mamba." + k): v
        for k, v in no_zero.items()
    }
    candidates.append(("add-mamba-prefix", add_mamba))

    # Prefer the mapping with the greatest exact key overlap.
    best_name, best_state, best_overlap = None, None, -1
    for name, candidate in candidates:
        overlap = sum(k in live_keys for k in candidate.keys())
        if overlap > best_overlap:
            best_name, best_state, best_overlap = name, candidate, overlap

    if best_state is None or best_overlap <= 0:
        raise RuntimeError(
            "Could not map Mamba checkpoint to the live wrapper. "
            f"Live key examples={list(live_keys)[:5]}, "
            f"checkpoint key examples={list(raw_state)[:5]}"
        )

    print(
        f"[MAMBA] Checkpoint mapping: {best_name}; "
        f"matched {best_overlap}/{len(live_keys)} live state_dict keys",
        flush=True,
    )

    # Keep only keys belonging to the live model with matching shapes.
    compatible = {}
    wrong_shapes = []
    for key, value in best_state.items():
        if key not in live_keys:
            continue
        live_value = wrapper.state_dict()[key]
        if tuple(value.shape) != tuple(live_value.shape):
            wrong_shapes.append(
                (key, tuple(value.shape), tuple(live_value.shape))
            )
            continue
        compatible[key] = value

    missing, unexpected = wrapper.load_state_dict(compatible, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    real_unexpected = list(unexpected)

    if real_missing or real_unexpected or wrong_shapes:
        raise RuntimeError(
            "Mamba checkpoint mismatch after namespace mapping. "
            f"Mapping={best_name}; Missing={real_missing[:20]} "
            f"Unexpected={real_unexpected[:20]} "
            f"WrongShapes={wrong_shapes[:10]}"
        )

    return nn.Sequential(wrapper).to(device).eval()


def build_snn():
    add_project_root_to_path()
    from spikingjelly.clock_driven import neuron, surrogate
    from spikingjelly.clock_driven.model import sew_resnet
    return sew_resnet.multi_step_sew_resnet18(
        T=5,
        num_classes=10,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepParametricLIFNode,
        surrogate_function=surrogate.ATan(),
    )


def load_snn(device: torch.device, kind: str):
    add_project_root_to_path()
    from spikingjelly.clock_driven import functional
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
    model._snn_functional_module = functional
    return model.to(device).eval()


def load_models(device: torch.device) -> Dict[str, nn.Module]:
    return {
        "WRN": load_wrn(device),
        "CAIT": load_cait(device),
        "VIT": load_vit_final(device),
        "MAMBA": load_mamba(device),
        "DM_R18": load_snn(device, "DM_R18"),
        "FAT_R18": load_snn(device, "FAT_R18"),
    }


def forward_logits(name: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if name in ("DM_R18", "FAT_R18"):
        functional = getattr(model, "_snn_functional_module")
        functional.reset_net(model)
        out = model(x)
        if out.ndim == 3:
            return out.mean(dim=0)
        return out
    return model(x)


def predict_model(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    preds: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device, non_blocking=True)
        with torch.inference_mode():
            logits = forward_logits(name, model, xb)
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds, dim=0)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    roll_seed = args.seed if args.roll_seed is None else args.roll_seed
    skeleton_root = args.skeleton_root or default_skeleton_root(args.dataset)
    output_root = args.output_root or default_output_root(args.dataset)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available")

    p, probability_mode, game_path = get_probabilities(args)
    p_mapping = {name: float(prob) for name, prob in zip(MODEL_NAMES, p.tolist())}

    print("=" * 80)
    print("REPEATED FIRST WORKING SKELETON KEY PROBABILITY EXPERIMENT")
    print("=" * 80)
    print(f"Dataset                 : {args.dataset}")
    print(f"Skeleton-Key seed       : {args.seed}")
    print(f"Repeat count N/key      : {args.repeat_count}")
    print(f"Total prediction trials : {6 * args.repeat_count}")
    print(f"Probability mode        : {probability_mode}")
    if game_path:
        print(f"GaME results            : {game_path}")
    print(f"Skeleton-Key root       : {skeleton_root}")
    print(f"Output root             : {output_root}")
    print(f"Device                  : {device}")
    print(f"Batch size              : {args.batch_size}")

    print("\nDefender probabilities:")
    for name in MODEL_NAMES:
        print(f"  {DISPLAY_NAMES[name]:<18} {p_mapping[name]:.12f}")

    first_keys, seed_root = load_first_working_keys(args.seed, skeleton_root)

    print("\nFirst working key selected from each target group:")
    for name in MODEL_NAMES:
        r = first_keys[name]
        print(
            f"  {DISPLAY_NAMES[name]:<18} sample_index={r['sample_index']:<6d} "
            f"true_label={r['y']} source={r['source_file']}"
        )

    # One independent weighted-die draw per repetition. The draw sequence is
    # deterministic for a fixed roll seed and independent of model execution.
    rng = np.random.default_rng(roll_seed)
    source_trials: List[Dict[str, Any]] = []
    for source_name in MODEL_NAMES:
        r = first_keys[source_name]
        selected_indices = rng.choice(
            len(MODEL_NAMES), size=args.repeat_count, replace=True, p=p
        )
        for rep in range(args.repeat_count):
            selected_name = MODEL_NAMES[int(selected_indices[rep])]
            source_trials.append({
                "trial_id": len(source_trials),
                "source_target_model": source_name,
                "source_target_display": DISPLAY_NAMES[source_name],
                "repeat_number": rep + 1,
                "sample_index": r["sample_index"],
                "true_label": r["y"],
                "selected_model": selected_name,
                "selected_model_display": DISPLAY_NAMES[selected_name],
                "selected_probability": float(p[int(selected_indices[rep])]),
                "source_file": r["source_file"],
                "source_position": 0,
                "prediction": None,
                "correct": None,
            })

    # Group trial IDs by the selected defense model so each checkpoint is run only
    # on the keys actually assigned to it.
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, trial in enumerate(source_trials):
        grouped[trial["selected_model"]].append(idx)

    print("\nRandom selection counts:")
    for name in MODEL_NAMES:
        observed = len(grouped.get(name, []))
        expected = len(source_trials) * p_mapping[name]
        print(
            f"  {DISPLAY_NAMES[name]:<18} observed={observed:4d} expected={expected:8.2f}"
        )

    models = load_models(device)
    prediction = torch.empty(len(source_trials), dtype=torch.long)

    try:
        print("\nEvaluating selected defenses...")
        for selected_name in MODEL_NAMES:
            trial_ids = grouped.get(selected_name, [])
            if not trial_ids:
                continue

            # Cache each repeated source key once, then batch those repeated trials.
            x = torch.stack([
                first_keys[source_trials[i]["source_target_model"]]["x_adv"]
                for i in trial_ids
            ])

            print(
                f"  {DISPLAY_NAMES[selected_name]:<18}: {len(trial_ids)} trials",
                flush=True,
            )
            pred = predict_model(
                selected_name,
                models[selected_name],
                x,
                device,
                args.batch_size,
            )
            prediction[torch.tensor(trial_ids, dtype=torch.long)] = pred

        y = torch.tensor(
            [trial["true_label"] for trial in source_trials], dtype=torch.long
        )
        correct = prediction.eq(y)

        for i, trial in enumerate(source_trials):
            trial["prediction"] = int(prediction[i].item())
            trial["correct"] = bool(correct[i].item())

        total_trials = len(source_trials)
        correct_count = int(correct.sum().item())
        incorrect_count = total_trials - correct_count
        robustness = correct_count / total_trials
        attack_success = 1.0 - robustness

        # Summary by selected model.
        by_selected_model: Dict[str, Dict[str, Any]] = {}
        for name in MODEL_NAMES:
            ids = grouped.get(name, [])
            if ids:
                m = correct[torch.tensor(ids, dtype=torch.long)]
                c = int(m.sum().item())
                acc = c / len(ids)
            else:
                c = 0
                acc = None
            by_selected_model[name] = {
                "display_name": DISPLAY_NAMES[name],
                "probability": p_mapping[name],
                "selected_trials": len(ids),
                "correct": c,
                "incorrect": (len(ids) - c),
                "accuracy_when_selected": acc,
            }

        # Summary by generating/source target key.
        by_source_target: Dict[str, Dict[str, Any]] = {}
        for source_name in MODEL_NAMES:
            ids = [
                i for i, trial in enumerate(source_trials)
                if trial["source_target_model"] == source_name
            ]
            m = correct[torch.tensor(ids, dtype=torch.long)]
            c = int(m.sum().item())
            by_source_target[source_name] = {
                "display_name": DISPLAY_NAMES[source_name],
                "repetitions": len(ids),
                "true_label": first_keys[source_name]["y"],
                "sample_index": first_keys[source_name]["sample_index"],
                "correct": c,
                "incorrect": len(ids) - c,
                "accuracy": c / len(ids),
                "source_file": first_keys[source_name]["source_file"],
            }

        # Cross-tab: generating target x selected defense.
        source_selected: Dict[str, Dict[str, Dict[str, int]]] = {}
        for source_name in MODEL_NAMES:
            source_selected[source_name] = {}
            for selected_name in MODEL_NAMES:
                ids = [
                    i for i, trial in enumerate(source_trials)
                    if trial["source_target_model"] == source_name
                    and trial["selected_model"] == selected_name
                ]
                c = sum(1 for i in ids if source_trials[i]["correct"])
                source_selected[source_name][selected_name] = {
                    "trials": len(ids),
                    "correct": c,
                    "incorrect": len(ids) - c,
                }

        # Output files.
        run_dir = os.path.join(output_root, f"seed_{args.seed}", f"N_{args.repeat_count}")
        os.makedirs(run_dir, exist_ok=True)

        csv_path = os.path.join(run_dir, "repeated_skeleton_key_trials.csv")
        json_path = os.path.join(run_dir, "repeated_skeleton_key_metrics.json")
        pt_path = os.path.join(run_dir, "repeated_skeleton_key_results.pt")
        log_path = os.path.join(run_dir, "repeated_skeleton_key.log")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_id",
                "dataset",
                "seed",
                "repeat_count",
                "source_target_model",
                "source_target_display",
                "repeat_number",
                "sample_index",
                "true_label",
                "selected_model",
                "selected_model_display",
                "selected_probability",
                "prediction",
                "correct",
                "source_file",
                "source_position",
            ])
            for trial in source_trials:
                writer.writerow([
                    trial["trial_id"],
                    args.dataset,
                    args.seed,
                    args.repeat_count,
                    trial["source_target_model"],
                    trial["source_target_display"],
                    trial["repeat_number"],
                    trial["sample_index"],
                    trial["true_label"],
                    trial["selected_model"],
                    trial["selected_model_display"],
                    trial["selected_probability"],
                    trial["prediction"],
                    trial["correct"],
                    trial["source_file"],
                    trial["source_position"],
                ])

        result_payload = {
            "dataset": args.dataset,
            "seed": args.seed,
            "roll_seed": roll_seed,
            "repeat_count": args.repeat_count,
            "total_trials": total_trials,
            "model_order": MODEL_NAMES,
            "display_names": DISPLAY_NAMES,
            "probability_mode": probability_mode,
            "game_results": game_path,
            "defender_probabilities": p.tolist(),
            "first_working_keys": {
                name: {
                    "sample_index": first_keys[name]["sample_index"],
                    "true_label": first_keys[name]["y"],
                    "source_file": first_keys[name]["source_file"],
                    "source_position": 0,
                    "num_available_working_keys": first_keys[name]["num_available_working_keys"],
                    "original_predictions": first_keys[name]["original_predictions"],
                }
                for name in MODEL_NAMES
            },
            "trial_source_target_model": [t["source_target_model"] for t in source_trials],
            "trial_sample_index": [t["sample_index"] for t in source_trials],
            "trial_true_label": [t["true_label"] for t in source_trials],
            "trial_selected_model": [t["selected_model"] for t in source_trials],
            "trial_prediction": prediction,
            "trial_correct": correct,
            "robustness": robustness,
            "attack_success_rate": attack_success,
            "by_selected_model": by_selected_model,
            "by_source_target": by_source_target,
            "source_target_x_selected_model": source_selected,
        }
        torch.save(result_payload, pt_path)

        metrics = {
            "experiment": {
                "protocol": (
                    "Take the first working Skeleton Key from each of six target groups; "
                    "repeat each key N times; independently select one defense model per "
                    "repetition according to the supplied probability vector."
                ),
                "dataset": args.dataset,
                "seed": args.seed,
                "roll_seed": roll_seed,
                "repeat_count_per_key": args.repeat_count,
                "num_source_keys": 6,
                "total_trials": total_trials,
                "probability_mode": probability_mode,
                "game_results": game_path,
                "device": str(device),
                "batch_size": args.batch_size,
                "skeleton_root": seed_root,
                "started_at": datetime.now().isoformat(),
            },
            "defender_probabilities": {
                name: p_mapping[name] for name in MODEL_NAMES
            },
            "first_working_keys": {
                name: {
                    "sample_index": first_keys[name]["sample_index"],
                    "true_label": first_keys[name]["y"],
                    "source_file": first_keys[name]["source_file"],
                    "source_position": 0,
                    "num_available_working_keys": first_keys[name]["num_available_working_keys"],
                    "original_predictions": first_keys[name]["original_predictions"],
                }
                for name in MODEL_NAMES
            },
            "overall": {
                "total_trials": total_trials,
                "correct": correct_count,
                "incorrect": incorrect_count,
                "robustness": robustness,
                "robustness_percent": 100.0 * robustness,
                "attack_success_rate": attack_success,
                "attack_success_rate_percent": 100.0 * attack_success,
            },
            "selection_counts": {
                name: {
                    "observed": len(grouped.get(name, [])),
                    "expected": total_trials * p_mapping[name],
                }
                for name in MODEL_NAMES
            },
            "by_selected_model": by_selected_model,
            "by_source_target": by_source_target,
            "source_target_x_selected_model": source_selected,
            "files": {
                "csv": csv_path,
                "json": json_path,
                "pt": pt_path,
                "log": log_path,
            },
        }
        metrics["experiment"]["finished_at"] = datetime.now().isoformat()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("REPEATED FIRST WORKING SKELETON KEY PROBABILITY EXPERIMENT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Repeat count per key: {args.repeat_count}\n")
            f.write(f"Total trials: {total_trials}\n")
            f.write(f"Probability mode: {probability_mode}\n")
            f.write("\nDefender probabilities:\n")
            for name in MODEL_NAMES:
                f.write(f"{DISPLAY_NAMES[name]}: {p_mapping[name]:.12f}\n")
            f.write("\nFINAL RESULTS\n")
            f.write(f"Correct: {correct_count}\n")
            f.write(f"Incorrect: {incorrect_count}\n")
            f.write(f"Robustness: {robustness:.8f} ({100.0 * robustness:.4f}%)\n")
            f.write(f"Attack success: {attack_success:.8f} ({100.0 * attack_success:.4f}%)\n")
            f.write("\nSelection counts:\n")
            for name in MODEL_NAMES:
                f.write(f"{DISPLAY_NAMES[name]}: {len(grouped.get(name, []))}\n")

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"First working keys       : 6")
        print(f"Repetitions per key      : {args.repeat_count}")
        print(f"Total prediction trials  : {total_trials}")
        print(f"Correct predictions      : {correct_count}")
        print(f"Incorrect predictions    : {incorrect_count}")
        print(f"Empirical robustness     : {100.0 * robustness:.4f}%")
        print(f"Attack success rate      : {100.0 * attack_success:.4f}%")
        print(f"CSV                      : {csv_path}")
        print(f"JSON metrics             : {json_path}")
        print(f"Tensor results           : {pt_path}")
        print(f"Log                      : {log_path}")

    finally:
        for model in models.values():
            try:
                model.to("cpu")
            except Exception:
                pass
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    run(args)
