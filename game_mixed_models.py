#!/usr/bin/env python3
"""
GaME for a user-selected set of defense models.

Protocol:
- Use ALL 1,000 training-derived adversarial examples for every attack.
- No 800/200 split.
- For N selected defenses, use N APGD-CE attacks + C(N,2) ProtoSAGA attacks.
- Build an N_attack x N_defense accuracy payoff matrix A.
- Solve the defender LP for P_D = p*.
- Solve the dual attacker LP on 1-A for P_A = a*.
- Evaluate all selected defenders sequentially on a single visible GPU (or CPU).

The ordering supplied to --models is preserved in the defender probability vector.
"""

import argparse
import gc
import importlib
import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from spikingjelly.clock_driven import functional, neuron, surrogate
from spikingjelly.clock_driven.model import sew_resnet

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS = ["WRN", "CAIT", "VIT", "MAMBA", "DM_R18", "FAT_R18"]
NUM_CLASSES = 10
NUM_SAMPLES_PER_ATTACK = 1000
EVAL_BATCH_SIZE = 32

DISPLAY_NAMES = {
    "WRN": "WRN-28-10-SiLU",
    "CAIT": "CaiT-XXS-12",
    "VIT": "ViT-L_16",
    "MAMBA": "MambaVision-T",
    "DM_R18": "DM-R18",
    "FAT_R18": "FAT-R18",
}

CHECKPOINTS = {
    "WRN": os.path.join(PROJECT_ROOT, "checkpoint", "Wrn_28_10.pt"),
    "CAIT": os.path.join(PROJECT_ROOT, "checkpoint", "Cait_xxs.pt"),
    "VIT": os.path.join(PROJECT_ROOT, "checkpoint", "Vit_L_16.pt"),
    "MAMBA": os.path.join(PROJECT_ROOT, "checkpoint", "MambaVision_T.pt"),
    "DM_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_sew_resnet.pt"),
    "FAT_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_resnet18_cifar10_5_219_7315.pth"),
}

WRN = importlib.import_module("core.models.wideresnetwithswish")
CAIT = importlib.import_module("core.models.cait_model")
from core.models.vit_model import create_vit_2
from core.models.mamba_model import create_mamba as create_mamba_2

DEVICE = torch.device("cpu")


def parse_models(text):
    models = [p.strip().upper().replace("-", "_") for p in text.split(",") if p.strip()]
    if len(models) < 2:
        raise argparse.ArgumentTypeError("--models must contain at least two models")
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown model(s): {unknown}. Allowed: {','.join(ALL_MODELS)}"
        )
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("--models cannot contain duplicates")
    return models


def parse_args():
    p = argparse.ArgumentParser(description="GaME for a selected defense-model set.")
    p.add_argument("--dataset", required=True, choices=["cifar10", "tiny-imagenet"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=parse_models, required=True)
    p.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    return p.parse_args()


def configure_paths(dataset, models):
    model_tag = "-".join(m.lower() for m in models)
    suffix = f"mixed-{len(models)}-{model_tag}"
    output_dir = os.path.join(PROJECT_ROOT, "ResultsForGaME", dataset)
    adv_file = os.path.join(output_dir, f"Results-xadv-{dataset}-{suffix}.pt")
    manifest_file = os.path.join(output_dir, f"Results-xadv-{dataset}-{suffix}-manifest.json")
    results_file = os.path.join(output_dir, f"GaME-results-{suffix}.json")
    log_file = os.path.join(output_dir, f"game_{suffix}.log")
    return output_dir, adv_file, manifest_file, results_file, log_file


def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(data):
    if not isinstance(data, dict):
        return data
    for key in ("model_state_dict", "unaveraged_model_state_dict", "state_dict", "model"):
        if key in data:
            print(f"  Using checkpoint['{key}']", flush=True)
            return data[key]
    return data


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


def load_checked(model, state, name, zero_prefix=False):
    state = strip_common_prefixes(state)
    if zero_prefix:
        state = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint mismatch. Missing={real_missing[:20]} Unexpected={list(unexpected)[:20]}"
        )
    return model


def load_wrn():
    model = WRN.WideResNet(depth=28, width=10, num_classes=NUM_CLASSES, activation_fn=torch.nn.SiLU)
    state = extract_state_dict(load_checkpoint(CHECKPOINTS["WRN"]))
    return load_checked(model, state, "WRN", zero_prefix=True).to(DEVICE).eval()


def load_cait():
    cfg = CAIT.CAIT_CONFIGS["CaiT-XXS-12"]
    model = nn.Sequential(CAIT.CaiT(
        image_size=(32, 32), patch_size=cfg["patch_size"], num_classes=NUM_CLASSES,
        dim=cfg["dim"], depth=cfg["depth"], cls_depth=cfg["cls_depth"],
        heads=cfg["heads"], mlp_dim=cfg["mlp_dim"], dim_head=cfg["dim_head"],
    ))
    state = strip_common_prefixes(extract_state_dict(load_checkpoint(CHECKPOINTS["CAIT"])))
    state = {(k if k.startswith("0.") else "0." + k): v for k, v in state.items()}
    return load_checked(model, state, "CAIT").to(DEVICE).eval()


class ViTWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x)


def load_vit():
    base = create_vit_2(name="ViT-L_16", num_classes=NUM_CLASSES, input_size=224)
    state = strip_common_prefixes(extract_state_dict(load_checkpoint(CHECKPOINTS["VIT"])))
    target_keys = set(base.state_dict().keys())
    candidate = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
    if sum(k in target_keys for k in candidate) > sum(k in target_keys for k in state):
        state = candidate
    return ViTWrapper(load_checked(base, state, "VIT")).to(DEVICE).eval()


def load_mamba():
    model = create_mamba_2(name="mamba_vision_T", num_classes=NUM_CLASSES, input_size=224, pretrained=False)
    state = strip_common_prefixes(extract_state_dict(load_checkpoint(CHECKPOINTS["MAMBA"])))
    target_keys = set(model.state_dict().keys())
    candidate = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
    if sum(k in target_keys for k in candidate) > sum(k in target_keys for k in state):
        state = candidate
    return load_checked(model, state, "MAMBA").to(DEVICE).eval()


def build_snn():
    return sew_resnet.multi_step_sew_resnet18(
        T=5, num_classes=NUM_CLASSES, cnf="ADD",
        multi_step_neuron=neuron.MultiStepParametricLIFNode,
        surrogate_function=surrogate.ATan(),
    )


def load_snn(name):
    model = build_snn()
    state = extract_state_dict(load_checkpoint(CHECKPOINTS[name]))
    state = strip_common_prefixes(state)
    state = {(k[2:] if k.startswith("0.") else k): v for k, v in state.items()}
    compatible = {
        k: v for k, v in state.items()
        if k in model.state_dict() and tuple(v.shape) == tuple(model.state_dict()[k].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint mismatch. Missing={real_missing[:20]} Unexpected={list(unexpected)[:20]}"
        )
    return model.to(DEVICE).eval()


def load_model(name):
    if name == "WRN": return load_wrn()
    if name == "CAIT": return load_cait()
    if name == "VIT": return load_vit()
    if name == "MAMBA": return load_mamba()
    if name in ("DM_R18", "FAT_R18"): return load_snn(name)
    raise ValueError(name)


def forward_logits(name, model, x):
    if name in ("DM_R18", "FAT_R18"):
        functional.reset_net(model)
        out = model(x)
        return out.mean(dim=0) if out.ndim == 3 else out
    return model(x)


def accuracy(name, model, x, y):
    correct = 0
    total = len(y)
    for start in range(0, total, EVAL_BATCH_SIZE):
        xb = x[start:start + EVAL_BATCH_SIZE].to(DEVICE, non_blocking=True)
        yb = y[start:start + EVAL_BATCH_SIZE].to(DEVICE, non_blocking=True)
        with torch.inference_mode():
            pred = forward_logits(name, model, xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / total


def solve_defender_lp(A):
    n_attacks, n_defenses = A.shape
    c = np.zeros(n_defenses + 1)
    c[-1] = -1.0
    A_ub = np.zeros((n_attacks, n_defenses + 1))
    A_ub[:, :n_defenses] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_attacks)
    A_eq = np.zeros((1, n_defenses + 1))
    A_eq[0, :n_defenses] = 1.0
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n_defenses + [(None, None)], method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Defender LP failed: {result.message}")
    return result.x[:n_defenses], float(result.x[-1])


def solve_attacker_dual(A):
    B = 1.0 - A
    n_attacks, n_defenses = B.shape
    c = np.zeros(n_attacks + 1)
    c[-1] = -1.0
    A_ub = np.zeros((n_defenses, n_attacks + 1))
    A_ub[:, :n_attacks] = -B.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_defenses)
    A_eq = np.zeros((1, n_attacks + 1))
    A_eq[0, :n_attacks] = 1.0
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n_attacks + [(None, None)], method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Attacker dual LP failed: {result.message}")
    return result.x[:n_attacks], float(result.x[-1])


def evaluate_all_models_single_gpu(attack_keys, adv, models):
    global DEVICE
    print(f"[SINGLE DEVICE] Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"[SINGLE DEVICE] {torch.cuda.get_device_name(0)}", flush=True)

    result_cols = {}
    for name in models:
        print(f"[DEVICE {DEVICE}] Loading {DISPLAY_NAMES[name]}", flush=True)
        model = load_model(name)
        col = np.zeros(len(attack_keys), dtype=np.float64)
        try:
            for i, key in enumerate(attack_keys):
                x = adv[key]["x"].float()
                y = adv[key]["y"].long()
                col[i] = accuracy(name, model, x, y)
                print(
                    f"[DEVICE {DEVICE}] {name} {i+1:02d}/{len(attack_keys)} "
                    f"{key:<34s} Acc={col[i]:.4f}", flush=True
                )
            result_cols[name] = col
        finally:
            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return result_cols


def main():
    args = parse_args()
    global DEVICE

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is visible.")
        DEVICE = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        DEVICE = torch.device(args.device)

    models = args.models
    proto_pairs = list(combinations(models, 2))
    expected_attack_count = len(models) + len(proto_pairs)

    output_dir, adv_file, manifest_file, results_file, log_file = configure_paths(args.dataset, models)
    os.makedirs(output_dir, exist_ok=True)

    original_out, original_err = sys.stdout, sys.stderr
    log_handle = open(log_file, "a", buffering=1, encoding="utf-8")

    class Tee:
        def __init__(self, terminal, handle):
            self.terminal = terminal
            self.handle = handle
        def write(self, data):
            self.terminal.write(data)
            self.terminal.flush()
            self.handle.write(data)
            self.handle.flush()
        def flush(self):
            self.terminal.flush()
            self.handle.flush()
        def isatty(self):
            return self.terminal.isatty()

    sys.stdout = Tee(original_out, log_handle)
    sys.stderr = Tee(original_err, log_handle)

    try:
        start = __import__("datetime").datetime.now().isoformat()
        print("\n" + "=" * 78)
        print(f"GaME {len(models)}-MODEL / {expected_attack_count}-ATTACK")
        print("=" * 78)
        print(f"Dataset: {args.dataset}")
        print(f"Seed: {args.seed}")
        print(f"Models: {models}")
        print("PAYOFF PROTOCOL: all 1000 samples per attack; no 800/200 split.")
        print(f"Standardized adversarial dataset: {adv_file}")

        if not os.path.isfile(adv_file):
            raise FileNotFoundError(f"Missing {adv_file}. Run automate_mixed_models.py first.")

        if os.path.isfile(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if int(manifest.get("seed", args.seed)) != int(args.seed):
                raise RuntimeError("Manifest seed does not match --seed")
            if manifest.get("models") != models:
                raise RuntimeError(
                    f"Manifest models {manifest.get('models')} do not match --models {models}"
                )

        adv = torch.load(adv_file, map_location="cpu", weights_only=False)
        if len(adv) != expected_attack_count:
            raise RuntimeError(
                f"Expected {expected_attack_count} attacks, found {len(adv)}"
            )
        attack_keys = list(adv.keys())

        for key in attack_keys:
            x, y = adv[key]["x"], adv[key]["y"]
            if len(x) != NUM_SAMPLES_PER_ATTACK or len(y) != NUM_SAMPLES_PER_ATTACK:
                raise RuntimeError(f"{key}: expected 1000 samples")
            if args.dataset == "cifar10":
                counts = torch.bincount(y.long(), minlength=NUM_CLASSES).tolist()
                if counts != [100] * NUM_CLASSES:
                    raise RuntimeError(f"{key}: expected 100/class; got {counts}")

        result_cols = evaluate_all_models_single_gpu(attack_keys, adv, models)

        A = np.zeros((expected_attack_count, len(models)), dtype=np.float64)
        for j, name in enumerate(models):
            if name not in result_cols:
                raise RuntimeError(f"No payoff column returned for {name}")
            A[:, j] = result_cols[name]

        print("\n" + "=" * 78)
        print(f"{expected_attack_count} x {len(models)} PAYOFF MATRIX")
        print("=" * 78)
        for i, key in enumerate(attack_keys):
            print(f"{i+1:02d} {key:<34s} " + " ".join(f"{A[i,j]:.4f}" for j in range(len(models))))

        p_star, defender_value = solve_defender_lp(A)
        a_star, attacker_value = solve_attacker_dual(A)
        consistency = abs(attacker_value - (1.0 - defender_value))
        defender_expected = A @ p_star
        attacker_expected = (1.0 - A).T @ a_star

        print("\n" + "=" * 78)
        print("DEFENDER MIXED STRATEGY P_D = p*")
        print("=" * 78)
        for i, name in enumerate(models):
            print(f"p{i+1} ({DISPLAY_NAMES[name]}): {p_star[i]:.8f}")
        print(f"sum(P_D): {p_star.sum():.8f}")
        print(f"Defender game value: {defender_value:.8f}")

        print("\n" + "=" * 78)
        print("ATTACKER MIXED STRATEGY P_A = a*")
        print("=" * 78)
        for i, key in enumerate(attack_keys):
            print(f"a{i+1:02d} ({key}): {a_star[i]:.8f}")
        print(f"sum(P_A): {a_star.sum():.8f}")
        print(f"Attacker game value: {attacker_value:.8f}")
        print(f"1 - defender value: {1.0 - defender_value:.8f}")
        print(f"Dual consistency error: {consistency:.3e}")

        if consistency > 1e-6:
            raise RuntimeError("Attacker dual is inconsistent with defender game value")

        results = {
            "experiment": {
                "dataset": args.dataset,
                "seed": args.seed,
                "defense_models": models,
                "num_defense_models": len(models),
                "num_attack_strategies": expected_attack_count,
                "apgd_ce_attacks": len(models),
                "protosaga_attacks": len(proto_pairs),
                "samples_per_attack": NUM_SAMPLES_PER_ATTACK,
                "sampling_protocol": "All 1000 training-derived class-balanced adversarial examples per attack; no 800/200 split.",
                "execution": "single-GPU sequential defender evaluation" if DEVICE.type == "cuda" else "CPU sequential defender evaluation",
                "log_file": log_file,
                "started_at": start,
                "finished_at": __import__("datetime").datetime.now().isoformat(),
            },
            "defenses": [
                {"index": i + 1, "name": name, "display_name": DISPLAY_NAMES[name], "probability": float(p_star[i])}
                for i, name in enumerate(models)
            ],
            "attacks": [
                {"index": i + 1, "key": key, "probability": float(a_star[i])}
                for i, key in enumerate(attack_keys)
            ],
            "defender_strategy_P_D": p_star.tolist(),
            "attacker_strategy_P_A": a_star.tolist(),
            "defender_game_value": float(defender_value),
            "attacker_game_value": float(attacker_value),
            "one_minus_defender_value": float(1.0 - defender_value),
            "dual_consistency_error": float(consistency),
            "payoff_matrix_accuracy": A.tolist(),
            "defender_expected_payoff_per_attack": defender_expected.tolist(),
            "attacker_expected_success_per_defense": attacker_expected.tolist(),
            "binding_attacks_for_defender": [
                attack_keys[i] for i, v in enumerate(defender_expected)
                if abs(v - defender_value) < 1e-7
            ],
            "binding_defenses_for_attacker": [
                DISPLAY_NAMES[models[j]] for j, v in enumerate(attacker_expected)
                if abs(v - attacker_value) < 1e-7
            ],
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results: {results_file}")
        print(f"Saved log: {log_file}")
    finally:
        sys.stdout = original_out
        sys.stderr = original_err
        log_handle.close()


if __name__ == "__main__":
    main()
