#!/usr/bin/env python3
"""
Prepare already-generated APGD-CE + ProtoSAGA adversarial datasets for GaME.

The selected defense/model set is supplied through --models. The same model set
also defines the corresponding attacker set:
  N APGD-CE attacks (one per selected model)
  N choose 2 ProtoSAGA attacks (one per selected pair)

This script DOES NOT regenerate adversarial examples.
"""

import argparse
import glob
import json
import os
from itertools import combinations

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS = ["WRN", "CAIT", "VIT", "MAMBA", "DM_R18", "FAT_R18"]
NUM_SAMPLES = 1000
SAMPLES_PER_CLASS = 100
EPSILON = 0.031
NUM_STEPS = 100


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
    p = argparse.ArgumentParser(description="Prepare a selected GaME model set.")
    p.add_argument("--dataset", required=True, choices=["cifar10", "tiny-imagenet"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=parse_models, required=True)
    return p.parse_args()


def configure_paths(dataset, seed, models):
    seed_dir = f"seed_{seed}"
    root = os.path.join(PROJECT_ROOT, "generated_adversarial_examples")
    apgd_root = os.path.join(root, "apgd-ce", "random-training", dataset, seed_dir)
    proto_root = os.path.join(root, "protosaga", "random-training", dataset, seed_dir)

    model_tag = "-".join(m.lower() for m in models)
    suffix = f"mixed-{len(models)}-{model_tag}"
    output_dir = os.path.join(PROJECT_ROOT, "ResultsForGaME", dataset)
    adv_file = os.path.join(output_dir, f"Results-xadv-{dataset}-{suffix}.pt")
    manifest_file = os.path.join(
        output_dir, f"Results-xadv-{dataset}-{suffix}-manifest.json"
    )
    return apgd_root, proto_root, output_dir, adv_file, manifest_file, suffix


def load_pt(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected dictionary in {path}")
    return data


def extract_xy(data, path, dataset):
    x = data.get("x_adv", data.get("x"))
    y = data.get("y")
    if x is None:
        raise RuntimeError(f"No adversarial tensor found in {path}")
    if y is None:
        raise RuntimeError(f"No labels found in {path}")

    x = x.detach().cpu()
    y = y.detach().cpu().long()

    if dataset == "cifar10":
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
            raise RuntimeError(f"Unexpected CIFAR-10 shape in {path}: {tuple(x.shape)}")
        expected_counts = [SAMPLES_PER_CLASS] * 10
    else:
        # The current six-model checkpoint/evaluation stack is still CIFAR-10-specific.
        # Keep validation conservative rather than silently accepting an incompatible tensor.
        expected_counts = None

    if len(x) != NUM_SAMPLES or len(y) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} samples in {path}; got x={len(x)}, y={len(y)}"
        )

    if expected_counts is not None:
        counts = torch.bincount(y, minlength=10).tolist()
        if counts != expected_counts:
            raise RuntimeError(f"Class balance failure in {path}: {counts}")

    return x, y


def validate_linf(data, x_adv, label, path):
    if float(x_adv.min()) < -1e-6 or float(x_adv.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"Pixel range failure for {label}: {path}")
    x_clean = data.get("x_clean")
    if x_clean is None:
        return None
    x_clean = x_clean.detach().cpu()
    if tuple(x_clean.shape) != tuple(x_adv.shape):
        raise RuntimeError(f"x_clean/x_adv shape mismatch: {path}")
    max_linf = (x_adv - x_clean).abs().flatten(1).max(dim=1).values.max().item()
    if max_linf > EPSILON + 1e-5:
        raise RuntimeError(f"L_inf violation for {label}: {max_linf} > {EPSILON}")
    return max_linf


def find_apgd(model_name, root):
    d = os.path.join(root, model_name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"APGD directory not found: {d}")
    files = [p for ext in ("*.pt", "*.pth", "*.bin") for p in glob.glob(os.path.join(d, ext))]
    preferred = [
        p for p in files
        if "clean" not in os.path.basename(p).lower()
        and "eps_0p031" in os.path.basename(p).lower()
        and "steps_100" in os.path.basename(p).lower()
        and "n_1000" in os.path.basename(p).lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(f"Multiple APGD candidates for {model_name}: {preferred}")
    fallback = [p for p in files if os.path.basename(p).lower() != "clean.pt"]
    if len(fallback) == 1:
        return fallback[0]
    if not fallback:
        raise FileNotFoundError(f"No APGD-CE adversarial file in {d}")
    raise RuntimeError(f"Multiple APGD candidates for {model_name}: {fallback}")


def find_proto(a, b, root):
    d = os.path.join(root, f"{a}__{b}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"ProtoSAGA pair directory not found: {d}")
    files = glob.glob(os.path.join(d, "*.pt"))
    preferred = [
        p for p in files
        if "proto" in os.path.basename(p).lower()
        and os.path.basename(p).lower() != "clean.pt"
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(f"Multiple ProtoSAGA candidates in {d}: {preferred}")
    fallback = [p for p in files if os.path.basename(p).lower() != "clean.pt"]
    if len(fallback) != 1:
        raise RuntimeError(f"Expected one ProtoSAGA adversarial file in {d}; got {fallback}")
    return fallback[0]


def add_attack(advdict, manifest, key, data, path, attack_type, dataset, **extra):
    x, y = extract_xy(data, path, dataset)
    max_linf = validate_linf(data, x, key, path)
    record = {"x": x, "y": y, "attack_type": attack_type, "source_file": os.path.abspath(path), **extra}
    if max_linf is not None:
        record["max_linf"] = max_linf
    advdict[key] = record
    manifest["attacks"].append({"key": key, **{k: v for k, v in extra.items()}, "attack_type": attack_type, "source_file": os.path.abspath(path)})
    print(f"[OK] {key}: {path}")


def main():
    args = parse_args()
    models = args.models
    proto_pairs = list(combinations(models, 2))
    attack_count = len(models) + len(proto_pairs)
    apgd_root, proto_root, output_dir, adv_file, manifest_file, suffix = configure_paths(
        args.dataset, args.seed, models
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing {attack_count} attacker strategies for GaME")
    print(f"Dataset        : {args.dataset}")
    print(f"Seed           : {args.seed}")
    print(f"Models         : {models}")
    print(f"APGD root      : {apgd_root}")
    print(f"ProtoSAGA root : {proto_root}")
    print(f"Output         : {adv_file}")

    advdict = {}
    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "models": models,
        "num_attack_strategies": attack_count,
        "num_samples": NUM_SAMPLES,
        "samples_per_class": SAMPLES_PER_CLASS if args.dataset == "cifar10" else None,
        "epsilon": EPSILON,
        "num_steps": NUM_STEPS,
        "attacks": [],
    }

    for model in models:
        path = find_apgd(model, apgd_root)
        add_attack(
            advdict, manifest, f"APGD-CE({model})", load_pt(path), path,
            "APGD-CE", args.dataset, target_model=model,
        )

    for a, b in proto_pairs:
        path = find_proto(a, b, proto_root)
        add_attack(
            advdict, manifest, f"ProtoSAGA({a}+{b})", load_pt(path), path,
            "ProtoSAGA", args.dataset, target_models=[a, b],
        )

    if len(advdict) != attack_count:
        raise RuntimeError(f"Expected {attack_count} attacks, collected {len(advdict)}")

    torch.save(advdict, adv_file)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Attack strategies: {len(advdict)}")
    print(f"Saved advdict    : {adv_file}")
    print(f"Saved manifest   : {manifest_file}")


if __name__ == "__main__":
    main()
