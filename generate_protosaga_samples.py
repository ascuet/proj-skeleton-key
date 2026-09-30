import os
import gc
import json
import itertools
import logging
import traceback
import multiprocessing as mp
import argparse
import sys
from datetime import datetime

import torch


def _json_default(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import neuron
from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.model import sew_resnet

import Utilities.DataManagerPytorch as DMP
import AttackWrappersProtoSAGA as ProtoSAGA
import importlib

WRN = importlib.import_module("core.models.wideresnetwithswish")

CAIT = importlib.import_module("core.models.cait_model")

from core.models.vit_model import create_vit_2
from core.models.mamba_model import create_mamba as create_mamba_2


REQUESTED_GPU = os.environ.get("PROTO_SINGLE_GPU")
if torch.cuda.is_available():
    if REQUESTED_GPU is not None:
        torch.cuda.set_device(0)
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


NUM_CLASSES = 10
DATASET = "cifar10"

NUM_SAMPLES = 1000
SAMPLES_PER_CLASS = 100

EPSILON = 0.031
NUM_STEPS = 100
EPS_STEP = EPSILON / NUM_STEPS

ALPHA_LEARNING_RATE = 0.1
FITTING_FACTOR = 10

CLIP_MIN = 0.0
CLIP_MAX = 1.0


CLEAN_BATCH_SIZE = 128


PROTO_BATCH_SIZE = 128


PROTO_BATCH_CANDIDATES = [128, 64, 32, 16, 8, 4]

EVAL_BATCH_SIZE = 64


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS = {
    "WRN": os.path.join(PROJECT_ROOT, "checkpoint", "Wrn_28_10.pt"),
    "CAIT": os.path.join(PROJECT_ROOT, "checkpoint", "Cait_xxs.pt"),
    "VIT": os.path.join(PROJECT_ROOT, "checkpoint", "Vit_L_16.pt"),
    "MAMBA": os.path.join(PROJECT_ROOT, "checkpoint", "MambaVision_T.pt"),
    "DM_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_sew_resnet.pt"),
    "FAT_R18": os.path.join(PROJECT_ROOT, "checkpoint", "snn_resnet18_cifar10_5_219_7315.pth"),
}


OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "generated_adversarial_examples", "protosaga", "random-training", DATASET, "seed_42")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

WORKER_ID = os.environ.get("PROTO_WORKER_ID")
if WORKER_ID is not None:
    METRICS_PATH = os.path.join(OUTPUT_ROOT, f"metrics_worker_{WORKER_ID}.json")
else:
    METRICS_PATH = os.path.join(OUTPUT_ROOT, "metrics.json")

LOG_PATH = os.path.join(OUTPUT_ROOT, "protosaga.log")


logger = logging.getLogger("ProtoSAGA")

logger.setLevel(logging.INFO)

logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.FileHandler(LOG_PATH, mode="a")

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.addHandler(console_handler)


DISPLAY_NAMES = {
    "WRN": "WRN-28-10-SiLU",
    "CAIT": "CaiT-XXS-12",
    "VIT": "ViT-L_16",
    "MAMBA": "MambaVision-T",
    "DM_R18": "DM-R18",
    "FAT_R18": "FAT-R18",
}


MODEL_NAMES = ["WRN", "CAIT", "VIT", "MAMBA", "DM_R18", "FAT_R18"]


metrics = {
    "experiment": {
        "attack": "ProtoSAGA",
        "dataset": "CIFAR-10 training",
        "num_models": 6,
        "num_pairs": 15,
        "epsilon": EPSILON,
        "num_steps": NUM_STEPS,
        "eps_step": EPS_STEP,
        "alpha_learning_rate": ALPHA_LEARNING_RATE,
        "fitting_factor": FITTING_FACTOR,
        "num_samples_per_pair": NUM_SAMPLES,
        "samples_per_class": SAMPLES_PER_CLASS,
        "xstart_seed": int(os.environ.get("PROTO_SEED", "42")),
        "clean_selection": (
            "randomly selected, class-balanced CIFAR-10 training samples; "
            "no correctness filter; same common xStart for all 15 pairs"
        ),
        "clean_batch_size": CLEAN_BATCH_SIZE,
        "proto_batch_size_default": PROTO_BATCH_SIZE,
        "proto_batch_candidates": PROTO_BATCH_CANDIDATES,
        "started_at": datetime.now().isoformat(),
    },
    "pairs": {},
    "errors": [],
}


if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            existing_metrics = json.load(f)

        if isinstance(existing_metrics, dict):
            metrics.update(existing_metrics)

            if "pairs" not in metrics:
                metrics["pairs"] = {}

            if "errors" not in metrics:
                metrics["errors"] = []

    except Exception:
        logger.warning("Could not read existing metrics.json. " "Starting a new metrics structure.")

metrics["errors"] = []
metrics["experiment"]["started_at"] = datetime.now().isoformat()


def save_metrics():
    metrics["experiment"]["updated_at"] = datetime.now().isoformat()

    temp_path = METRICS_PATH + ".tmp"

    with open(temp_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    os.replace(temp_path, METRICS_PATH)


def record_error(pair_name, error):
    error_record = {
        "pair": pair_name,
        "timestamp": datetime.now().isoformat(),
        "error": str(error),
        "traceback": traceback.format_exc(),
    }

    metrics["errors"].append(error_record)

    save_metrics()

    logger.error("PAIR FAILED: %s", pair_name, exc_info=True)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_state_dict", "unaveraged_model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            logger.info("Using checkpoint['%s']", key)

            return checkpoint[key]

    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError("Unable to identify state_dict. " f"Checkpoint keys: {list(checkpoint.keys())}")


def clean_state_dict(state_dict, remove_zero_prefix=False):
    cleaned = {}

    prefixes = ("module.", "_orig_mod.", "model.")

    for key, value in state_dict.items():
        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]

                    changed = True

        if remove_zero_prefix and new_key.startswith("0."):
            new_key = new_key[2:]

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location="cpu", weights_only=False)


class ViTProtoWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()

        self.vit = vit

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        return self.vit(x)

    def forward2(self, x, labels=None):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        if hasattr(self.vit, "forward2"):
            return self.vit.forward2(x, labels=labels)

        return self.vit(x)


def load_wrn():
    path = CHECKPOINTS["WRN"]

    logger.info("Loading WRN: %s", path)

    backbone = WRN.WideResNet(depth=28, width=10, num_classes=NUM_CLASSES, activation_fn=torch.nn.SiLU)

    checkpoint = load_checkpoint(path)

    state_dict = extract_state_dict(checkpoint)

    state_dict = clean_state_dict(state_dict, remove_zero_prefix=True)

    backbone.load_state_dict(state_dict, strict=True)

    backbone.eval()

    logger.info("WRN loaded successfully.")

    return backbone


def load_cait():
    path = CHECKPOINTS["CAIT"]
    logger.info("Loading CaiT: %s", path)

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

    checkpoint = load_checkpoint(path)
    state_dict = extract_state_dict(checkpoint)

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned_state_dict[key] = value

    ckpt_first = next(iter(cleaned_state_dict.keys()))
    model_first = next(iter(model.state_dict().keys()))

    logger.info("CaiT checkpoint first key after cleanup: %s", ckpt_first)

    logger.info("CaiT model first key: %s", model_first)

    if ckpt_first != model_first:
        raise RuntimeError(
            "CaiT key cleanup failed.\n" f"Checkpoint first key: {ckpt_first}\n" f"Model first key: {model_first}"
        )

    model.load_state_dict(cleaned_state_dict, strict=True)

    model.eval()

    logger.info("CaiT loaded successfully with strict=True.")

    return model


def load_vit():
    path = CHECKPOINTS["VIT"]

    logger.info("Loading ViT-L/16: %s", path)

    wrapper = create_vit_2(name="ViT-L_16", num_classes=NUM_CLASSES, input_size=224)

    checkpoint = load_checkpoint(path)

    state_dict = extract_state_dict(checkpoint)

    keys = list(state_dict.keys())

    if not keys:
        raise RuntimeError("ViT checkpoint is empty.")

    first_key = keys[0]

    for prefix in ("module.0.", "module."):
        if first_key.startswith(prefix):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}

            break

    live_keys = list(wrapper.state_dict().keys())

    live_has_vit_prefix = any(k.startswith("vit.") for k in live_keys)

    checkpoint_has_vit_prefix = any(k.startswith("vit.") for k in state_dict.keys())

    if live_has_vit_prefix and not checkpoint_has_vit_prefix:
        state_dict = {"vit." + k: v for k, v in state_dict.items()}

    wrapper.load_state_dict(state_dict, strict=True)

    model = ViTProtoWrapper(wrapper.vit)

    model.eval()

    logger.info("ViT-L/16 loaded successfully.")

    return model


def load_mamba():
    path = CHECKPOINTS["MAMBA"]

    logger.info("Loading MambaVision-T: %s", path)

    wrapper = create_mamba_2(name="mamba_vision_T", num_classes=NUM_CLASSES, input_size=224, pretrained=False)

    checkpoint = load_checkpoint(path)

    state_dict = extract_state_dict(checkpoint)

    keys = list(state_dict.keys())

    if not keys:
        raise RuntimeError("Mamba checkpoint is empty.")

    first_key = keys[0]

    for prefix in ("module.0.", "module."):
        if first_key.startswith(prefix):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}

            break

    live_keys = list(wrapper.state_dict().keys())

    live_has_mamba_prefix = any(k.startswith("mamba.") for k in live_keys)

    checkpoint_has_mamba_prefix = any(k.startswith("mamba.") for k in state_dict.keys())

    if live_has_mamba_prefix and not checkpoint_has_mamba_prefix:
        state_dict = {"mamba." + k: v for k, v in state_dict.items()}

    wrapper.load_state_dict(state_dict, strict=True)

    wrapper.eval()

    logger.info("MambaVision-T loaded successfully.")

    return wrapper


def build_snn():
    return sew_resnet.multi_step_sew_resnet18(
        T=5,
        num_classes=NUM_CLASSES,
        cnf="ADD",
        multi_step_neuron=(neuron.MultiStepParametricLIFNode),
        surrogate_function=surrogate.ATan(),
    )


def load_dm_r18():
    path = CHECKPOINTS["DM_R18"]
    logger.info("Loading DM-R18: %s", path)
    model = build_snn()
    checkpoint = load_checkpoint(path)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("DM-R18 checkpoint does not contain 'model_state_dict'.")
    state_dict = checkpoint["model_state_dict"]
    raw_first_key = next(iter(state_dict.keys()))
    logger.info("DM-R18 checkpoint first raw key: %s", raw_first_key)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.0."):
            key = key[len("module.0.") :]
        elif key.startswith("module."):
            key = key[len("module.") :]
            if key.startswith("0."):
                key = key[2:]
        elif key.startswith("0."):
            key = key[2:]
        cleaned_state_dict[key] = value
    model_state = model.state_dict()
    compatible_state_dict = {}
    wrong_shapes = []
    for key, value in cleaned_state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            wrong_shapes.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state_dict[key] = value
    missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    allowed_missing = [k for k in missing if k.endswith("num_batches_tracked")]
    logger.info("DM-R18 first cleaned key: %s", next(iter(cleaned_state_dict.keys())))
    logger.info("DM-R18 first model key: %s", next(iter(model_state.keys())))
    logger.info("DM-R18 mapped parameters: %d / %d", len(compatible_state_dict), len(model_state))
    logger.info("DM-R18 missing keys: %d", len(missing))
    logger.info("DM-R18 allowed missing num_batches_tracked: %d", len(allowed_missing))
    logger.info("DM-R18 unexpected keys: %d", len(unexpected))
    logger.info("DM-R18 wrong-shape keys: %d", len(wrong_shapes))
    if real_missing:
        raise RuntimeError(f"DM-R18 has real missing parameters: {real_missing}")
    if unexpected:
        raise RuntimeError(f"DM-R18 has unexpected parameters: {unexpected}")
    if wrong_shapes:
        raise RuntimeError(f"DM-R18 has wrong-shaped parameters: {wrong_shapes}")
    model.eval()
    logger.info("DM-R18 loaded successfully.")
    return model


def load_fat_r18():
    path = CHECKPOINTS["FAT_R18"]
    logger.info("Loading FAT-R18: %s", path)
    model = build_snn()
    checkpoint = load_checkpoint(path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info("Using checkpoint['state_dict'].")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info("Using checkpoint['model_state_dict'].")
    else:
        raise RuntimeError("FAT-R18 checkpoint does not contain state_dict or model_state_dict.")
    raw_first_key = next(iter(state_dict.keys()))
    logger.info("FAT-R18 checkpoint first raw key: %s", raw_first_key)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.0."):
            key = key[len("module.0.") :]
        elif key.startswith("module."):
            key = key[len("module.") :]
            if key.startswith("0."):
                key = key[2:]
        elif key.startswith("0."):
            key = key[2:]
        cleaned_state_dict[key] = value
    model_state = model.state_dict()
    compatible_state_dict = {}
    wrong_shapes = []
    for key, value in cleaned_state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            wrong_shapes.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state_dict[key] = value
    missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    allowed_missing = [k for k in missing if k.endswith("num_batches_tracked")]
    logger.info("FAT-R18 first cleaned key: %s", next(iter(cleaned_state_dict.keys())))
    logger.info("FAT-R18 first model key: %s", next(iter(model_state.keys())))
    logger.info("FAT-R18 mapped parameters: %d / %d", len(compatible_state_dict), len(model_state))
    logger.info("FAT-R18 missing keys: %d", len(missing))
    logger.info("FAT-R18 allowed missing num_batches_tracked: %d", len(allowed_missing))
    logger.info("FAT-R18 unexpected keys: %d", len(unexpected))
    logger.info("FAT-R18 wrong-shape keys: %d", len(wrong_shapes))
    if real_missing:
        raise RuntimeError(f"FAT-R18 has real missing parameters: {real_missing}")
    if unexpected:
        raise RuntimeError(f"FAT-R18 has unexpected parameters: {unexpected}")
    if wrong_shapes:
        raise RuntimeError(f"FAT-R18 has wrong-shaped parameters: {wrong_shapes}")
    model.eval()
    logger.info("FAT-R18 loaded successfully.")
    return model


def load_model(name):
    if name == "WRN":
        return load_wrn()

    if name == "CAIT":
        return load_cait()

    if name == "VIT":
        return load_vit()

    if name == "MAMBA":
        return load_mamba()

    if name == "DM_R18":
        return load_dm_r18()

    if name == "FAT_R18":
        return load_fat_r18()

    raise ValueError(f"Unknown model: {name}")


class ProtoModelPlus:
    def __init__(self, name, model, device):
        self.modelName = name
        self.model = model
        self.device = device

        self.imgSizeH = 32
        self.imgSizeW = 32

        self.batchSize = PROTO_BATCH_SIZE

        self.n_classes = NUM_CLASSES

        self.rays = False

    def formatDataLoader(self, dataLoader):
        return dataLoader


def snn_logits(model, x):
    functional.reset_net(model)

    output = model(x)

    if output.ndim == 3:
        output = output.mean(dim=0)

    elif output.ndim != 2:
        raise RuntimeError("Unexpected SNN output shape: " f"{tuple(output.shape)}")

    return output


def forward_logits(model_name, model, x):
    if model_name in ("DM_R18", "FAT_R18"):
        return snn_logits(model, x)
    return model(x)


DEFAULT_APGD_XSTART = os.path.join(
    PROJECT_ROOT, "generated_adversarial_examples", "apgd-ce", "random-training", DATASET, "seed_{seed}", "common_xStart.pt"
)


def resolve_apgd_xstart_path(seed):
    return DEFAULT_APGD_XSTART.format(seed=seed)


def load_common_apgd_xstart(path, seed):
    """Load the exact common xStart produced by generate_apgd_samples.py.

    This function intentionally does NOT regenerate samples. ProtoSAGA must use
    exactly the same clean sample tensor used by the APGD experiment.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "APGD common xStart was not found. " "Run the APGD random-training experiment first.\n" f"Expected: {path}"
        )

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected xStart format: {type(data)}")

    required = {"x_clean", "y", "sample_indices", "class_counts", "seed"}
    missing = required - set(data.keys())
    if missing:
        raise RuntimeError(f"APGD common xStart is missing required fields: {sorted(missing)}")

    file_seed = int(data["seed"])
    if file_seed != int(seed):
        raise RuntimeError(f"APGD xStart seed mismatch: file={file_seed}, requested={seed}")

    x_clean = data["x_clean"].detach().cpu().clone()
    y_clean = data["y"].detach().cpu().clone().long()
    sample_indices = data["sample_indices"].detach().cpu().clone().long()
    class_counts = [int(v) for v in data["class_counts"]]

    if x_clean.ndim != 4 or tuple(x_clean.shape[1:]) != (3, 32, 32):
        raise RuntimeError(f"Unexpected APGD xStart shape: {tuple(x_clean.shape)}")
    if len(x_clean) != NUM_SAMPLES or len(y_clean) != NUM_SAMPLES:
        raise RuntimeError(f"Expected {NUM_SAMPLES} APGD xStart samples, got " f"x={len(x_clean)}, y={len(y_clean)}")
    if len(sample_indices) != NUM_SAMPLES:
        raise RuntimeError("APGD sample_indices length does not equal NUM_SAMPLES")
    if class_counts != [SAMPLES_PER_CLASS] * NUM_CLASSES:
        raise RuntimeError(f"APGD xStart is not class balanced as expected: {class_counts}")

    # Check that labels represented by the stored samples are also balanced.
    observed_counts = torch.bincount(y_clean, minlength=NUM_CLASSES).tolist()
    if observed_counts != class_counts:
        raise RuntimeError(
            f"APGD xStart label counts do not match class_counts: " f"observed={observed_counts}, stored={class_counts}"
        )

    logger.info("Loaded EXACT APGD common xStart: %s", path)
    logger.info("x_clean shape: %s", tuple(x_clean.shape))
    logger.info("class counts: %s", class_counts)
    logger.info("seed: %d", seed)

    return x_clean, y_clean, sample_indices, class_counts


def validate_common_xstart(x_clean, y_clean, sample_indices, class_counts):
    if x_clean.shape != (NUM_SAMPLES, 3, 32, 32):
        raise RuntimeError(f"Unexpected xStart shape: {tuple(x_clean.shape)}")
    if y_clean.shape != (NUM_SAMPLES,):
        raise RuntimeError(f"Unexpected xStart label shape: {tuple(y_clean.shape)}")
    if sample_indices.shape != (NUM_SAMPLES,):
        raise RuntimeError(f"Unexpected xStart index shape: {tuple(sample_indices.shape)}")
    if class_counts != [SAMPLES_PER_CLASS] * NUM_CLASSES:
        raise RuntimeError(f"xStart class balance failure: {class_counts}")


def save_common_pair_copy(pair_dir, x_clean, y_clean, sample_indices, class_counts, seed):
    """Save a pair-local copy for auditability without changing the common source."""
    path = os.path.join(pair_dir, "clean.pt")
    torch.save(
        {
            "x_clean": x_clean.cpu(),
            "y": y_clean.cpu(),
            "sample_indices": sample_indices.cpu(),
            "class_counts": [int(v) for v in class_counts],
            "seed": int(seed),
            "dataset": "CIFAR-10 training",
            "selection": "EXACT SAME RANDOM CLASS-BALANCED SET AS APGD; NO CORRECTNESS FILTER",
            "source": "APGD common_xStart.pt",
            "num_samples": int(len(y_clean)),
            "samples_per_class": int(SAMPLES_PER_CLASS),
        },
        path,
    )
    return path


def configure_experiment(seed):
    """Set seed-specific output paths and initialize metrics/logging."""
    global OUTPUT_ROOT, METRICS_PATH, LOG_PATH, file_handler, console_handler

    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT, "generated_adversarial_examples", "protosaga", "random-training", DATASET, f"seed_{seed}"
    )
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    worker_id = os.environ.get("PROTO_WORKER_ID")
    if worker_id is not None:
        METRICS_PATH = os.path.join(OUTPUT_ROOT, f"metrics_worker_{worker_id}.json")
        log_name = f"protosaga_gpu{os.environ.get('PROTO_GPU', worker_id)}.log"
    else:
        METRICS_PATH = os.path.join(OUTPUT_ROOT, "metrics.json")
        log_name = "protosaga.log"

    LOG_PATH = os.path.join(OUTPUT_ROOT, log_name)

    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)


def initialize_metrics(seed, apgd_xstart_path):
    global metrics
    metrics = {
        "experiment": {
            "attack": "ProtoSAGA",
            "dataset": "CIFAR-10 training",
            "num_models": 6,
            "num_pairs": 15,
            "epsilon": EPSILON,
            "num_steps": NUM_STEPS,
            "eps_step": EPS_STEP,
            "alpha_learning_rate": ALPHA_LEARNING_RATE,
            "fitting_factor": FITTING_FACTOR,
            "num_samples_per_pair": NUM_SAMPLES,
            "samples_per_class": SAMPLES_PER_CLASS,
            "clean_selection": "same exact random class-balanced xStart as APGD random-training; no correctness filter",
            "xstart_source": os.path.abspath(apgd_xstart_path),
            "clean_batch_size": CLEAN_BATCH_SIZE,
            "proto_batch_size_default": PROTO_BATCH_SIZE,
            "proto_batch_candidates": PROTO_BATCH_CANDIDATES,
            "gpu_ids": GPU_IDS,
            "parallel_workers": len(GPU_IDS),
            "seed": int(seed),
            "started_at": datetime.now().isoformat(),
        },
        "pairs": {},
        "errors": [],
    }
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                old = json.load(f)
            if isinstance(old, dict):
                metrics["pairs"] = old.get("pairs", {})
        except Exception:
            pass
    save_metrics()


def evaluate_clean_accuracy(model_name, model, x_clean, y_clean):
    loader = DMP.TensorToDataLoader(x_clean, y_clean, transforms=None, batchSize=EVAL_BATCH_SIZE, randomizer=None)

    model = model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            output = forward_logits(model_name, model, x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total if total else 0.0


def evaluate_robust_accuracy(model_name, model, x_adv, y):
    loader = DMP.TensorToDataLoader(x_adv, y, transforms=None, batchSize=EVAL_BATCH_SIZE, randomizer=None)

    model = model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, target in loader:
            x = x.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            output = forward_logits(model_name, model, x)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total if total else 0.0


def proto_name(model_name):
    # ProtoSAGA uses special names for the two SNN defenses.
    if model_name == "DM_R18":
        return "jelly-DM-R18"
    if model_name == "FAT_R18":
        return "jelly-FAT-R18"
    return DISPLAY_NAMES[model_name]


def run_pair_random(
    pair_index, total_pairs, model_a_name, model_b_name, x_common, y_common, common_indices, common_class_counts, seed
):
    """Run ProtoSAGA on one pair using the exact common APGD xStart."""
    pair_label = f"{model_a_name}__{model_b_name}"
    pair_display = f"{DISPLAY_NAMES[model_a_name]} + {DISPLAY_NAMES[model_b_name]}"
    pair_dir = os.path.join(OUTPUT_ROOT, pair_label)
    os.makedirs(pair_dir, exist_ok=True)

    adv_path = os.path.join(pair_dir, f"{pair_label}_ProtoSAGA_eps_0p031_steps_100_n_1000.pt")
    metrics_path_for_pair = os.path.join(pair_dir, "metrics.json")
    log_path_for_pair = os.path.join(pair_dir, "protosaga.log")

    # Pair-local log handler.
    pair_logger = logging.getLogger(f"ProtoSAGAPair.{pair_label}")
    pair_logger.handlers.clear()
    pair_logger.setLevel(logging.INFO)
    pair_fh = logging.FileHandler(log_path_for_pair, mode="a")
    pair_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    pair_ch = logging.StreamHandler()
    pair_ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    pair_logger.addHandler(pair_fh)
    pair_logger.addHandler(pair_ch)

    pair_logger.info("=" * 80)
    pair_logger.info("PROTO SAGA RANDOM-TRAINING PAIR %d/%d", pair_index, total_pairs)
    pair_logger.info("Pair: %s", pair_display)
    pair_logger.info("Seed: %d", seed)
    pair_logger.info("xStart source: EXACT APGD common xStart")
    pair_logger.info("Samples: %d (100/class)", len(y_common))

    x_clean = x_common.detach().cpu().clone()
    y_clean = y_common.detach().cpu().clone().long()
    common_indices = common_indices.detach().cpu().clone().long()
    class_counts = list(common_class_counts)
    validate_common_xstart(x_clean, y_clean, common_indices, class_counts)

    clean_path = save_common_pair_copy(pair_dir, x_clean, y_clean, common_indices, class_counts, seed)
    pair_logger.info("Clean samples saved: %s", clean_path)

    model_a = None
    model_b = None
    model_a_plus = None
    model_b_plus = None
    proto_loader = None
    adv_loader = None

    try:
        model_a = load_model(model_a_name).to(DEVICE).eval()
        model_b = load_model(model_b_name).to(DEVICE).eval()

        clean_acc_a = evaluate_clean_accuracy(model_a_name, model_a, x_clean, y_clean)
        clean_acc_b = evaluate_clean_accuracy(model_b_name, model_b, x_clean, y_clean)
        pair_logger.info("Clean accuracy %s: %.2f%%", DISPLAY_NAMES[model_a_name], clean_acc_a * 100)
        pair_logger.info("Clean accuracy %s: %.2f%%", DISPLAY_NAMES[model_b_name], clean_acc_b * 100)

        x_adv = None
        y_adv = None
        used_proto_batch_size = None
        last_oom = None

        for attempt_batch_size in PROTO_BATCH_CANDIDATES:
            pair_logger.info("Trying ProtoSAGA batch size=%d", attempt_batch_size)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model_a_plus = ProtoModelPlus(proto_name(model_a_name), model_a, DEVICE)
                model_b_plus = ProtoModelPlus(proto_name(model_b_name), model_b, DEVICE)
                model_a_plus.batchSize = attempt_batch_size
                model_b_plus.batchSize = attempt_batch_size

                # Clone ordinary tensors before entering gradient code.
                x_attack = x_clean.detach().clone()
                y_attack = y_clean.detach().clone().long()
                proto_loader = DMP.TensorToDataLoader(
                    x_attack, y_attack, transforms=None, batchSize=attempt_batch_size, randomizer=None
                )

                with torch.enable_grad(), torch.inference_mode(False):
                    adv_loader = ProtoSAGA.SelfAttentionGradientAttack_EOT(
                        device=DEVICE,
                        epsMax=EPSILON,
                        epsStep=EPS_STEP,
                        numSteps=NUM_STEPS,
                        modelListPlus=[model_a_plus, model_b_plus],
                        dataLoader=proto_loader,
                        clipMin=CLIP_MIN,
                        clipMax=CLIP_MAX,
                        alphaLearningRate=ALPHA_LEARNING_RATE,
                        fittingFactor=FITTING_FACTOR,
                        advLoader=None,
                        numClasses=NUM_CLASSES,
                        decay=0,
                    )

                x_adv, y_adv = DMP.DataLoaderToTensor(adv_loader)
                y_adv = y_adv.long()
                used_proto_batch_size = attempt_batch_size
                pair_logger.info("ProtoSAGA succeeded with batch size=%d", attempt_batch_size)
                break

            except torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                pair_logger.warning("CUDA OOM at batch size=%d; trying smaller.", attempt_batch_size)
                for obj_name in ("proto_loader", "adv_loader", "model_a_plus", "model_b_plus"):
                    obj = locals().get(obj_name)
                    if obj is not None:
                        try:
                            del obj
                        except Exception:
                            pass
                proto_loader = None
                adv_loader = None
                model_a_plus = None
                model_b_plus = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if x_adv is None or y_adv is None:
            raise RuntimeError(f"ProtoSAGA failed at all batch sizes {PROTO_BATCH_CANDIDATES}; last OOM={last_oom}")

        if len(x_adv) != NUM_SAMPLES:
            raise RuntimeError(f"Expected {NUM_SAMPLES} adversarial samples, got {len(x_adv)}")
        if not torch.equal(y_clean, y_adv):
            raise RuntimeError("Adversarial labels do not match common xStart labels")

        min_val = x_adv.min().item()
        max_val = x_adv.max().item()
        if min_val < CLIP_MIN - 1e-6 or max_val > CLIP_MAX + 1e-6:
            raise RuntimeError(f"Adversarial pixel range invalid: min={min_val}, max={max_val}")

        perturbation = (x_adv - x_clean).abs()
        linf_per_sample = perturbation.flatten(1).max(dim=1).values
        max_linf = linf_per_sample.max().item()
        mean_linf = linf_per_sample.mean().item()
        if max_linf > EPSILON + 1e-6:
            raise RuntimeError(f"L_inf violation: {max_linf} > {EPSILON}")

        robust_a = evaluate_robust_accuracy(model_a_name, model_a, x_adv, y_adv)
        robust_b = evaluate_robust_accuracy(model_b_name, model_b, x_adv, y_adv)
        attack_success_a = 1.0 - robust_a
        attack_success_b = 1.0 - robust_b

        payload = {
            "x_clean": x_clean.cpu(),
            "x_adv": x_adv.cpu(),
            "y": y_clean.cpu(),
            "sample_indices": common_indices.cpu(),
            "class_counts": class_counts,
            "seed": int(seed),
            "dataset": "CIFAR-10 training",
            "selection": "same exact random class-balanced xStart as APGD random-training; no correctness filter",
            "xstart_source": resolve_apgd_xstart_path(seed),
            "model_1": model_a_name,
            "model_2": model_b_name,
            "model_1_display": DISPLAY_NAMES[model_a_name],
            "model_2_display": DISPLAY_NAMES[model_b_name],
            "attack": "ProtoSAGA",
            "epsilon": EPSILON,
            "eps_step": EPS_STEP,
            "steps": NUM_STEPS,
            "alpha_learning_rate": ALPHA_LEARNING_RATE,
            "fitting_factor": FITTING_FACTOR,
            "num_samples": NUM_SAMPLES,
            "samples_per_class": SAMPLES_PER_CLASS,
            "proto_batch_size": used_proto_batch_size,
            "clean_accuracy_model_1": clean_acc_a,
            "clean_accuracy_model_2": clean_acc_b,
            "robust_accuracy_model_1": robust_a,
            "robust_accuracy_model_2": robust_b,
            "attack_success_rate_model_1": attack_success_a,
            "attack_success_rate_model_2": attack_success_b,
            "max_linf": max_linf,
            "mean_linf": mean_linf,
            "clean_file": clean_path,
            "generated_at": datetime.now().isoformat(),
        }
        torch.save(payload, adv_path)

        pair_metrics = dict(payload)
        pair_metrics.pop("x_clean", None)
        pair_metrics.pop("x_adv", None)
        pair_metrics["status"] = "completed"
        pair_metrics["adversarial_file"] = adv_path
        with open(metrics_path_for_pair, "w") as f:
            json.dump(pair_metrics, f, indent=2, default=_json_default)

        metrics["pairs"][pair_label] = pair_metrics
        save_metrics()

        pair_logger.info("FINAL RESULTS - %s + %s", DISPLAY_NAMES[model_a_name], DISPLAY_NAMES[model_b_name])
        pair_logger.info("Clean accuracy %s: %.2f%%", DISPLAY_NAMES[model_a_name], clean_acc_a * 100)
        pair_logger.info("Clean accuracy %s: %.2f%%", DISPLAY_NAMES[model_b_name], clean_acc_b * 100)
        pair_logger.info("Robust accuracy %s: %.2f%%", DISPLAY_NAMES[model_a_name], robust_a * 100)
        pair_logger.info("Robust accuracy %s: %.2f%%", DISPLAY_NAMES[model_b_name], robust_b * 100)
        pair_logger.info("Attack success %s: %.2f%%", DISPLAY_NAMES[model_a_name], attack_success_a * 100)
        pair_logger.info("Attack success %s: %.2f%%", DISPLAY_NAMES[model_b_name], attack_success_b * 100)
        pair_logger.info("Max L_inf: %.8f", max_linf)
        pair_logger.info("Mean L_inf: %.8f", mean_linf)
        pair_logger.info("Effective ProtoSAGA batch size: %d", used_proto_batch_size)
        pair_logger.info("Adversarial dataset: %s", adv_path)
        pair_logger.info("PAIR COMPLETED: %s", pair_display)

    finally:
        try:
            pair_fh.close()
        except Exception:
            pass
        try:
            pair_ch.close()
        except Exception:
            pass
        if model_a_plus is not None:
            del model_a_plus
        if model_b_plus is not None:
            del model_b_plus
        if proto_loader is not None:
            del proto_loader
        if adv_loader is not None:
            del adv_loader
        if model_a is not None:
            model_a.to("cpu")
        if model_b is not None:
            model_b.to("cpu")
        del model_a
        del model_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


GPU_IDS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ProtoSAGA for all 15 pairs of six defense models using "
            "the exact common random class-balanced xStart from the APGD experiment. "
            "GPU devices are detected automatically from the SLURM/CUDA environment."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--apgd-xstart",
        type=str,
        default=None,
        help="Path to the exact APGD common_xStart.pt. Default is the seed-specific APGD path.",
    )
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--logical-gpu", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def detect_visible_gpus():
    """Return logical CUDA device IDs made visible to this process by SLURM."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ProtoSAGA experiment.")
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("CUDA is available but no visible GPU devices were detected.")
    return list(range(count))


def merge_worker_metrics(seed, total_pairs, num_workers):
    all_pairs = {}
    all_errors = []
    for worker_id in range(num_workers):
        path = os.path.join(OUTPUT_ROOT, f"metrics_worker_{worker_id}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            all_pairs.update(data.get("pairs", {}))
            all_errors.extend(data.get("errors", []))
        except Exception as exc:
            logger.warning("Could not merge %s: %s", path, exc)

    merged = {
        "experiment": {
            "attack": "ProtoSAGA",
            "dataset": "CIFAR-10 training",
            "num_models": len(MODEL_NAMES),
            "num_pairs": total_pairs,
            "epsilon": EPSILON,
            "num_steps": NUM_STEPS,
            "eps_step": EPS_STEP,
            "alpha_learning_rate": ALPHA_LEARNING_RATE,
            "fitting_factor": FITTING_FACTOR,
            "num_samples_per_pair": NUM_SAMPLES,
            "samples_per_class": SAMPLES_PER_CLASS,
            "clean_selection": "same exact random class-balanced xStart as APGD random-training; no correctness filter",
            "xstart_source": resolve_apgd_xstart_path(seed),
            "gpu_assignment": "SLURM/CUDA visible logical devices",
            "parallel_workers": num_workers,
            "seed": int(seed),
            "started_at": datetime.now().isoformat(),
        },
        "pairs": all_pairs,
        "errors": all_errors,
    }
    merged["experiment"]["completed_pairs"] = sum(
        1
        for d in all_pairs.values()
        if d.get("status") == "completed"
        and d.get("epsilon") == EPSILON
        and d.get("steps") == NUM_STEPS
        and d.get("samples_per_class") == SAMPLES_PER_CLASS
    )
    merged["experiment"]["failed_pairs"] = len({e.get("pair") for e in all_errors if e.get("pair")})
    merged["experiment"]["expected_adversarial_examples"] = total_pairs * NUM_SAMPLES
    merged["experiment"]["completed_adversarial_examples"] = merged["experiment"]["completed_pairs"] * NUM_SAMPLES
    merged["experiment"]["finished_at"] = datetime.now().isoformat()

    final_path = os.path.join(OUTPUT_ROOT, "metrics.json")
    with open(final_path + ".tmp", "w") as f:
        json.dump(merged, f, indent=2, default=_json_default)
    os.replace(final_path + ".tmp", final_path)
    return merged


def run_worker(worker_id, num_workers, logical_gpu, seed, apgd_xstart_path):
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError(f"worker_id must be 0..{num_workers - 1}")
    if logical_gpu < 0 or logical_gpu >= torch.cuda.device_count():
        raise ValueError(
            f"logical_gpu={logical_gpu} is invalid; this process sees " f"{torch.cuda.device_count()} CUDA device(s)."
        )

    torch.cuda.set_device(logical_gpu)
    global DEVICE, METRICS_PATH
    DEVICE = torch.device(f"cuda:{logical_gpu}")
    os.environ["PROTO_GPU"] = str(logical_gpu)
    configure_experiment(seed)

    METRICS_PATH = os.path.join(OUTPUT_ROOT, f"metrics_worker_{worker_id}.json")
    initialize_metrics(seed, apgd_xstart_path)

    pairs = list(itertools.combinations(MODEL_NAMES, 2))
    assigned_pairs = [(idx, pair) for idx, pair in enumerate(pairs, start=1) if (idx - 1) % num_workers == worker_id]

    logger.info("=" * 80)
    logger.info("PROTO SAGA RANDOM-TRAINING WORKER")
    logger.info("Worker %d | logical GPU cuda:%d | %s", worker_id, logical_gpu, torch.cuda.get_device_name(logical_gpu))
    logger.info("Seed: %d", seed)
    logger.info("Assigned pairs: %d/%d", len(assigned_pairs), len(pairs))
    logger.info("Exact APGD xStart: %s", apgd_xstart_path)

    x_common, y_common, common_indices, common_class_counts = load_common_apgd_xstart(apgd_xstart_path, seed)

    for pair_index, (model_a_name, model_b_name) in assigned_pairs:
        pair_label = f"{model_a_name}__{model_b_name}"
        try:
            run_pair_random(
                pair_index,
                len(pairs),
                model_a_name,
                model_b_name,
                x_common,
                y_common,
                common_indices,
                common_class_counts,
                seed,
            )
        except Exception as exc:
            record_error(pair_label, exc)
            logger.error("PAIR FAILED: %s", pair_label, exc_info=True)
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Worker %d finished.", worker_id)


def launch_all_workers(seed, apgd_xstart_path):
    import subprocess

    configure_experiment(seed)

    visible_gpus = detect_visible_gpus()
    num_workers = len(visible_gpus)
    if num_workers > 6:
        # There are only 15 pairs; keep the original maximum of six parallel workers.
        visible_gpus = visible_gpus[:6]
        num_workers = 6

    initialize_metrics(seed, apgd_xstart_path)

    # Validate the source xStart BEFORE starting GPU worker processes.
    x_common, y_common, common_indices, common_class_counts = load_common_apgd_xstart(apgd_xstart_path, seed)
    validate_common_xstart(x_common, y_common, common_indices, common_class_counts)

    source_manifest = os.path.join(OUTPUT_ROOT, "common_xStart_source.json")
    with open(source_manifest, "w") as f:
        json.dump(
            {
                "source_apgd_xstart": os.path.abspath(apgd_xstart_path),
                "seed": int(seed),
                "num_samples": NUM_SAMPLES,
                "samples_per_class": SAMPLES_PER_CLASS,
                "class_counts": common_class_counts,
                "sample_indices": common_indices.tolist(),
                "selection": "random class-balanced CIFAR-10 training; no correctness filter",
                "visible_cuda_devices": visible_gpus,
                "num_workers": num_workers,
            },
            f,
            indent=2,
            default=_json_default,
        )

    logger.info("=" * 80)
    logger.info("PROTO SAGA RANDOM CIFAR-10 TRAINING - SLURM MULTI-GPU EXPERIMENT")
    logger.info("Seed: %d", seed)
    logger.info("Common xStart: %s", apgd_xstart_path)
    logger.info("Samples: %d (100/class)", NUM_SAMPLES)
    logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    logger.info("Visible logical CUDA devices: %s", visible_gpus)
    logger.info("Parallel workers: %d", num_workers)
    logger.info("15 model pairs")
    logger.info("=" * 80)

    script = os.path.abspath(__file__)
    processes = []
    # Do NOT rewrite CUDA_VISIBLE_DEVICES. SLURM owns GPU visibility.
    # Each child receives the same SLURM visibility and selects one logical cuda:N.
    for worker_id, logical_gpu in enumerate(visible_gpus):
        env = os.environ.copy()
        env["PROTO_WORKER_ID"] = str(worker_id)
        env["PROTO_SEED"] = str(seed)
        env["PROTO_GPU"] = str(logical_gpu)
        cmd = [
            sys.executable,
            script,
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(num_workers),
            "--logical-gpu",
            str(logical_gpu),
            "--seed",
            str(seed),
            "--apgd-xstart",
            apgd_xstart_path,
        ]
        logger.info("Launching worker %d on logical cuda:%d", worker_id, logical_gpu)
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in processes]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more ProtoSAGA GPU workers failed: exit codes={exit_codes}")

    total_pairs = len(list(itertools.combinations(MODEL_NAMES, 2)))
    merged = merge_worker_metrics(seed, total_pairs, num_workers)
    logger.info("Completed %d/%d pairs", merged["experiment"]["completed_pairs"], merged["experiment"]["num_pairs"])
    logger.info(
        "Generated %d/%d adversarial examples",
        merged["experiment"]["completed_adversarial_examples"],
        merged["experiment"]["expected_adversarial_examples"],
    )
    logger.info("Merged metrics: %s", os.path.join(OUTPUT_ROOT, "metrics.json"))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ProtoSAGA experiment.")

    seed = int(args.seed)
    apgd_xstart_path = args.apgd_xstart or resolve_apgd_xstart_path(seed)

    if args.worker_id is None:
        launch_all_workers(seed, apgd_xstart_path)
    else:
        if args.num_workers is None or args.logical_gpu is None:
            raise RuntimeError("Internal worker launch is missing --num-workers or --logical-gpu.")
        run_worker(args.worker_id, args.num_workers, args.logical_gpu, seed, apgd_xstart_path)


if __name__ == "__main__":
    main()
