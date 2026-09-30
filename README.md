# Skeleton Keys: Estimating Randomized Model Selection Distributions With Limited Queries

## Installation
The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

**Requirements**
- Linux with an NVIDIA GPU (the pinned PyTorch build targets CUDA 11.1)
- Python 3.9

**1. Install uv** (skip if already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Create the environment and install dependencies:**
```bash
uv sync --python 3.9
```
This creates a `.venv/` in the project root and installs everything listed in [pyproject.toml](pyproject.toml), including PyTorch 1.8.0 / torchvision 0.9.0 (CUDA 11.1) from the PyTorch wheel index, [AutoAttack](https://github.com/fra31/auto-attack), and [RandAugment](https://github.com/ildoonet/pytorch-randaugment).

**3. MambaVision support.** MambaVision-T requires `mamba-ssm` and `mambavision`, which must be built against a CUDA toolkit. This is needed for training MambaVision-T and for every later step, since they all load it:
```bash
bash scripts/install_mamba.sh
```
The script loads CUDA 12.3 via `module load`, which is specific to HPC clusters using environment modules; on other systems, remove that line and make sure `nvcc` is on your `PATH`.

**Running scripts.** Either activate the environment with `source .venv/bin/activate`, or prefix commands with `uv run`, e.g.:
```bash
uv run python train-wa.py --help
```

## Acknowledgement
The adversarial training code is adapted from [DM-Improves-AT](https://github.com/wzekai/DM-Improves-AT). 

Two SNN (Spiking Neural Network) adversarial models are taken from [attacking-the-spike-mdse](https://github.com/nuoxuxxx/attacking-the-spike-mdse/). 

We also adapt Python scripts from [Game-Theoretic Mixed Experts (GaME)](https://github.com/EthanRath/Game-Theoretic-Mixed-Experts) to build the payoff matrix and compute the defender's mixed-strategy probabilities.

## Adversarial Model Training
For adversarial model training, please refer to [TRAIN.md](TRAIN.md).

Once training is complete, create a `checkpoint/` directory in the project root. For each trained model, copy `trained_models/<desc>/weights-best.pt` into it and rename it as follows: `Wrn_28_10.pt`, `Cait_xxs.pt`, `Vit_L_16.pt`, and `MambaVision_T.pt`.

The two SNN models are not trained here. Download their checkpoints using the links in [TRAIN.md](TRAIN.md) and place them in `checkpoint/` as `snn_sew_resnet.pt` (SNN-DM-ResNet-18) and `snn_resnet18_cifar10_5_219_7315.pth` (SNN-FAT-ResNet-18).

## Generate Adversarial Samples
All six checkpoints must be in `checkpoint/` before running these steps. Outputs are written to `generated_adversarial_examples/`.

### 1. APGD-CE
Selects 1,000 random class-balanced CIFAR-10 training samples (100 per class) and attacks each of the six models with APGD-CE. 
```bash
uv run python generate_apgd_samples.py --seed 42
```
The selected clean samples are saved to `generated_adversarial_examples/apgd-ce/random-training/cifar10/seed_42/common_xStart.pt`.

### 2. ProtoSAGA
Attacks all 15 model pairs with ProtoSAGA, reusing the exact samples selected in step 1. 
```bash
uv run python generate_protosaga_samples.py --seed 42
```

## Game-Theoretic Mixed Experts (GaME)

### 1. Collect the adversarial datasets
Combines the APGD-CE and ProtoSAGA samples for the selected models into a single file under `ResultsForGaME/cifar10/`:
```bash
uv run python automate_mixed_models.py --dataset cifar10 --seed 42 --models WRN,CAIT,VIT,MAMBA,DM_R18,FAT_R18
```

### 2. Compute the mixed strategies
Evaluates every selected model on every attack to build the payoff matrix, then solves for the defender's and attacker's mixed strategies:
```bash
uv run python game_mixed_models.py --dataset cifar10 --seed 42 --models WRN,CAIT,VIT,MAMBA,DM_R18,FAT_R18
```
`--models` accepts any subset of two or more of `WRN, CAIT, VIT, MAMBA, DM_R18, FAT_R18`; use the same list in both steps. Results are saved to `ResultsForGaME/cifar10/GaME-results-mixed-<N>-<models>.json`, e.g. `GaME-results-mixed-6-wrn-cait-vit-mamba-dm_r18-fat_r18.json`.

## Skeleton Key Attack

A **Skeleton Key** is an adversarial example that one chosen *target* model classifies correctly while every other defense model misclassifies it. In a randomized ensemble such as GaME, only the target model can then defend against the key. This shows how much the ensemble's robustness depends on which model the defender happens to pick.

The code runs in two steps. The first step generates keys for each target model. The second step tests them against a randomly selected defense.

### Step 1: Generate Skeleton Keys (`skeleton_key.py`)

All six models are loaded, and one of them is the target: `0`=ViT-L/16, `1`=CaiT-XXS-12, `2`=WRN-28-10-SiLU, `3`=MambaVision-T, `4`=DM-R18, `5`=FAT-R18 (set with `--target`).

The script draws a class-balanced random set of CIFAR-10 test images (`--num-per-class` per class) once per seed and saves it as `common_xStart.pt`. All six target runs start from these same images, so the results can be compared across targets. 

```bash
uv run python skeleton_key.py --target 2 --num-per-class 200 --seed 42 \
    --epsilon 0.031 --steps 100 --batch-size 32 \
    --output-root generated_adversarial_examples/skeleton_key/cifar10/42/0_031
```

Outputs are saved to `<output-root>/seed_<seed>/<target>/`:

| File | Contents |
|---|---|
| `working_keys.pt` / `non_working_keys.pt` | Clean images, adversarial images, labels and predictions from all six models |
| `all_skeleton_keys.pt` | Every generated sample, whether it worked or not |
| `results.csv`, `predictions.json` | Clean and adversarial prediction from each model for every sample |
| `metrics.json` | Yield, clean accuracies, max/mean L∞, attack settings |

### Step 2: Test against a randomized defense (`evaluate_skeleton_keys.py`)

This script loads the **first working key** for each of the six targets and repeats each key `--repeat-count` times. It generates no new keys. For every repetition it picks one defense model at random, using a chosen probability vector, and records whether that model classifies the key correctly. This gives 6 × N trials in total. Only the target model can classify a key correctly, so the attack success rate shows how often the randomized defense picks the wrong model.

`--probability-mode` sets the defense probabilities:
- `game`: reads P_D from a GaME results JSON (`--game-results GaME-results-*.json`)
- `equal`: 1/6 for each model
- `custom`: six values passed with `--probabilities`, in the order `WRN,CAIT,VIT,MAMBA,DM_R18,FAT_R18`

```bash
uv run python evaluate_skeleton_keys.py --dataset cifar10 --seed 42 \
    --repeat-count 400 --roll-seed 42 \
    --probability-mode game --game-results ResultsForGaME/cifar10/GaME-results-mixed-6-wrn-cait-vit-mamba-dm_r18-fat_r18.json \
    --skeleton-root generated_adversarial_examples/skeleton_key/cifar10/42/0_031 \
    --device cuda:0 --batch-size 32
```

`--skeleton-root` must be the `--output-root` used in Step 1; it contains `seed_<seed>/<target>/working_keys.pt`. If both are omitted, each script defaults to `generated_adversarial_examples/skeleton_key/<dataset>`. The results are saved to `ResultsForGaME/<dataset>/skeleton_key_repeated_probability/seed_<seed>/N_<N>/`. They include per-trial CSV rows, a metrics JSON with overall robustness and attack success rate, and breakdowns by selected model and by source target.


