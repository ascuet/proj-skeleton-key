# Adversarial Model Training on CIFAR-10
This process requires the 1M EDM data inside the following directory:
```.bash
edm_data/cifar10/1m.npz
```
The 1M data can be downloaded from this [URL](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz).

After executing the following commands for each model, a new directory ` trained_models/<desc>` will be created inside the project root directory.

## WRN-28-10
To train a WRN-28-10 model via [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10 with the 1M generated data provided by [EDM](https://github.com/NVlabs/edm), run the following command:

```bash
python train-wa.py --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'DESIRED_OUTPUT_DIRECTORY_NAME' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'edm_data/cifar10/1m.npz' \
    --ls 0.1
```

## CaiT-XXS-12
Run the following command to train a CaiT-XXS-12 model via TRADES on CIFAR-10 with the 1M generated data provided by [EDM](https://github.com/NVlabs/edm).
```bash
python -u train_wa_cait.py \
   --data-dir 'dataset-data' \
   --log-dir 'trained_models' \
   --desc 'DESIRED_OUTPUT_DIRECTORY_NAME' \
   --data cifar10s \
   --batch-size 512 \
   --model CaiT-XXS-12 \
   --num-adv-epochs 400 \
   --lr 1e-3 \
   --beta 5.0 \
   --unsup-fraction 0.7 \
   --aux-data-filename 'edm_data/cifar10/1m.npz' \
   --ls 0.1
```

## ViT-L/16
The following command initializes ViT-L/16 with a pretrained clean CIFAR-10 checkpoint and performs adversarial training on CIFAR-10.

```bash
python -u train_wa_vit.py \
    --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'DESIRED_OUTPUT_DIRECTORY_NAME' \
    --data cifar10s \
    --model ViT-L_16 \
    --vit-checkpoint 'trained_models/ViT-L_16,cifar10,run0_15K_checkpoint.bin' \
    --input-size 224 \
    --optimizer adamw \
    --batch-size 64 \
    --batch-size-validation 64 \
    --lr 5e-5 \
    --weight-decay 0.05 \
    --beta 5 \
    --ls 0.1 \
    --num-adv-epochs 400 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'edm_data/cifar10/1m.npz'
```

## MambaVision-T
The following command initializes MambaVision-T with a pretrained clean CIFAR-10 checkpoint and performs adversarial training on CIFAR-10:

```bash
python -u train_wa_mamba.py \
    --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'DESIRED_OUTPUT_DIRECTORY_NAME' \
    --data cifar10s \
    --model mamba_vision_T \
    --mamba-checkpoint trained_models/best_mambavision-T_cifar10.pt \
    --input-size 224 \
    --optimizer adamw \
    --batch-size 64 \
    --batch-size-validation 64  \
    --lr 5e-5 \
    --weight-decay 0.05 \
    --beta 5 \
    --ls 0.1 \
    --num-adv-epochs 400 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'edm_data/cifar10/1m.npz'
```
## SNN-FAT-ResNet-18 and SNN-DM-ResNet-18 Models
We also utilized two spiking neural network (SNN) models taken directly from [attacking-the-spike-mdse](https://github.com/nuoxuxxx/attacking-the-spike-mdse/): SNN-DM-ResNet-18, trained with diffusion-model-generated data, and SNN-FAT-ResNet-18, trained with Friendly Adversarial Training (FAT). They are not trained here; download the checkpoints: [SNN-FAT-ResNet-18](https://drive.google.com/file/d/1w6HXrwzj3xQbS8fdFns7IRDspGmxGhty/view?usp=drive_link), [SNN-DM-ResNet-18](https://drive.google.com/file/d/1eXwT8QYflxltRjiQnQaD6_9Y6mb6SPaf/view?usp=drive_link).
