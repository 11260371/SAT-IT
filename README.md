# SAT-IT

Official PyTorch implementation for adversarial training experiments in our paper. This repository includes our proposed method (**SAT-IT**), standard baselines, ResNet / MobileNetV2 model definitions, and white-box / black-box robustness evaluation scripts.

## Repository layout

```
.
├── model/                  # Network architectures (ResNet18, MobileNetV2, Tiny-ImageNet ResNet, …)
├── source/                 # Training objectives (SAT-IT + baselines)
├── train/                  # Training entry scripts
├── attack/                 # Robustness evaluation (FGSM, PGD, CW, AutoAttack, transfer attacks)
├── checkpoints/            # Place trained weights here (.pt / .pth)
└── logs/                   # Created automatically during training
```

### Method ↔ code mapping

| Method | Loss implementation | Training script | Default checkpoint dir |
|--------|---------------------|-----------------|------------------------|
| **SAT-IT (ours)** | `source/ema_mart_weight_reg3_2.py` | `train/train_resnet_mart_ema_reg3_2.py` | `./checkpoints/sat_it` |
| Standard AT | `attack/AT.py` | `train/train_resnet_AT.py` | `./checkpoints/at_cifar10` |
| FAT | `source/FAT.py` | `train/train_FAT.py` | `./checkpoints/fat_cifar10` |
| MART | `source/mart.py` | `train/train_resnet_mart.py` | `./checkpoints/mart_tinyimagenet` |
| TRADES | `source/trades.py` | `train/train_trades.py` | `./checkpoints/trades_tinyimagenet` |
| LOAT | `source/loat.py` | `train/train_resnet_loat.py` | `./checkpoints/loat_tinyimagenet` |
| LBGAT | `source/lbgat.py` | `train/train_resnet_lbgat.py` | `./checkpoints/lbgat_tinyimagenet` |
| RSLAD | `source/RSLAD.py` | `train/train_resnet_RSLAD.py` | `./checkpoints/rslad_tinyimagenet` |
| Clean (no AT) | — | `train/train_resnet_clean.py` | `./checkpoints/clean_tinyimagenet` |

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Datasets

**CIFAR-10** (used by SAT-IT, AT, FAT) is downloaded automatically to `./data/` on first run.

**Tiny-ImageNet-200** (used by MART, TRADES, LOAT, LBGAT, RSLAD, clean training) must be prepared manually:

1. Download [Tiny-ImageNet-200](https://tiny-imagenet.herokuapp.com/).
2. Extract so that the layout is:

```
data/tiny-imagenet-200/
├── train/
└── val/
```

**CIFAR-100** is used only in evaluation scripts and is downloaded automatically.

## Environment setup

Training and evaluation scripts import modules from `model/`, `source/`, and `attack/`. Set `PYTHONPATH` to the repository root before running.

**Linux / macOS**

```bash
export PYTHONPATH="/path/to/source_SAT-IT:/path/to/source_SAT-IT/source:/path/to/source_SAT-IT/attack"
```

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH = "D:\source_SAT-IT;D:\source_SAT-IT\source;D:\source_SAT-IT\attack"
```

Replace the path with your local clone directory.

## Training

Run scripts from the `train/` directory (or pass the full path). Checkpoints are saved under `--model` (default: `./checkpoints/<method>/`). Training logs are written to `./logs/<method>/`.

### SAT-IT (CIFAR-10 + ResNet18)

```bash
cd train
python train_resnet_mart_ema_reg3_2.py
```

Optional arguments: `--epochs`, `--lr`, `--batch-size`, `--beta`, `--model`, etc.

### Baselines on CIFAR-10

```bash
python train_resnet_AT.py
python train_FAT.py
```

### Baselines on Tiny-ImageNet-200

```bash
python train_resnet_mart.py
python train_trades.py
python train_resnet_loat.py
python train_resnet_lbgat.py
python train_resnet_RSLAD.py
python train_resnet_clean.py
```

### Teacher checkpoints (RSLAD / LBGAT)

Before training RSLAD or LBGAT, place a pretrained teacher weight file at:

```
checkpoints/teacher_mart.pt
```

Replace this path in the training script if your teacher checkpoint is stored elsewhere.

### Resume training

By default, training starts from scratch (`checkpoint_epoch = 0` in each script). To resume, set `checkpoint_epoch` in the corresponding `train/*.py` file to the saved epoch index. The script will load:

- `model-res-epoch{N}.pt`
- `opt-res-checkpoint_epoch{N}.tar`
- (SAT-IT only) `ema{N}.pt`

from the `--model` directory.

## Robustness evaluation

Place your trained weights under `checkpoints/`, then update the `model_path` (or `student_path` / `teacher_path`) variable at the top of each evaluation script.

Run from the `attack/` directory:

```bash
cd attack
python attack_cifar10_resnet18.py
python attack_cifar10_mobilenetv2.py
python attack_cifar100_resnet18.py
python attack_cifar100_mobilenetv2.py
```

Default checkpoint placeholders:

| Script | Default weight path |
|--------|---------------------|
| `attack_cifar10_resnet18.py` | `./checkpoints/cifar10_resnet18.pt` |
| `attack_cifar10_mobilenetv2.py` | `./checkpoints/cifar10_mobilenetv2.pt` |
| `attack_cifar100_resnet18.py` | `./checkpoints/cifar100_resnet18.pt` |
| `attack_cifar100_mobilenetv2.py` | `./checkpoints/cifar100_mobilenetv2.pt` |

White-box attacks: **FGSM**, **PGD-100**, **CW-L∞**, and **AutoAttack** (L∞, ε = 8/255).

### Black-box transfer attacks

```bash
python PGD_blackbox.py
python CW_blackbox.py
```

Set `./checkpoints/student.pt` and `./checkpoints/teacher.pt` to the victim and surrogate model weights respectively.

## Default training settings (common)

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum 0.9, weight decay 3.5e-3) |
| Learning rate | 0.01 with step decay |
| Epochs | 120 |
| PGD training steps | 10 |
| PGD ε | 0.031 (8/255) |
| PGD step size | 0.007 (training) / 0.003–0.007 (eval, script-dependent) |

Refer to each `train/*.py` script for method-specific hyperparameters.

## License

This project is released for academic research use. Please contact the authors for commercial use.
