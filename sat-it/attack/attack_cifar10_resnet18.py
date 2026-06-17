from __future__ import print_function
import os
import math
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from model.mobilenetv2 import *
from model.resnet import *
from model.resnet_tinyimagenet import *
from Attack import _fgsm_whitebox, _pgd100_whitebox, eval_adv_test_whitebox_with_attack
from autoattack import AutoAttack

import torch.nn.functional as F


def attack_cw_inf(model, X, y, epsilon=8/255, step_size=2/255, num_steps=30,
                  confidence=50, num_classes=10):
    device = X.device
    X_pgd = X.detach().clone().requires_grad_(True)

    for _ in range(num_steps):
        with torch.enable_grad():
            output = model(X_pgd)

            target_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)
            real = torch.sum(target_onehot * output, dim=1)
            other = torch.max((1 - target_onehot) * output - target_onehot * 10000, dim=1)[0]
            loss = -torch.clamp(real - other + confidence, min=0.).mean()

        grad = torch.autograd.grad(loss, X_pgd)[0]
        X_pgd = X_pgd.detach() + step_size * grad.sign()

        eta = torch.clamp(X_pgd - X, -epsilon, epsilon)
        X_pgd = torch.clamp(X + eta, 0, 1.0).detach().requires_grad_(True)

    err = (model(X).max(1)[1] != y).float().sum()
    err_pgd = (model(X_pgd).max(1)[1] != y).float().sum()

    return err, err_pgd


def ResNet18_CIFAR10():
    try:
        model = torchvision.models.resnet18(weights=None)
    except TypeError:
        model = torchvision.models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def load_checkpoint(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    new_ckpt = {}
    for k, v in ckpt.items():
        new_k = k.replace("module.", "")
        new_ckpt[new_k] = v

    model.load_state_dict(new_ckpt)
    return model


def eval_autoattack(model, device, test_loader, epsilon):
    model.eval()
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)

    x_all = []
    y_all = []

    for data, target in test_loader:
        x_all.append(data)
        y_all.append(target)

    x_all = torch.cat(x_all, 0).to(device)
    y_all = torch.cat(y_all, 0).to(device)

    x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=100)

    with torch.no_grad():
        outputs_nat = model(x_all)
        _, pred_nat = outputs_nat.max(1)
        nat_acc = (pred_nat == y_all).float().mean().item()

        outputs_adv = model(x_adv)
        _, pred_adv = outputs_adv.max(1)
        rob_acc = (pred_adv == y_all).float().mean().item()

    return nat_acc, rob_acc


def main():
    # Replace with your trained checkpoint (.pt/.pth) before running evaluation.
    model_path = "./checkpoints/cifar10_resnet18.pt"
    eps = 8.0 / 255.0
    pgd_step = 2.0 / 255.0
    batch_size = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    print(f"testset size = {len(testset)}")

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == 'cuda')
    )

    model = ResNet18().to(device)
    model = load_checkpoint(model, model_path, device)
    model.eval()

    print(f"Loaded model from {model_path}")

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = outputs.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    natural_acc = correct / total
    print(f"\nNatural Accuracy (no attack): {natural_acc:.4f}")

    attacks = [
        ("CW-Linf", attack_cw_inf, {
            'confidence': 50,
            'num_classes': 10,
            'epsilon': eps,
            'step_size': pgd_step,
            'num_steps': 30
        }),
        ("FGSM", _fgsm_whitebox, {
            'epsilon': eps
        }),
        ("PGD-100", _pgd100_whitebox, {
            'epsilon': eps,
            'step_size': pgd_step
        }),
    ]

    for name, fn, kwargs in attacks:
        print(f"\n=== {name} ===")
        try:
            nat_acc, rob_acc = eval_adv_test_whitebox_with_attack(
                model,
                device,
                test_loader,
                attack_fn=fn,
                attack_kwargs=kwargs
            )
            print(f"{name}: natural_acc={nat_acc:.4f}, robust_acc={rob_acc:.4f}")
        except Exception as e:
            print(f"{name} failed: {e}")

    print("\n=== AutoAttack ===")
    try:
        nat_acc, rob_acc = eval_autoattack(model, device, test_loader, eps)
        print(f"AutoAttack: natural_acc={nat_acc:.4f}, robust_acc={rob_acc:.4f}")
    except Exception as e:
        print(f"AutoAttack failed: {e}")

    print("\nCIFAR-10 + ResNet18 evaluation finished.")


if __name__ == '__main__':
    main()
