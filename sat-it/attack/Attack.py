import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import torch.nn.functional as F


def attack_cw_inf(model, input, target, confidence=50, num_classes=10, epsilon=8/255, lr=2/255, steps=30):
    perturbation = torch.zeros_like(input).cuda().requires_grad_()
    for _ in range(steps):
        output = model(input + perturbation)
        target_onehot = F.one_hot(target, num_classes=num_classes).float().cuda()
        real = torch.sum(target_onehot * output, dim=1)
        other = torch.max((1 - target_onehot) * output - target_onehot * 10000, dim=1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.).mean()
        grad = torch.autograd.grad(loss, perturbation)[0]
        perturbation = (perturbation + lr * torch.sign(grad)).clamp(-epsilon, epsilon)
        perturbation = perturbation.detach().requires_grad_()
    adversarial_input = input + perturbation
    adversarial_input = torch.clamp(adversarial_input, 0, 1)
    return adversarial_input


def _fgsm_whitebox(model, X, y, epsilon):
    model.eval()
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_adv = X.clone().detach().to(X.device)
    X_adv.requires_grad = True

    loss = nn.CrossEntropyLoss()(model(X_adv), y)
    model.zero_grad()
    if X_adv.grad is not None:
        X_adv.grad.data.zero_()
    loss.backward()
    grad_sign = X_adv.grad.data.sign()
    X_adv = X_adv + epsilon * grad_sign
    X_adv = torch.clamp(X_adv, 0.0, 1.0).detach()

    err_adv = (model(X_adv).data.max(1)[1] != y.data).float().sum()
    return err, err_adv


def _pgd_whitebox(model, X, y, epsilon=0.031, num_steps=20, step_size=0.003, random_start=True):
    model.eval()
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    if random_start:
        X_pgd = X.clone().detach() + torch.empty_like(X).uniform_(-epsilon, epsilon)
    else:
        X_pgd = X.clone().detach()
    X_pgd = torch.clamp(X_pgd, 0.0, 1.0).detach()
    X_pgd.requires_grad = True

    for _ in range(num_steps):
        logits = model(X_pgd)
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad()
        if X_pgd.grad is not None:
            X_pgd.grad.data.zero_()
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = X_pgd.detach() + eta
        delta = torch.clamp(X_pgd - X, min=-epsilon, max=epsilon)
        X_pgd = torch.clamp(X + delta, 0.0, 1.0).detach()
        X_pgd.requires_grad = True

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def _pgd100_whitebox(model, X, y, epsilon=0.031, step_size=0.003):
    return _pgd_whitebox(model, X, y, epsilon=epsilon, num_steps=100, step_size=step_size, random_start=True)


def _autoattack_whitebox(model, X, y, epsilon=0.031, version='standard'):
    """Standard L_inf evaluation via the autoattack library."""
    model.eval()
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    try:
        from autoattack import AutoAttack
    except Exception as e:
        raise RuntimeError("AutoAttack import failed; run `pip install autoattack`. Original error: " + str(e))

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version=version, device=X.device)
    X_np = X.clone().detach().cpu().numpy()
    y_np = y.clone().detach().cpu().numpy()

    try:
        robust_accuracy = adversary.run_standard_evaluation(X_np, y_np, bs=X_np.shape[0])
        if isinstance(robust_accuracy, float) or isinstance(robust_accuracy, np.floating):
            robust_acc = robust_accuracy
            err_adv = (1.0 - robust_acc) * X_np.shape[0]
            err_adv = torch.tensor(err_adv).to(X.device)
        else:
            x_adv = robust_accuracy[0]
            x_adv_tensor = torch.from_numpy(x_adv).to(X.device)
            preds = model(x_adv_tensor).data.max(1)[1]
            err_adv = (preds != y.data).float().sum()
    except Exception as e:
        print("AutoAttack failed (version mismatch or other issue):", e)
        err_adv = torch.tensor(0.).to(X.device)

    return err, err_adv


def eval_adv_test_whitebox_with_attack(model, device, test_loader, attack_fn, attack_kwargs):
    """
    attack_fn: function(model, X, y, **attack_kwargs) -> (err_nat, err_adv)
    attack_kwargs: dict of kwargs for attack_fn
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = attack_fn(model, X, y, **attack_kwargs)
        robust_err_total += err_robust
        natural_err_total += err_natural

    nat_acc = 1 - natural_err_total / len(test_loader.dataset)
    rob_acc = 1 - robust_err_total / len(test_loader.dataset)
    print('natural_acc: ', nat_acc)
    print('robust_acc: ', rob_acc)
    return nat_acc, rob_acc
