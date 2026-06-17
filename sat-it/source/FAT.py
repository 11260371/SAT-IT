# FAT.py / fat_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def early_stop_pgd_linf(
        model,
        x_natural,
        y,
        step_size=0.007,
        epsilon=0.031,
        perturb_steps=10,
        tau=0,
        loss_type="ce",
        rand_init=True,
        omega=0.001):
    """Early-stopped PGD for FAT (ce) or FAT-TRADES (kl)."""

    epsilon = float(epsilon)
    step_size = float(step_size)
    perturb_steps = int(perturb_steps)
    tau = int(tau)

    device = x_natural.device
    batch_size = x_natural.size(0)

    was_training = model.training
    model.eval()

    if rand_init:
        noise = omega * torch.empty_like(x_natural).uniform_(-1.0, 1.0)
        x_adv = x_natural.detach() + noise
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach().clone()

    if loss_type == "kl":
        with torch.no_grad():
            natural_probs = F.softmax(model(x_natural), dim=1)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    remain_tau = torch.full((batch_size,), tau, dtype=torch.long, device=device)
    saved_adv = x_adv.detach().clone()

    mask_shape = (batch_size,) + (1,) * (x_natural.dim() - 1)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        logits_adv = model(x_adv)
        pred = logits_adv.argmax(dim=1)

        misclassified = pred.ne(y)

        stop_mask = misclassified & (remain_tau <= 0) & (~finished)
        if stop_mask.any():
            saved_adv[stop_mask] = x_adv.detach()[stop_mask]
            finished[stop_mask] = True

        continue_mask = misclassified & (remain_tau > 0) & (~finished)
        if continue_mask.any():
            remain_tau[continue_mask] -= 1

        if finished.all():
            break

        active = ~finished

        if loss_type == "kl":
            loss_adv = F.kl_div(
                F.log_softmax(logits_adv[active], dim=1),
                natural_probs[active],
                reduction="sum"
            )
        else:
            loss_adv = F.cross_entropy(
                logits_adv[active],
                y[active],
                reduction="sum"
            )

        grad = torch.autograd.grad(loss_adv, x_adv)[0]

        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())

        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = torch.where(
            finished.view(mask_shape),
            saved_adv,
            x_adv
        )

    saved_adv[~finished] = x_adv.detach()[~finished]

    if was_training:
        model.train()
    else:
        model.eval()

    return saved_adv.detach()


def fat_loss(
        model,
        x_natural,
        y,
        optimizer=None,
        step_size=0.007,
        epsilon=0.031,
        perturb_steps=10,
        tau=0,
        rand_init=True,
        omega=0.001):
    """FAT loss: early-stopped PGD + cross-entropy."""

    x_adv = early_stop_pgd_linf(
        model=model,
        x_natural=x_natural,
        y=y,
        step_size=step_size,
        epsilon=epsilon,
        perturb_steps=perturb_steps,
        tau=tau,
        loss_type="ce",
        rand_init=rand_init,
        omega=omega
    )

    if optimizer is not None:
        optimizer.zero_grad()

    model.train()
    logits_adv = model(x_adv)
    loss = F.cross_entropy(logits_adv, y)

    return loss


def fat_trades_loss(
        model,
        x_natural,
        y,
        optimizer=None,
        step_size=0.007,
        epsilon=0.031,
        perturb_steps=10,
        beta=5.0,
        tau=0,
        rand_init=True,
        omega=0.001):
    """FAT-TRADES loss: early-stopped PGD + TRADES objective."""

    batch_size = x_natural.size(0)

    x_adv = early_stop_pgd_linf(
        model=model,
        x_natural=x_natural,
        y=y,
        step_size=step_size,
        epsilon=epsilon,
        perturb_steps=perturb_steps,
        tau=tau,
        loss_type="kl",
        rand_init=rand_init,
        omega=omega
    )

    if optimizer is not None:
        optimizer.zero_grad()

    model.train()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    loss_natural = F.cross_entropy(logits, y)

    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits, dim=1),
        reduction="sum"
    ) / batch_size

    loss = loss_natural + beta * loss_robust

    return loss
