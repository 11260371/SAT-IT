import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def at_loss(model,
            x_natural,
            y,
            optimizer,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10):

    """
    Standard Adversarial Training (PGD-L∞)
    """
    model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()

    logits_adv = model(x_adv)
    loss = F.cross_entropy(logits_adv, y)

    return loss
