import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

##used before epoch 75
def loss_reg_entropy(nat_probs, weights):
    entropy = -torch.sum(nat_probs * torch.log(nat_probs + 1e-12), dim=1)  # [B]
    loss_reg = (weights * entropy).mean()
    return loss_reg


##used >=epoch 75
def loss_reg_entropy2(adv_probs, weights):
    entropy = -torch.sum(adv_probs * torch.log(adv_probs + 1e-12), dim=1)  # [B]
    loss_reg = (weights * entropy).mean()
    return loss_reg


def calculate_uncertainty_weight(logits_uncertainty, perturb_steps, batch_size, device, y):
    # logits_uncertainty: [perturb_steps, batch_size, num_classes]
    pred_uncertainty = torch.zeros((perturb_steps, batch_size), dtype=torch.long).to(device)

    for i in range(perturb_steps):
        pred_uncertainty[i] = torch.argmax(logits_uncertainty[i], dim=1)

    pred_uncertainty_t = pred_uncertainty.transpose(0, 1)  # [batch_size, perturb_steps]

    flip_mask = (pred_uncertainty_t != y.unsqueeze(1))  # [batch_size, perturb_steps]
    has_flip = flip_mask.any(dim=1).float()  # [batch_size]

    probs = torch.softmax(logits_uncertainty, dim=2)  # [perturb_steps, batch_size, num_classes]
    mean_probs = probs.mean(dim=0)  # [batch_size, num_classes]

    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=1)  # [batch_size]
    expected_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=2), dim=0)  # [batch_size]
    mutual_information = predictive_entropy - expected_entropy  # [batch_size]

    mi_min, mi_max = mutual_information.min(), mutual_information.max()
    norm_mi = (mutual_information - mi_min) / (mi_max - mi_min + 1e-12)

    weights = has_flip + (1 - has_flip) * norm_mi

    return weights  # [batch_size]


def ema_loss(model,
              x_natural,
              y,
              optimizer,
                ema_probs,
                indices,
              device,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
                a=0.9,
            b=0.9,
            c=0.1,
              distance='l_inf'
              ):
    kl = nn.KLDivLoss(reduction='none')

    num_classes = 10

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    logits_uncertainty = torch.zeros(
        (perturb_steps, x_natural.size(0), num_classes),
        device=x_natural.device
    )  # [perturb_steps, batch_size, num_classes]

    if distance == 'l_inf':
        for idx in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            logits_uncertainty[idx] = model(x_adv).detach()

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)  # [B, C]

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])  # [B]

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()  # [B]

    alpha = 1.0

    weights = calculate_uncertainty_weight(logits_uncertainty, perturb_steps, batch_size, device, y)

    weights = torch.clamp(weights * alpha, min=0.0, max=1.0)

    probs_natural = F.softmax(logits, dim=1)
    probs_adv = F.softmax(logits_adv, dim=1)

    mixed_probs = c * probs_natural + (1 - c) * probs_adv

    ema_probs[indices] = a * ema_probs[indices] + (1 - a) * mixed_probs.detach()

    y_onehot = torch.zeros_like(probs_natural).scatter_(1, y.unsqueeze(1), 1)
    y_soft = b * ema_probs[indices] + (1 - b) * y_onehot

    loss_natural = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * weights)

    loss_reg = loss_reg_entropy(nat_probs, weights)

    lambda_ = 0.6

    loss = loss_natural + loss_adv + float(beta) * loss_robust - lambda_* loss_reg

    return loss
