import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def lbgat_loss(model, model_teacher,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    criterion_kl = nn.KLDivLoss(size_average=False)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    model.eval()
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()

    logits_adv = model(x_adv)
    logits = model(x_natural)
    logits_teacher = model_teacher(x_natural)

    adv_probs_log = F.log_softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)

    loss_mse = mse(logits_adv, logits_teacher) + ce(logits_teacher, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(adv_probs_log, nat_probs)
    loss = loss_mse + float(beta) * loss_robust

    return loss
