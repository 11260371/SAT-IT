from __future__ import print_function
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms

from model.resnet_tinyimagenet import *
from lbgat import *
import numpy as np
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='./checkpoints/lbgat_tinyimagenet',
                    help='directory for saving checkpoints (.pt/.pth)')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_dir = './logs/lbgat_tinyimagenet'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True



# setup data loader for Tiny-ImageNet-200
import math
from torch.utils.data import Subset
tinyimagenet_root = './data/tiny-imagenet-200'
train_dir = os.path.join(tinyimagenet_root, 'train')
val_dir = os.path.join(tinyimagenet_root, 'val')

print("train_dir =", train_dir, os.path.exists(train_dir))
print("val_dir   =", val_dir, os.path.exists(val_dir))

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

full_trainset = torchvision.datasets.ImageFolder(
    root=train_dir,
    transform=transform_train
)

full_testset = torchvision.datasets.ImageFolder(
    root=val_dir,
    transform=transform_test
)

def first_k_percent_per_class(dataset, ratio=0.2):
    targets = dataset.targets
    class_to_indices = {}

    for idx, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(idx)

    selected_indices = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        k = math.ceil(len(indices) * ratio)
        selected_indices.extend(indices[:k])

    return Subset(dataset, selected_indices)

trainset = first_k_percent_per_class(full_trainset, ratio=0.2)
testset = first_k_percent_per_class(full_testset, ratio=0.2)

print(f"full_trainset size = {len(full_trainset)}")
print(f"trainset 20% size  = {len(trainset)}")
print(f"full_testset size  = {len(full_testset)}")
print(f"testset 20% size   = {len(testset)}")

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=10,
    pin_memory=True
)


def train(args, model, device, train_loader, optimizer, epoch, model_teacher):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = lbgat_loss(model=model,
                       model_teacher = model_teacher,
                       x_natural=data,
                       y=target,
                       optimizer=optimizer,
                       step_size=args.step_size,
                       epsilon=args.epsilon,
                       perturb_steps=args.num_steps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = total_loss / total_batches
    return avg_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))
    return 1 - natural_err_total / len(test_loader.dataset), 1 - robust_err_total / len(test_loader.dataset)


def main():
    model = ResNet18_tinyimagenet200().to(device)
    model_teacher = ResNet18_tinyimagenet200().to(device)
    # optimizer
    optimizer = optim.SGD([{'params':model.parameters()},{'params':model_teacher.parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    epochs_list = []
    natural_acc = []
    robust_acc = []

    file_name = os.path.join(log_dir, 'train_stats.npz')

    if os.path.exists(file_name):
        data = np.load(file_name)
        epochs_list = list(data['epochs'])
        natural_acc = list(data['natural_acc'])
        robust_acc = list(data['robust_acc'])
        print(f"Loaded prior training stats with {len(epochs_list)} epoch records")

    checkpoint_epoch = 0  # Set > 0 to resume from model_dir/model-res-epoch{N}.pt

    if checkpoint_epoch and checkpoint_epoch > 0:
        model_path = os.path.join(model_dir, f'model-res-epoch{checkpoint_epoch}.pt')
        model_teacher_path = os.path.join(model_dir, f'model-teacher-res-epoch{checkpoint_epoch}.pt')
        opt_path = os.path.join(model_dir, f'opt-res-checkpoint_epoch{checkpoint_epoch}.tar')

        if os.path.exists(model_path) and os.path.exists(opt_path):
            print(f"Loaded checkpoint; resume training from epoch {checkpoint_epoch + 1}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_teacher.load_state_dict(torch.load(model_teacher_path, map_location=device))
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            start_epoch = checkpoint_epoch + 1
        else:
            print("Checkpoint not found; training from scratch")
            start_epoch = 1
            epochs_list = []
            natural_acc = []
            robust_acc = []
    else:
        print("Training from scratch")
        start_epoch = 1
        epochs_list = []
        natural_acc = []
        robust_acc = []

    if start_epoch == 1:
        # Replace with your pretrained teacher checkpoint (.pt/.pth)
        model_teacher_path = "./checkpoints/teacher_mart.pt"
        print("Loaded teacher model", model_teacher_path)
        model_teacher.load_state_dict(torch.load(model_teacher_path, map_location=device))

    for epoch in range(start_epoch, args.epochs + 1):

        adjust_learning_rate(optimizer, epoch)

        start_time = time.time()

        train(args, model, device, train_loader, optimizer, epoch,model_teacher)

        print('================================================================')

        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader)

        print('using time:', time.time() - start_time)

        epochs_list.append(epoch)
        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)
        print('================================================================')

        np.savez(file_name,
                 epochs=np.array(epochs_list),
                 natural_acc=np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in natural_acc]),
                 robust_acc=np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in robust_acc])
                 )

        if epoch % args.save_freq == 0 and epoch >=75:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model-res-epoch{epoch}.pt'))
            torch.save(model_teacher.state_dict(), os.path.join(model_dir, f'model-teacher-res-epoch{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, f'opt-res-checkpoint_epoch{epoch}.tar'))



if __name__ == '__main__':
    main()
