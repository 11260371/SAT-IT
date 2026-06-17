import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

from model.resnet import ResNet18


student = ResNet18()
teacher = ResNet18()

student_path = "./checkpoints/student.pt"  # Replace with your student model checkpoint (.pt/.pth)
teacher_path = "./checkpoints/teacher.pt"  # Replace with your teacher/source model checkpoint (.pt/.pth)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


def extract_model_state_dict(obj):
    if not isinstance(obj, dict):
        return obj

    candidate_keys = ["model", "state_dict", "model_state_dict", "net"]
    for key in candidate_keys:
        if key in obj:
            return obj[key]

    return obj


def load_model_weight(model, path, device):
    if not path or not isinstance(path, str):
        raise ValueError("Model weight path is empty; set student_path / teacher_path.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")

    obj = torch.load(path, map_location="cpu")
    state_dict = extract_model_state_dict(obj)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Could not parse state_dict from: {path}")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if isinstance(k, str) else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Loaded model weights from: {path}")
    return model


def cw_margin_loss(logits, labels, confidence=50.0):
    """Untargeted CW margin loss."""
    num_classes = logits.size(1)

    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    correct_logit = torch.sum(one_hot * logits, dim=1)
    wrong_logit = torch.max((1.0 - one_hot) * logits - 1e4 * one_hot, dim=1)[0]

    loss = torch.clamp(wrong_logit - correct_logit + confidence, min=0.0)
    return loss.mean()


def attack_cw_inf(model, batch_data, batch_labels, attack_iters=20, step_size=0.003, epsilon=8.0 / 255.0, confidence=50.0):
    adv_data = batch_data.detach() + torch.empty_like(batch_data).uniform_(-epsilon, epsilon)
    adv_data = torch.clamp(adv_data, 0.0, 1.0).detach()

    for _ in range(attack_iters):
        adv_data.requires_grad_(True)

        model.zero_grad()

        logits = model(adv_data)
        loss = cw_margin_loss(logits, batch_labels, confidence=confidence)
        loss.backward()

        grad = adv_data.grad
        if grad is None:
            raise RuntimeError("adv_data.grad is None; check that model forward pass is differentiable.")

        adv_data = adv_data.detach() + step_size * torch.sign(grad.detach())

        perturbation = torch.clamp(adv_data - batch_data, min=-epsilon, max=epsilon)
        adv_data = torch.clamp(batch_data + perturbation, min=0.0, max=1.0).detach()

    return adv_data


def eval_cw_blackbox(student, teacher, testloader, attack_iters=20, step_size=0.003, epsilon=8.0 / 255.0, confidence=50.0):
    print("=============== CW L_inf Black-box Transfer Attack Evaluation ===============")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = 0
    correct = 0

    for step, (test_batch_data, test_batch_labels) in enumerate(testloader):
        test_batch_data = test_batch_data.float().to(device)
        test_batch_labels = test_batch_labels.to(device)

        adv_data = attack_cw_inf(
            teacher,
            test_batch_data,
            test_batch_labels,
            attack_iters=attack_iters,
            step_size=step_size,
            epsilon=epsilon,
            confidence=confidence
        )

        with torch.no_grad():
            logits = student(adv_data)
            predictions = torch.argmax(logits, dim=1)

        correct += (predictions == test_batch_labels).sum().item()
        total += test_batch_labels.size(0)

        if (step + 1) % 20 == 0 or (step + 1) == len(testloader):
            current_acc = correct / total
            print(f"[INFO] Step [{step + 1}/{len(testloader)}] | current robust acc: {current_acc:.4f}")

    robust_acc = correct / total
    print(f"[RESULT] student robust acc under CW L_inf black-box transfer attack: {robust_acc:.4f}")


if __name__ == "__main__":
    print(f"[INFO] Using device: {device}")

    student = load_model_weight(student, student_path, device)
    teacher = load_model_weight(teacher, teacher_path, device)

    eval_cw_blackbox(
        student,
        teacher,
        testloader,
        attack_iters=20,
        step_size=0.003,
        epsilon=8.0 / 255.0,
        confidence=50.0
    )
