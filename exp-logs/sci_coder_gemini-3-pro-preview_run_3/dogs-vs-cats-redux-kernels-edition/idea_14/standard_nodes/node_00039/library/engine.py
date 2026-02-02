import torch
import torch.nn as nn
from library.utils import AverageMeter
from library.config import DISTILLATION_ALPHA, DISTILLATION_TEMP


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of standard training for a teacher model.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The training dataloader yielding (images, targets).
        optimizer (Optimizer): The optimizer.
        device (str): Device to run on ('cuda' or 'cpu').
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {losses.avg}")
    return losses.avg


def train_distill_one_epoch(student, teachers, loader, optimizer, device, epoch):
    """
    Performs one epoch of distillation training for the student model.

    Args:
        student (nn.Module): The student model (MaxViT) to train.
        teachers (list[nn.Module]): List of frozen teacher models.
        loader (DataLoader): DualResolutionDataset loader yielding (img_teacher, img_student, target).
        optimizer (Optimizer): The optimizer for the student.
        device (str): Device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average combined loss for the epoch.
    """
    student.train()
    for teacher in teachers:
        teacher.eval()

    losses = AverageMeter()
    hard_losses = AverageMeter()
    soft_losses = AverageMeter()

    # Hard Loss: Standard BCE against ground truth
    criterion_hard = nn.BCEWithLogitsLoss()

    # Soft Loss: BCE against teacher probabilities (Equivalent to KL Div for binary)
    criterion_soft = nn.BCEWithLogitsLoss()

    for batch_idx, (img_teacher, img_student, targets) in enumerate(loader):
        img_teacher = img_teacher.to(device)
        img_student = img_student.to(device)
        targets = targets.to(device).unsqueeze(1)

        # 1. Generate Soft Targets from Teachers
        with torch.no_grad():
            teacher_logits_list = []
            for teacher in teachers:
                # Teachers see the higher resolution image (Pipeline A)
                logits = teacher(img_teacher)
                teacher_logits_list.append(logits)

            # Average the logits from all teachers
            avg_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)

            # Convert to probabilities with temperature scaling
            teacher_probs = torch.sigmoid(avg_teacher_logits / DISTILLATION_TEMP)

        # 2. Student Forward Pass
        optimizer.zero_grad()
        # Student sees the native resolution image (Pipeline B)
        student_logits = student(img_student)

        # 3. Compute Losses
        loss_hard = criterion_hard(student_logits, targets)

        # For soft loss, scale student logits by temperature to match teacher distribution
        loss_soft = criterion_soft(student_logits / DISTILLATION_TEMP, teacher_probs)

        # Combined Loss
        loss = (DISTILLATION_ALPHA * loss_hard) + (
            (1.0 - DISTILLATION_ALPHA) * loss_soft
        )

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_student.size(0))
        hard_losses.update(loss_hard.item(), img_student.size(0))
        soft_losses.update(loss_soft.item(), img_student.size(0))

    print(
        f"Epoch [{epoch}] Distill Loss: {losses.avg} (Hard: {hard_losses.avg}, Soft: {soft_losses.avg})"
    )
    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader yielding (images, targets).
        device (str): Device to run on.

    Returns:
        float: Average Log Loss (BCE) on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    print(f"Validation Loss: {losses.avg}")
    return losses.avg
