import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_score


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using Mixed Precision and Mixup.
    """
    model.train()

    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    print_freq = Config.print_freq

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        if Config.mixup_prob > 0 and np.random.rand() < Config.mixup_prob:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.mixup_alpha, use_cuda=True
            )
            # Reshape targets for BCEWithLogitsLoss
            targets_a = targets_a.view(-1, 1)
            targets_b = targets_b.view(-1, 1)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            targets = targets.view(-1, 1)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Note: Scheduler is typically stepped per epoch in the main loop for CosineAnnealingWarmRestarts

        if step % print_freq == 0 or step == (len(dataloader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(dataloader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model and calculates ROC AUC.
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    valid_targets = []

    print_freq = Config.print_freq

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)
            batch_size = images.size(0)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities
            preds.append(outputs.sigmoid().to("cpu").numpy())
            valid_targets.append(targets.to("cpu").numpy())

            if step % print_freq == 0 or step == (len(dataloader) - 1):
                print(
                    f"EVAL: [{step}/{len(dataloader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
                )

    predictions = np.concatenate(preds)
    valid_targets = np.concatenate(valid_targets)

    score = get_score(valid_targets, predictions)

    # Print metrics without rounding as requested
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {score}")

    return losses.avg, score


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    """
    model.eval()
    preds = []

    print_freq = Config.print_freq

    with torch.no_grad():
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # 1. Original Prediction
            with torch.cuda.amp.autocast():
                output = model(images)
                pred_orig = output.sigmoid()

            # 2. TTA: Horizontal Flip
            if Config.tta:
                # Flip along the frequency axis (width), which is the last dimension
                images_flip = torch.flip(images, dims=[-1])
                with torch.cuda.amp.autocast():
                    output_flip = model(images_flip)
                    pred_flip = output_flip.sigmoid()

                # Average predictions
                pred = (pred_orig + pred_flip) / 2.0
            else:
                pred = pred_orig

            preds.append(pred.to("cpu").numpy())

            if step % print_freq == 0:
                print(f"Inference step {step}/{len(dataloader)}")

    predictions = np.concatenate(preds)
    return predictions
