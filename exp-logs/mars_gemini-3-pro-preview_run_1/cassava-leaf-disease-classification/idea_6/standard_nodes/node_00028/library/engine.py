import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CFG
from library.utils import AverageMeter, get_score
from library.dataset import rand_bbox


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy Loss that accepts soft targets (probabilities) instead of hard labels.
    Required for MixUp and CutMix training.
    """

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        # x: logits (Batch, NumClasses)
        # target: soft probabilities (Batch, NumClasses)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def train_one_epoch(
    epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None
):
    """
    Performs one epoch of training with MixUp/CutMix and Gradient Accumulation.
    """
    model.train()

    losses = AverageMeter()
    start = time.time()

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Prepare targets (One-Hot Encoding)
        # Labels are indices (0-4), convert to float tensor for mixing
        targets = F.one_hot(labels, CFG.num_classes).float()

        # Apply MixUp / CutMix Augmentation
        if np.random.rand() < CFG.mixup_prob:
            # Generate random permutation for mixing
            rand_index = torch.randperm(batch_size).to(device)
            target_a = targets
            target_b = targets[rand_index]

            # 50% chance for MixUp, 50% for CutMix
            if np.random.rand() < 0.5:
                # MixUp
                lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
                images = lam * images + (1 - lam) * images[rand_index]
                targets = lam * target_a + (1 - lam) * target_b
            else:
                # CutMix
                lam = np.random.beta(CFG.cutmix_alpha, CFG.cutmix_alpha)
                # rand_bbox expects (H, W) tuple
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape[2:], lam)

                # Adjust lambda to match the exact pixel area replaced
                lam = 1 - (
                    (bbx2 - bbx1)
                    * (bby2 - bby1)
                    / (images.shape[-1] * images.shape[-2])
                )

                # Apply CutMix patch
                images[:, :, bbx1:bbx2, bby1:bby2] = images[
                    rand_index, :, bbx1:bbx2, bby1:bby2
                ]
                targets = lam * target_a + (1 - lam) * target_b

        # Forward Pass
        y_preds = model(images)
        loss = loss_fn(y_preds, targets)

        # Gradient Accumulation
        loss = loss / CFG.grad_accumulation
        loss.backward()

        if (step + 1) % CFG.grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics (scale loss back up for reporting)
        losses.update(loss.item() * CFG.grad_accumulation, batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"Elapsed: {time.time() - start:.0f}s"
            )

    return losses.avg


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    """
    Performs validation on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    preds = []
    valid_labels = []
    start = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Construct One-Hot targets for compatibility with SoftTargetCrossEntropy
            targets = F.one_hot(labels, CFG.num_classes).float()

            y_preds = model(images)
            loss = loss_fn(y_preds, targets)

            losses.update(loss.item(), batch_size)

            # Store predictions (Softmax probabilities)
            preds.append(y_preds.softmax(1).to("cpu").numpy())
            valid_labels.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    # Calculate Accuracy
    pred_labels = predictions.argmax(1)
    score = get_score(valid_labels, pred_labels)

    print(
        f"EVAL: [{epoch + 1}] Loss: {losses.avg:.6f} Accuracy: {score:.6f} Elapsed: {time.time() - start:.0f}s"
    )

    return losses.avg, score


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set, optionally using Test Time Augmentation (TTA).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # 1. Original Image Prediction
            out = model(images)
            out = out.softmax(1)

            # 2. Test Time Augmentation (TTA)
            if CFG.tta:
                # Horizontal Flip (dim 3 is width)
                out += model(torch.flip(images, [3])).softmax(1)

                # Vertical Flip (dim 2 is height)
                out += model(torch.flip(images, [2])).softmax(1)

                # Average predictions
                out /= 3.0

            preds.append(out.to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions
