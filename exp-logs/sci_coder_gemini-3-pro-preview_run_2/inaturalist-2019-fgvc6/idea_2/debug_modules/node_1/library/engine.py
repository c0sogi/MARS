import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, device, epoch):
    """
    Handles the training of one epoch using Mixed Precision.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Automatic Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            output = model(images)
            loss = criterion(output, targets)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        acc1 = accuracy(output, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

    print(f"Epoch: {epoch} Train Loss: {losses.avg} Train Acc: {top1.avg}")
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            acc1 = accuracy(output, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

    # Print full precision as requested
    print(f"Validation Loss: {losses.avg} Validation Acc: {top1.avg}")
    return losses.avg, top1.avg


def inference(test_loader, model, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward pass
            output = model(images)

            # Get top 5 predictions (sorted by confidence)
            _, topk_indices = torch.topk(output, k=5, dim=1)

            all_ids.extend(image_ids)
            all_preds.append(topk_indices.cpu().numpy())

    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Format predictions as space-separated strings
    formatted_preds = []
    for row in all_preds:
        pred_str = " ".join(map(str, row))
        formatted_preds.append(pred_str)

    # Create DataFrame
    df = pd.DataFrame({"id": all_ids, "predicted": formatted_preds})

    # Save to CSV
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
