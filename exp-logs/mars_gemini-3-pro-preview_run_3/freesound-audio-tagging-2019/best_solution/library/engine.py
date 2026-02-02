import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, calculate_lwlrap
from library.dataset import get_class_names


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for _, (data, target, _) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        # Apply Mixup if enabled
        if Config.mixup_alpha > 0:
            lam = np.random.beta(Config.mixup_alpha, Config.mixup_alpha)
            index = torch.randperm(data.size(0)).to(device)

            mixed_data = lam * data + (1 - lam) * data[index]
            mixed_target = lam * target + (1 - lam) * target[index]

            output = model(mixed_data)
            loss = criterion(output, mixed_target)
        else:
            output = model(data)
            loss = criterion(output, target)

        losses.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    print(f"Epoch: {epoch} Train Loss: {losses.avg}")
    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and computes LWLRAP.
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target, _ in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(output)

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    score = calculate_lwlrap(all_targets, all_preds)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation LWLRAP: {score}")

    return losses.avg, score


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs=Config.epochs
):
    """
    Runs the full training loop with Early Stopping.
    """
    best_score = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    counter = 0

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        _, val_score = validate(model, val_loader, criterion, device)

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0

            # Save best model to disk
            torch.save(best_weights, Config.checkpoint_path)
            print(f"New best model saved with LWLRAP: {best_score}")
        else:
            counter += 1
            print(f"No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights before returning
    model.load_state_dict(best_weights)
    return model


def predict(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    all_preds = []
    all_fnames = []

    with torch.no_grad():
        for data, _, fnames in test_loader:
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output)

            all_preds.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

    all_preds = np.concatenate(all_preds, axis=0)

    # Retrieve class names
    class_names = get_class_names()

    # Create DataFrame
    sub_df = pd.DataFrame(all_preds, columns=class_names)
    sub_df.insert(0, "fname", all_fnames)

    # Save submission
    sub_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
