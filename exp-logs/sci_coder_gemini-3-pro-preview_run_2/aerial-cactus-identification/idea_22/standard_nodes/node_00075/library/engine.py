import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library import config, utils, model

# Global Loss Function
CRITERION = nn.BCEWithLogitsLoss()


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        device: The device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = utils.AverageMeter()

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = CRITERION(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        device: The device to run on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    losses = utils.AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = CRITERION(outputs, labels)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate results from all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate ROC AUC
    auc = utils.calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc


def predict_tta(model, loader, device):
    """
    Predicts using Test Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model: The PyTorch model.
        loader: The test DataLoader.
        device: The device to run on.

    Returns:
        tuple: (List of IDs, Numpy array of probabilities)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is Width)
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is Height)
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds)

    # Flatten to 1D array if necessary
    if all_preds.ndim > 1:
        all_preds = all_preds.flatten()

    return all_ids, all_preds


def train_seed(seed, dataloaders, device, epochs=config.EPOCHS):
    """
    Trains a single model instance for a specific seed with Early Stopping.

    Args:
        seed (int): The random seed.
        dataloaders (dict): Dictionary of DataLoaders.
        device (str): Device to train on.
        epochs (int): Maximum number of epochs.

    Returns:
        float: Best validation AUC achieved.
    """
    print(f"\n[Seed {seed}] Initializing training...")
    utils.set_seed(seed)

    # Initialize Model
    net = model.WideSEResNet().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.ETA_MIN
    )

    best_auc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    model_filename = f"model_seed_{seed}.pth"

    for epoch in range(epochs):
        train_loss = train_one_epoch(net, dataloaders["train"], optimizer, device)
        val_loss, val_auc = validate(net, dataloaders["val"], device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Checkpoint Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            utils.save_checkpoint(net, optimizer, epoch, val_auc, model_filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"[Seed {seed}] Finished. Best AUC: {best_auc} at Epoch {best_epoch+1}")
    return best_auc


def run_inference_ensemble(dataloaders, device):
    """
    Loads all trained seed models, performs TTA inference, averages predictions,
    and saves the submission file.
    """
    print("\nStarting Ensemble Inference...")

    test_loader = dataloaders["test"]
    ensemble_preds = None
    test_ids = []

    successful_models = 0

    for seed in config.SEEDS:
        model_filename = f"model_seed_{seed}.pth"
        net = model.WideSEResNet().to(device)

        try:
            checkpoint = utils.load_checkpoint(net, model_filename, device=device)
            print(
                f"Loaded model for seed {seed} (Epoch {checkpoint['epoch']+1}, AUC {checkpoint['score']})"
            )
        except FileNotFoundError:
            print(f"Warning: Checkpoint for seed {seed} not found. Skipping.")
            continue

        ids, preds = predict_tta(net, test_loader, device)

        if ensemble_preds is None:
            ensemble_preds = preds
            test_ids = ids
        else:
            ensemble_preds += preds

        successful_models += 1

    if successful_models == 0:
        print("Error: No models were loaded successfully. Cannot generate submission.")
        return

    # Average predictions across all models
    final_preds = ensemble_preds / successful_models

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save to disk
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
