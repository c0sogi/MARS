import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, map5, save_checkpoint
from library.dataset import get_class_list


def train_fn(dataloader, model, criterion, optimizer, device, scaler):
    """
    Performs one epoch of training.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.
        scaler: GradScaler for Automatic Mixed Precision (AMP).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            # Model forward with labels returns ArcFace logits (with margin)
            outputs = model(images, labels)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Performs validation with Test-Time Augmentation (TTA).

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The neural network model.
        device: 'cuda' or 'cpu'.

    Returns:
        float: MAP@5 score.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])

            # Forward pass in inference mode (labels=None)
            # Returns scaled cosine similarities
            logits_orig = model(images, labels=None)
            logits_flip = model(images_flip, labels=None)

            # Average predictions
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Get top 5 predictions
            _, top_indices = torch.topk(avg_logits, k=5, dim=1)

            preds.extend(top_indices.cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())

    # Format targets for map5 (list of lists)
    targets_list = [[t] for t in targets]

    score = map5(targets_list, preds)
    return score


def inference_fn(dataloader, model, device):
    """
    Generates predictions for the test set using TTA.

    Args:
        dataloader: PyTorch DataLoader for test data.
        model: The neural network model.
        device: 'cuda' or 'cpu'.

    Returns:
        pd.DataFrame: DataFrame containing 'Image' and 'Id' columns.
    """
    model.eval()
    results = []

    # Load class list to map indices back to IDs
    # We use load_cached_data=True as it should have been generated during training setup
    class_list = get_class_list(load_cached_data=True)
    idx_to_class = {i: c for i, c in enumerate(class_list)}

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device, non_blocking=True)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])

            logits_orig = model(images, labels=None)
            logits_flip = model(images_flip, labels=None)

            avg_logits = (logits_orig + logits_flip) / 2.0

            _, top_indices = torch.topk(avg_logits, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            # Convert indices to string labels
            for i, img_id in enumerate(image_ids):
                pred_indices = top_indices[i]
                pred_labels = [idx_to_class[idx] for idx in pred_indices]
                pred_str = " ".join(pred_labels)
                results.append({"Image": img_id, "Id": pred_str})

    return pd.DataFrame(results)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    epochs,
    patience=5,
):
    """
    Main training loop with Early Stopping.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        epochs: Total number of epochs.
        patience: Early stopping patience.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    best_score = 0.0
    patience_counter = 0

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scaler)

        # Validate
        val_score = eval_fn(val_loader, model, device)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MAP@5: {val_score}"
        )

        # Checkpoint & Early Stopping
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                result_dir=Config.WORKING_DIR,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val MAP@5: {best_score}")


def generate_submission(dataloader, model, device):
    """
    Generates and saves the submission CSV file.

    Args:
        dataloader: Test DataLoader.
        model: Trained model.
        device: Device.
    """
    print("Generating submission...")
    df_sub = inference_fn(dataloader, model, device)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
