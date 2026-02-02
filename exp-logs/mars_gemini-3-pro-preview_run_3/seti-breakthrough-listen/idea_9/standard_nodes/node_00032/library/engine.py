import os
import numpy as np
import torch
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, mixup_data, mixup_criterion, get_score


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        # Forward pass
        # Squeeze(1) to match target shape (B)
        outputs = model(images).squeeze(1)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    valid_labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            valid_labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    auc = get_score(valid_labels, preds)

    return losses.avg, auc


def inference(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    TTA Strategy: Original, H-Flip, V-Flip, H+V Flip.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_preds = []

            # TTA Loop: Iterate through all 4 combinations of H and V flips
            for flip_h in [False, True]:
                for flip_v in [False, True]:
                    img_aug = images.clone()

                    # Apply augmentations on the fly
                    if flip_h:
                        img_aug = torch.flip(img_aug, dims=[-1])  # Time axis
                    if flip_v:
                        img_aug = torch.flip(img_aug, dims=[-2])  # Freq axis

                    out = model(img_aug).squeeze(1)
                    batch_preds.append(torch.sigmoid(out).cpu().numpy())

            # Average predictions across TTA variations
            batch_preds = np.mean(batch_preds, axis=0)
            preds.append(batch_preds)

    return np.concatenate(preds)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
    save_path=os.path.join(Config.WORK_DIR, "best_model.pth"),
):
    """
    Main training loop with Early Stopping.
    """
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full Precision)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping & Model Saving
        if val_auc > best_auc:
            print(
                f"AUC Improved ({best_auc:.15f} -> {val_auc:.15f}). Saving model to {save_path}"
            )
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.15f}")
    return best_auc


def create_submission(
    model, loader, device, test_df, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating predictions with TTA...")
    predictions = inference(model, loader, device)

    test_df["target"] = predictions

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_df[["id", "target"]].to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
