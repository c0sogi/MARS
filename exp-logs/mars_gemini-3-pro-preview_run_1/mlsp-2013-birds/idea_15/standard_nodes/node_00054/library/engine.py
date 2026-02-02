import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import roc_auc_score

from library.utils import save_checkpoint
from library.dataset import mixup_batch


def train_one_epoch(model, loader, optimizer, device, epoch, mixup_alpha=0.2):
    """
    Trains the model for one epoch using Mixup regularization.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for i, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Apply Mixup
        data, target = mixup_batch(data, target, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and computes ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()

            # Apply Sigmoid to get probabilities
            preds = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Calculate ROC AUC
    # Cite {debug_lesson_2}: Explicitly Handle NaN Returns in Metric Calculations
    # We calculate AUC per column to handle missing classes in the validation set robustly.
    aucs = []
    num_classes = all_targets.shape[1]

    for i in range(num_classes):
        try:
            # Check if class exists in targets
            if len(np.unique(all_targets[:, i])) < 2:
                continue

            score = roc_auc_score(all_targets[:, i], all_preds[:, i])
            if not np.isnan(score):
                aucs.append(score)
        except ValueError:
            pass

    if len(aucs) > 0:
        auc = np.mean(aucs)
    else:
        # Fallback if no classes are valid
        auc = 0.5

    return running_loss / len(loader), auc


def run_training_session(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    save_dir,
    patience=10,
):
    """
    Manages the full training session including SWA and Early Stopping safety logic.
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()

    # SWA Configuration
    swa_start = int(epochs * 0.75)
    swa_model = AveragedModel(model)
    # Use a conservative constant LR for SWA phase
    swa_scheduler = SWALR(optimizer, swa_lr=1e-3)

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    print(f"Starting training session. Epochs: {epochs}. SWA Start: {swa_start}")

    for epoch in range(epochs):
        in_swa = epoch >= swa_start

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        if in_swa:
            # SWA Phase: Update averaged model and use SWA scheduler
            swa_model.update_parameters(model)
            swa_scheduler.step()

            # Log progress (Validation is optional here for speed, but good for monitoring)
            print(f"Epoch {epoch+1}/{epochs} [SWA] - Train Loss: {train_loss:.6f}")

        else:
            # Standard Phase: Validate and check Early Stopping
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.15f}"
            )

            is_best = val_auc > best_auc
            # Save if best or if it's the first epoch to ensure file exists
            if is_best or epoch == 0:
                if is_best:
                    best_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0

                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_auc": best_auc,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=True,
                    save_dir=save_dir,
                    best_filename="model_base_best.pth",
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.15f}"
                )
                early_stopped = True
                break

    if early_stopped:
        print("Training terminated early. Returning best base model.")
        best_model_path = os.path.join(save_dir, "model_base_best.pth")
        if not os.path.exists(best_model_path):
            # Fallback: Save current model if no best model was ever saved
            print("Warning: No best model found. Saving current state as best.")
            save_checkpoint(
                {
                    "epoch": epochs,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                save_dir=save_dir,
                best_filename="model_base_best.pth",
            )
        return best_model_path
    else:
        print("Finalizing SWA...")
        # Update Batch Norm statistics for the averaged model
        update_bn(train_loader, swa_model, device=device)

        # Validate final SWA model
        swa_val_loss, swa_val_auc = validate(swa_model, val_loader, criterion, device)
        print(
            f"SWA Final Validation - Loss: {swa_val_loss:.6f} - AUC: {swa_val_auc:.15f}"
        )

        save_path = os.path.join(save_dir, "model_swa.pth")
        torch.save(
            {"state_dict": swa_model.state_dict(), "auc": swa_val_auc}, save_path
        )

        return save_path


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for data, _, rec_ids in loader:
            data = data.to(device)

            # 1. Original Forward Pass
            out1 = model(data)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip Forward Pass (Time Inversion)
            data_flip = torch.flip(data, dims=[3])  # [B, C, H, W], flip W
            out2 = model(data_flip)
            prob2 = torch.sigmoid(out2)

            # Average probabilities
            avg_prob = (prob1 + prob2) / 2.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.append(rec_ids.numpy())

    return np.vstack(all_preds), np.concatenate(all_ids)
