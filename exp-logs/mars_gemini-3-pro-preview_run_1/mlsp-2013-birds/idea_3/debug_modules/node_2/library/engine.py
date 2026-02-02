import os
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import mixup_data, mixup_criterion


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, labels, alpha=Config.MIXUP_ALPHA, use_cuda=(device != "cpu")
        )

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set using sliding window inference.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    # Sliding window parameters
    crop_width = Config.CROP_WIDTH
    stride = Config.STRIDE

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # inputs shape: (B, 3, H, W_full)
            # We need to generate crops for each image in the batch

            B, C, H, W = inputs.shape

            # Calculate start indices for sliding window
            # Ensure we cover the whole image.
            # Strategy: Standard stride, plus one final crop right-aligned to the end.
            starts = list(range(0, W - crop_width + 1, stride))
            if W > crop_width and (starts[-1] + crop_width < W):
                starts.append(W - crop_width)
            elif W <= crop_width:
                starts = [0]  # Should be handled by padding in dataset, but safe check

            # Collect all crops for this batch
            batch_crops = []

            for i in range(B):
                img = inputs[i]  # (3, H, W)
                for s in starts:
                    # Crop: (3, H, crop_width)
                    crop = img[:, :, s : s + crop_width]
                    batch_crops.append(crop)

            # Stack all crops: (B * N_windows, 3, H, crop_width)
            if len(batch_crops) > 0:
                batch_crops_tensor = torch.stack(batch_crops)

                # Forward pass on all crops
                # Output: (B * N_windows, Num_Classes)
                crop_outputs = model(batch_crops_tensor)

                # Reshape to (B, N_windows, Num_Classes) to average
                num_windows = len(starts)
                crop_outputs = crop_outputs.view(B, num_windows, Config.NUM_CLASSES)

                # Average logits across windows
                avg_outputs = torch.mean(crop_outputs, dim=1)
            else:
                # Fallback for empty batch (should not happen)
                avg_outputs = torch.zeros((B, Config.NUM_CLASSES)).to(device)

            # Compute loss on averaged logits
            loss = criterion(avg_outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(avg_outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    try:
        # Macro-average ROC AUC
        auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        # Handle edge cases where a class might not be present in the batch/split
        auc = 0.5

    return epoch_loss, auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
):
    """
    Main training loop with Early Stopping.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.15f}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.15f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    Uses the same sliding window aggregation as validation.
    """
    model.eval()
    results = []

    crop_width = Config.CROP_WIDTH
    stride = Config.STRIDE

    print("Generating submission predictions...")

    with torch.no_grad():
        for inputs, _, rec_ids in test_loader:
            inputs = inputs.to(device)
            B, C, H, W = inputs.shape

            # Sliding Window Logic (Same as validate)
            starts = list(range(0, W - crop_width + 1, stride))
            if W > crop_width and (starts[-1] + crop_width < W):
                starts.append(W - crop_width)
            elif W <= crop_width:
                starts = [0]

            batch_crops = []
            for i in range(B):
                img = inputs[i]
                for s in starts:
                    crop = img[:, :, s : s + crop_width]
                    batch_crops.append(crop)

            if len(batch_crops) > 0:
                batch_crops_tensor = torch.stack(batch_crops)
                crop_outputs = model(batch_crops_tensor)
                num_windows = len(starts)
                crop_outputs = crop_outputs.view(B, num_windows, Config.NUM_CLASSES)
                avg_outputs = torch.mean(crop_outputs, dim=1)
                probs = torch.sigmoid(avg_outputs)
            else:
                probs = torch.zeros((B, Config.NUM_CLASSES)).to(device)

            probs_np = probs.cpu().numpy()
            rec_ids_np = rec_ids.numpy()

            # Format: Id = rec_id * 100 + species_id
            for i in range(B):
                rid = rec_ids_np[i]
                row_probs = probs_np[i]
                for species_idx in range(Config.NUM_CLASSES):
                    submission_id = rid * 100 + species_idx
                    probability = row_probs[species_idx]
                    results.append({"Id": submission_id, "Probability": probability})

    # Create DataFrame and save
    df_sub = pd.DataFrame(results)

    # Sort by Id to be neat (though not strictly required if all IDs are present)
    df_sub = df_sub.sort_values("Id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
