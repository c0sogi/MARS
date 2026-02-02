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
    Evaluates the model on the validation set using global inference (no sliding window).
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Forward pass (Global context, no sliding window)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    try:
        # Macro-average ROC AUC
        # Calculate per-class AUCs to handle potential NaNs from missing classes
        aucs = roc_auc_score(all_targets, all_preds, average=None)
        # Filter out NaNs (undefined AUCs for classes with no positive samples)
        valid_aucs = aucs[~np.isnan(aucs)]
        if len(valid_aucs) > 0:
            auc = np.mean(valid_aucs)
        else:
            auc = 0.5
    except ValueError:
        # Handle edge cases where no class has valid targets
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
    """
    model.eval()
    results = []

    print("Generating submission predictions...")

    with torch.no_grad():
        for inputs, _, rec_ids in test_loader:
            inputs = inputs.to(device)
            B = inputs.size(0)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

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
