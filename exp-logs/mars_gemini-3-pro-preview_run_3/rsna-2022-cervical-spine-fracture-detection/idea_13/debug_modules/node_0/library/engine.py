import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library import utils


def weighted_multitask_loss(logits, targets):
    """
    Implicitly Weighted Multi-Task Loss.
    L = mean(BCE_C1..C7) + BCE_Patient
    """
    # Vertebrae: Indices 0-6
    # reduction='mean' averages over batch and the 7 classes
    vert_loss = F.binary_cross_entropy_with_logits(
        logits[:, :7], targets[:, :7], reduction="mean"
    )

    # Patient: Index 7
    patient_loss = F.binary_cross_entropy_with_logits(
        logits[:, 7], targets[:, 7], reduction="mean"
    )

    return vert_loss + patient_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (global_input, local_input, targets) in enumerate(dataloader):
        global_input = global_input.to(device)
        local_input = local_input.to(device)
        targets = targets.to(device)

        batch_size = global_input.size(0)

        optimizer.zero_grad()

        logits = model(global_input, local_input)
        loss = weighted_multitask_loss(logits, targets)

        loss.backward()

        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for global_input, local_input, targets in dataloader:
            global_input = global_input.to(device)
            local_input = local_input.to(device)
            targets = targets.to(device)

            batch_size = global_input.size(0)

            logits = model(global_input, local_input)
            loss = weighted_multitask_loss(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for metric calculation
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate for metric calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate competition metric
    metric = utils.competition_metric(all_targets, all_preds)

    return epoch_loss, metric


def fit(model, train_loader, val_loader, device):
    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Setup Scheduler (Decoupled Cosine Annealing)
    # T_max is 1.5x epochs as per config
    t_max = int(Config.EPOCHS * Config.T_MAX_MULT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    best_metric = float("inf")
    best_model_path = os.path.join("working", "best_model.pth")

    # Early Stopping parameters
    patience = 5
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_metric = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.8f}")
        print(f"Val Loss:   {val_loss:.8f}")
        print(f"Val Metric: {val_metric:.8f}")

        # Checkpointing and Early Stopping
        # We minimize the competition metric (Log Loss)
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Metric improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Metric: {best_metric:.8f}")


def inference(model, test_loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    best_model_path = os.path.join("working", "best_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found. Using current model weights.")

    model.eval()
    results = []

    # Columns for the 8 outputs
    target_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    print("Generating predictions...")

    with torch.no_grad():
        # We need to access the StudyInstanceUIDs from the dataset to map rows
        # The test_loader dataset (RSNADataset) has a df attribute
        test_df = test_loader.dataset.df

        # Iterate loader
        # Note: DataLoader might shuffle if not configured carefully, but usually test loaders don't shuffle.
        # We assume the loader yields batches in the order of the dataframe.

        batch_idx = 0
        for global_input, local_input, _ in test_loader:
            global_input = global_input.to(device)
            local_input = local_input.to(device)

            logits = model(global_input, local_input)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Map predictions to study IDs
            # Get the slice of the dataframe corresponding to this batch
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + global_input.size(0)
            batch_ids = test_df.iloc[start_idx:end_idx]["StudyInstanceUID"].values

            for i, study_id in enumerate(batch_ids):
                study_probs = probs[i]

                # Create 8 rows per study
                for col_idx, col_name in enumerate(target_cols):
                    row_id = f"{study_id}_{col_name}"
                    prob = study_probs[col_idx]
                    results.append({"row_id": row_id, "fractured": prob})

            batch_idx += 1

    # Create submission DataFrame
    submission_df = pd.DataFrame(results)

    # Save
    Config.create_directories()
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
