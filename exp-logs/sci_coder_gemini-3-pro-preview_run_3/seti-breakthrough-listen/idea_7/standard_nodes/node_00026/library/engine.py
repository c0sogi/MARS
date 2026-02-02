import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import mixup_data, mixup_criterion, get_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for data, target in loader:
        stream_a = data["stream_a"].to(device)
        stream_b = data["stream_b"].to(device)
        target = target.to(device).view(-1, 1)

        # Siamese Mixup Strategy:
        # Concatenate streams to ensure identical mixup parameters (lambda and permutation)
        # are applied to both the signal (A) and the reference (B).
        # stream_a: (B, 3, H, W), stream_b: (B, 3, H, W) -> combined: (B, 6, H, W)
        combined = torch.cat([stream_a, stream_b], dim=1)

        # Apply Mixup
        mixed_combined, y_a, y_b, lam = mixup_data(
            combined, target, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Split back into streams
        mixed_a = mixed_combined[:, :3, :, :]
        mixed_b = mixed_combined[:, 3:, :, :]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_a, mixed_b)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * stream_a.size(0)
        dataset_size += stream_a.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            stream_a = data["stream_a"].to(device)
            stream_b = data["stream_b"].to(device)
            target = target.to(device).view(-1, 1)

            outputs = model(stream_a, stream_b)
            loss = criterion(outputs, target)

            running_loss += loss.item() * stream_a.size(0)
            dataset_size += stream_a.size(0)

            # Convert logits to probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate ROC AUC
    auc = get_score(all_targets, all_preds)
    epoch_loss = running_loss / dataset_size

    return epoch_loss, auc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping and Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load the best model weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def generate_submission(
    model, loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for data, _ in loader:
            stream_a = data["stream_a"].to(device)
            stream_b = data["stream_b"].to(device)

            outputs = model(stream_a, stream_b)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds).flatten()

    # Load test metadata to map predictions to IDs
    # We assume the loader iterates in the same order as the CSV (shuffle=False)
    test_df = pd.read_csv(Config.TEST_CSV)

    if len(preds) != len(test_df):
        print(
            f"Warning: Number of predictions ({len(preds)}) does not match number of test samples ({len(test_df)})"
        )

    test_df["target"] = preds

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save submission file
    submission_df = test_df[["id", "target"]]
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
