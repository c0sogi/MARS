import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.utils import calculate_roc_auc
from library.loss import WeightedSoftCrossEntropy, get_class_weights
from library.dataset import TARGET_COLS


def train_one_epoch(model, dataloader, criterion, optimizer, device, cutmix_fn):
    """
    Trains the model for one epoch using CutMix regularization.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply CutMix
        # cutmix_fn returns: mixed_images, target_a, target_b, lam
        images, target_a, target_b, lam = cutmix_fn(images, labels)

        # Create soft targets for the weighted loss
        # labels are already one-hot/float, so we can mix them directly
        mixed_labels = lam * target_a + (1 - lam) * target_b

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, mixed_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Store predictions (probabilities) and labels for metric calculation
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches

    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate ROC AUC
    auc_score = calculate_roc_auc(all_labels, all_preds)

    return avg_loss, auc_score


def predict_with_tta(model, dataloader, device):
    """
    Performs inference with Test Time Augmentation (Horizontal Flip).
    Returns image_ids and predicted probabilities.
    """
    model.eval()
    all_preds = []

    # We need image IDs to map predictions back to files
    # The dataloader's dataset object holds these
    image_ids = dataloader.dataset.get_image_ids()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Prediction on original image
            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # 2. Prediction on horizontally flipped image
            # dim 3 is width (B, C, H, W)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)

            # Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0
            all_preds.append(avg_probs.cpu().numpy())

    final_preds = np.vstack(all_preds)
    return image_ids, final_preds


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-4,
    patience=3,
    cutmix_fn=None,
    save_path="./working/best_model.pth",
):
    """
    Orchestrates the training process with Early Stopping.
    """
    # Initialize Criterion with Class Weights
    class_weights = get_class_weights(load_cached_data=True).to(device)
    criterion = WeightedSoftCrossEntropy(weights=class_weights)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cutmix_fn
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def generate_submission(model, test_loader, device, output_dir="./submission"):
    """
    Generates the submission file using TTA predictions.
    """
    print("Generating submission with TTA...")

    # Get predictions
    image_ids, preds = predict_with_tta(model, test_loader, device)

    # Create DataFrame
    # Columns must match sample_submission.csv format
    # image_id, healthy, multiple_diseases, rust, scab
    df = pd.DataFrame(preds, columns=TARGET_COLS)
    df.insert(0, "image_id", image_ids)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
