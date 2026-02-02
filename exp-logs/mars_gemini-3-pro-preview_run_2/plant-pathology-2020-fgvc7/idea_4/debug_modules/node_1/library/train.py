import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import get_dataloaders
from library.model import AppleClassifier
from library.loss import WeightedLabelSmoothCrossEntropy


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Validation loop for one epoch. Returns loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            preds_all.append(probs.cpu().numpy())
            targets_all.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    # Calculate Metric
    val_auc = calculate_metric(targets_all, preds_all)

    return epoch_loss, val_auc


def inference(model, loader, device, use_tta=Config.TTA_FLIP):
    """
    Performs inference on the test set, optionally using Test Time Augmentation (TTA).
    """
    model.eval()
    preds_all = []
    image_ids_all = []

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # Forward pass (Original)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # Test Time Augmentation (Horizontal Flip)
            if use_tta:
                # Flip images horizontally (dim 3 is width in B,C,H,W)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.softmax(logits_flipped, dim=1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            preds_all.append(probs.cpu().numpy())
            image_ids_all.extend(image_ids)

    preds_all = np.concatenate(preds_all, axis=0)
    return image_ids_all, preds_all


def run_training(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main execution function to train the model and generate submission.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 2. Calculate Class Weights (Inverse Class Frequency)
    # Extract labels from the training dataset
    train_labels = train_loader.dataset.labels  # shape (N, 4)
    class_counts = np.sum(train_labels, axis=0)
    num_samples = len(train_labels)
    num_classes = train_labels.shape[1]

    # Weight = Total / (Num_Classes * Count)
    # Add small epsilon to avoid division by zero if any class is missing in debug mode
    class_weights = num_samples / (num_classes * (class_counts + 1e-6))

    print("Class Counts:", class_counts)
    print("Class Weights:", class_weights)

    # Convert weights to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    model = AppleClassifier(pretrained=True)
    model.to(device)

    criterion = WeightedLabelSmoothCrossEntropy(
        weight=class_weights_tensor, smoothing=Config.LABEL_SMOOTHING
    )
    criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.MIN_LR
    )

    # 4. Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [+] Saved new best model. AUC: {best_val_auc}")
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("\nStarting Inference on Test Set...")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found, using current weights.")

    image_ids, preds = inference(model, test_loader, device)

    # 6. Generate Submission
    # The columns must match the order in dataset.LABEL_COLS: ["healthy", "multiple_diseases", "rust", "scab"]
    submission_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": preds[:, 0],
            "multiple_diseases": preds[:, 1],
            "rust": preds[:, 2],
            "scab": preds[:, 3],
        }
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
