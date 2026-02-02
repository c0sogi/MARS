import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, calculate_metric
from library.dataset import get_dataloaders
from library.model import BirdResNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, labels, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Apply Sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auc_score = calculate_metric(all_targets, all_preds)

    return epoch_loss, auc_score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            rec_ids = rec_ids.numpy()

            # Format predictions
            # Id = rec_id * 100 + species_idx
            for i in range(len(rec_ids)):
                rid = rec_ids[i]
                probs = preds[i]
                for species_idx, prob in enumerate(probs):
                    row_id = int(rid * 100 + species_idx)
                    results.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(results)
    # Sort by Id just to be clean, though not strictly required if all IDs are present
    df_sub = df_sub.sort_values("Id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False):
    """
    Main function to run the training loop, evaluation, and submission generation.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Model
    model = BirdResNet(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [+] New Best AUC! Model saved.")
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc:.15f}")

    # Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    return best_auc
