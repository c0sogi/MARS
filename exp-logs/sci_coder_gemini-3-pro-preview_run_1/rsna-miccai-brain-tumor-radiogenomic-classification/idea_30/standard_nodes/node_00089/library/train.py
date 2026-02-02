import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library import config, utils, data, model


def train_one_epoch(train_loader, net, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    net.train()
    losses = utils.AverageMeter()

    for images, targets, _ in train_loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(val_loader, net, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    net.eval()
    losses = utils.AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = net(images)
            loss = criterion(outputs, targets)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Handle edge case where only one class is present in batch/set
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return losses.avg, auc


def run_training(load_cached_data=True):
    """
    Orchestrates the training process:
    1. Sets seeds.
    2. Loads data.
    3. Initializes model, optimizer, loss.
    4. Runs training loop with validation and early stopping.
    5. Saves the best model.
    """
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    print(f"Device: {device}")

    # Load Data
    print("Initializing Data Loaders...")
    train_loader = data.get_dataloader("train", load_cached_data=load_cached_data)
    val_loader = data.get_dataloader("val", load_cached_data=load_cached_data)

    # Initialize Model
    print("Initializing SIRV EfficientNet Model...")
    net = model.SIRVEfficientNet().to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler (Optional but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    # Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(train_loader, net, criterion, optimizer, device)
        val_loss, val_auc = validate_one_epoch(val_loader, net, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                best_model_path,
            )
            print(f"  -> New Best AUC! Model saved to {best_model_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_model_path


def predict_and_submit(model_path, load_cached_data=True):
    """
    Loads the best model, runs inference on the test set, and generates the submission file.
    """
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    # Load Test Data
    print("Initializing Test Loader...")
    test_loader = data.get_dataloader("test", load_cached_data=load_cached_data)

    # Load Model
    print(f"Loading model from {model_path}...")
    net = model.SIRVEfficientNet().to(device)
    checkpoint = utils.load_checkpoint(model_path, net)
    if checkpoint is None:
        raise FileNotFoundError(f"Could not load model checkpoint from {model_path}")

    net.eval()

    bra_ids = []
    probabilities = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = net(images)
            preds = torch.sigmoid(outputs)

            probabilities.extend(preds.cpu().numpy().flatten())
            bra_ids.extend(ids.numpy().flatten())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": bra_ids, "MGMT_value": probabilities})

    # Ensure BraTS21ID is formatted correctly (5 digits string if needed, but sample uses int)
    # The sample submission shows IDs like 00001, but pandas reads as int often.
    # The task description shows:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # However, the sample_submission.csv provided in metadata info shows:
    # BraTS21ID (int64)
    # We will stick to the format provided in the sample submission.

    # Sort by ID
    df_sub = df_sub.sort_values("BraTS21ID")

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(df_sub.head())


def main(load_cached_data=True):
    """
    Main pipeline execution.
    """
    # 1. Train
    best_model_path = run_training(load_cached_data=load_cached_data)

    # 2. Inference & Submit
    predict_and_submit(best_model_path, load_cached_data=load_cached_data)
