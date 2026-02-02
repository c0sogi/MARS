import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from library import config, utils, data, model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    # Use tqdm for progress tracking if running interactively, otherwise silent or simple print
    # Per instructions: "Only print the required information. Do not print progress bars"
    # So we will iterate directly.

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)
        images_contra = batch["image_contra"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Siamese network takes both target and contralateral images
        logits = model(images, images_contra)

        # Compute Loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer Step
        # Note: Gradient clipping is explicitly disabled per instructions
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            images_contra = batch["image_contra"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            logits = model(images, images_contra)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
    else:
        y_pred = np.array([])
        y_true = np.array([])

    # Compute Metric
    pf1 = utils.probabilistic_f1(y_true, y_pred)

    return avg_loss, pf1


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set, aggregates by prediction_id,
    and saves to CSV.
    """
    model.eval()
    results = []

    print("Generating submission predictions...")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            images_contra = batch["image_contra"].to(device, non_blocking=True)
            prediction_ids = batch["prediction_id"]

            logits = model(images, images_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("Warning: No predictions generated.")
        # Create empty submission with correct columns just in case
        df_sub = pd.DataFrame(columns=["prediction_id", "cancer"])
    else:
        # Aggregate: Max probability per prediction_id (handling multiple views)
        df_sub = df_results.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True, debug_subset_size=None):
    """
    Main training function.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=load_cached_data, debug_subset_size=debug_subset_size
    )

    # 3. Model
    print("Initializing model...")
    net = model.SiameseEfficientNet()
    net.to(device)

    # 4. Optimization
    # Aggressive positive weighting for imbalance
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=config.MIN_LR
    )

    # 5. Training Loop
    best_pf1 = -1.0
    best_epoch = -1
    patience = 3
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            net, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        # Checkpoint & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_pf1": best_pf1,
            }
            utils.save_checkpoint(checkpoint, filename="best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best pF1: {best_pf1}")
            break

    # 6. Submission
    print(f"Training finished. Best Epoch: {best_epoch} with pF1: {best_pf1}")

    # Load best model for inference
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model for submission...")
        utils.load_checkpoint(best_model_path, net, device=device)
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    generate_submission(net, test_loader, device, config.SUBMISSION_PATH)
