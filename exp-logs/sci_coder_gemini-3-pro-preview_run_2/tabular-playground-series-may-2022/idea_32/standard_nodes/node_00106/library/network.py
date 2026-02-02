import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloaders
from library.modules import EarlyFusionResFunnelModel


def train_model():
    """
    Orchestrates the training process including data loading, model initialization,
    optimizer configuration, training loop, validation, and early stopping.
    """
    print("Initializing Training...")

    # 1. Device Setup
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = EarlyFusionResFunnelModel(config=Config).to(device)

    # 4. Optimizer with Strict Decoupled Weight Decay
    # Group 1: Decay (Weights), Group 2: No Decay (Biases, Norms, PosEmbed)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify parameters that should not decay
        if (
            (param.ndim <= 1)
            or (name.endswith(".bias"))
            or ("norm" in name)
            or ("pos_embed" in name)
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY_ENCODER},
        {"params": no_decay_params, "weight_decay": Config.WEIGHT_DECAY_BIAS},
    ]

    optimizer = optim.AdamW(optimizer_groups, lr=Config.LEARNING_RATE)

    # 5. Scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 6. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 7. Training Loop Variables
    best_auc = 0.0
    patience_counter = 0

    print("Starting Training Loop...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()

            logits = model(continuous, sequence)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                continuous = batch["continuous"].to(device)
                sequence = batch["sequence"].to(device)
                targets = batch["target"].to(device)

                logits = model(continuous, sequence)
                probs = torch.sigmoid(logits).squeeze(1)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > (best_auc + Config.EARLY_STOPPING_MIN_DELTA):
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")


def predict_and_submit():
    """
    Loads the best model, performs inference on the test set, and saves the submission file.
    """
    print("Starting Inference...")

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load Data
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=True)

    # Load Model
    model = EarlyFusionResFunnelModel(config=Config).to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No model checkpoint found. Using random initialization.")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)

            logits = model(continuous, sequence)
            probs = torch.sigmoid(logits).squeeze(1)

            all_preds.extend(probs.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main execution entry point.
    """
    try:
        train_model()
        predict_and_submit()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


# Execute the pipeline
if __name__ == "__main__":
    run()
