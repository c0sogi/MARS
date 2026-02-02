import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Import pre-defined classes and utilities to avoid re-implementation
from library.config import Config, HybridResFunnel, ConformerBlock, GLUBlock, set_seed
from library.dataset import get_dataloaders

# ==========================================
# Model Aliases
# ==========================================
# Mapping library implementations to the specific names requested in the description
ConformerHybridResFunnel = HybridResFunnel
ResFunnelBlock = GLUBlock


# ==========================================
# Training and Inference Pipeline
# ==========================================
def train_and_predict(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=5,
    subset_size=None,
    device=Config.DEVICE,
    output_submission=Config.OUTPUT_SUBMISSION,
):
    """
    Trains the ConformerHybridResFunnel model, performs validation with early stopping,
    and generates the final submission file.

    Args:
        epochs (int): Maximum training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for AdamW.
        patience (int): Number of epochs to wait for improvement before early stopping.
        subset_size (int, optional): If provided, limits the number of samples per epoch (for debugging).
        device (str): Device to train on ('cuda' or 'cpu').
        output_submission (str): Path to save the submission CSV.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Initializing pipeline on {device}...")

    # Load Data
    # subset_size is handled by breaking the loop, as loaders are pre-configured
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, batch_size=batch_size
    )

    # Initialize Model
    # The HybridResFunnel expects the Config class to access architectural params
    model = ConformerHybridResFunnel(Config).to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs (Patience: {patience})...")

    for epoch in range(epochs):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for i, (x_cat, x_cont, y) in enumerate(train_loader):
            # Debugging constraint: limit dataset size
            if subset_size and (i * batch_size >= subset_size):
                break

            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

        # --- Validation Step ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for i, (x_cat, x_cont, y) in enumerate(val_loader):
                if subset_size and (i * batch_size >= subset_size):
                    break

                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                logits = model(x_cat, x_cont)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.numpy())

        if len(val_preds) > 0:
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_auc = roc_auc_score(val_targets, val_preds)
        else:
            val_auc = 0.0

        # Print metrics (Full Precision)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.10f} | Val AUC: {val_auc:.10f} | LR: {current_lr:.10f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs. Best AUC: {best_auc:.10f}"
            )
            break

        scheduler.step()

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # ==========================================
    # Inference
    # ==========================================
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    model.eval()
    test_preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # ==========================================
    # Submission
    # ==========================================
    os.makedirs(os.path.dirname(output_submission), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "target": test_preds})
    submission.to_csv(output_submission, index=False)
    print(f"Submission saved to {output_submission}")
