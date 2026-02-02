import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import ModelConfig, seed_everything
from library.utils import save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import IIResFunnelGLU
from library.engine import train_one_epoch, validate, predict


def perform_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    model.eval()
    all_cont = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)
            target = batch["target"].to(device).view(-1)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits).view(-1)

            all_cont.append(cont.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_cont = np.concatenate(all_cont, axis=0)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate Absolute Error
    errors = np.abs(all_targets - all_preds)

    print("\n--- Failure Analysis ---")
    print("Correlation between Error Magnitude and Continuous Features:")

    # Compute correlation for each continuous feature
    num_features = all_cont.shape[1]
    correlations = []
    for i in range(num_features):
        feat_values = all_cont[:, i]
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((i, corr))
        print(f"Feature {i}: {corr:.6f}")

    # Identify top correlated feature
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(
        f"Most correlated feature with error: Feature {correlations[0][0]} (Corr: {correlations[0][1]:.6f})"
    )
    print("------------------------\n")


def main():
    # 1. Setup
    config = ModelConfig
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=config.DEBUG
    )

    # 3. Model Initialization
    model = IIResFunnelGLU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print("Starting training...")
    best_auc = 0.0
    patience = 0

    # Create directory for saving model if it doesn't exist
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Checkpoint based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            save_checkpoint(
                model, optimizer, None, epoch, val_auc, config.MODEL_SAVE_PATH
            )
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    # 6. Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    checkpoint = load_checkpoint(config.MODEL_SAVE_PATH, model, device=device)
    best_epoch = checkpoint["epoch"]
    best_metric = checkpoint["metric"]

    print(f"Loaded model from epoch {best_epoch}")

    # Required Output: Final Validation Metric
    print(f"Final Validation Metric: {best_metric}")

    # Perform Failure Analysis on Validation Set
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission Logic
    THRESHOLD = 0.9957464342157875

    if best_metric > THRESHOLD:
        print(
            f"Validation AUC ({best_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        predictions = predict(model, test_loader, device)

        submission = pd.DataFrame({"id": test_ids, "target": predictions})

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation AUC ({best_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
