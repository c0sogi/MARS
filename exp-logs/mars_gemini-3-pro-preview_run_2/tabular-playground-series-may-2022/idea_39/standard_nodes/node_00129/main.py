import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Ensure library can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.dataset import get_dataloaders
from library.model import HybridSwiGLUNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Use full training schedule for multi-stage annealing (Cite solution_lesson_node_00128)
    # Config.EPOCHS is set to 35 in config.py

    # Check for existing processed data in root working dir to save time
    if os.path.exists("./working/processed_data.npz"):
        print(
            "Found cached data at ./working/processed_data.npz. Updating Config path."
        )
        Config.PROCESSED_DATA_PATH = "./working/processed_data.npz"

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading Data...")
    # load_cached_data=True will attempt to load from Config.PROCESSED_DATA_PATH
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing Model...")
    model = HybridSwiGLUNet().to(device)

    # --------------------------------------------------------------------------
    # 4. Training Setup
    # --------------------------------------------------------------------------
    # Use strict decoupled weight decay as per idea
    param_groups = get_optimizer_grouped_parameters(
        model,
        weight_decay_group1=Config.WEIGHT_DECAY_GROUP1,
        weight_decay_group2=Config.WEIGHT_DECAY_GROUP2,
    )

    optimizer = torch.optim.AdamW(param_groups, lr=Config.LEARNING_RATE)

    # Scheduler: Step Decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCELoss()

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0.0
    print(f"Starting Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | LR: {current_lr:.1e} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --------------------------------------------------------------------------
    # 6. Final Evaluation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nLoading Best Model for Final Evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # We need to compute the metric on the full validation set and perform analysis
    # We'll iterate the val_loader one last time to collect everything

    all_targets = []
    all_preds = []
    all_continuous = []  # For failure analysis

    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            outputs = model(continuous, categorical)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_continuous.append(continuous.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_continuous = np.concatenate(all_continuous, axis=0)  # Shape (N, 30)

    final_auc = roc_auc_score(all_targets, all_preds)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Input Features
    print("\n--- Failure Analysis ---")
    errors = np.abs(all_targets - all_preds)

    # Compute correlation for each continuous feature
    feature_correlations = []
    num_features = all_continuous.shape[1]

    for i in range(num_features):
        feat_values = all_continuous[:, i]
        # Avoid correlation calculation if std is 0
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]

        feature_correlations.append((f"Feature_{i:02d}", corr))

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Features correlated with Prediction Error:")
    for name, corr in feature_correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972883264620234

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict(model, test_loader, device)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "target": test_preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
