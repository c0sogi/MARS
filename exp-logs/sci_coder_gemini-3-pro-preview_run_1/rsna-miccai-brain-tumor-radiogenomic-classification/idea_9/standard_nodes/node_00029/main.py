import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import MGMTClassifier
from library.train import train_one_epoch, validate_epoch, generate_submission


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # Load cached data if available to speed up execution
    train_loader, val_loader, test_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = MGMTClassifier().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_auc = validate_epoch(model, val_loader, device)

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            break

    # 6. Load Best Model for Final Evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 7. Final Validation Metric
    # Must print full precision as required
    final_val_auc = validate_epoch(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # 8. Failure Analysis
    print("Performing Failure Analysis on Validation Set...")
    model.eval()
    preds = []

    # Collect predictions manually to map back to dataframe for analysis
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs.flatten())

    # Map slice predictions back to subjects
    val_df_analysis = val_loader.dataset.df.copy()
    val_df_analysis["pred"] = preds

    # Aggregate to subject level (Consensus)
    subject_df = (
        val_df_analysis.groupby("BraTS21ID")
        .agg({"pred": "mean", "MGMT_value": "first"})
        .reset_index()
    )

    # Calculate Error Magnitude
    subject_df["error"] = (subject_df["pred"] - subject_df["MGMT_value"]).abs()

    # Calculate Correlation with available features
    # We analyze correlation with the Target (to check for class bias)
    # and ID (to check for temporal/site bias)
    correlations = subject_df[["BraTS21ID", "MGMT_value", "error"]].corr()["error"]
    print("Correlation between Model Error and Features:")
    print(correlations)

    # 9. Conditional Submission
    threshold = 0.6705454545454544
    if final_val_auc > threshold:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_val_auc} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
