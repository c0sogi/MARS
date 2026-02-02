import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import BraTS25DEfficientNet
from library.train import train_one_epoch, validate, predict_and_submit


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    seed_everything(42)
    device = get_device()

    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    SAVE_DIR = "./working/idea_opt"
    BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"
    THRESHOLD = 0.6832727272727273

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Using load_cached_data=True to speed up execution by using pre-processed .npy files
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = BraTS25DEfficientNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # ==========================================
    # 5. Final Validation & Metrics
    # ==========================================
    # Load best model for analysis
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    model.eval()
    all_targets = []
    all_probs = []

    # Re-run validation inference to get per-sample predictions for failure analysis
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Inputs are already (B, 64, H, W)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate Final Metric
    try:
        final_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        final_auc = 0.5

    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    # Calculate absolute error
    errors = np.abs(all_targets - all_probs)

    # Load validation metadata to get features (slice counts)
    val_df = pd.read_parquet("./metadata/val.parquet")

    # Calculate correlations between error and slice counts
    print("Failure Analysis - Correlation with Error Magnitude:")
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate count of files for each patient
        counts = val_df[col_name].apply(lambda x: len(x) if x is not None else 0).values

        # Ensure lengths match before correlation
        if len(counts) == len(errors):
            corr = np.corrcoef(counts, errors)[0, 1]
            print(f"{mod}_count correlation: {corr:.4f}")

    # ==========================================
    # 7. Conditional Submission
    # ==========================================
    if final_auc > THRESHOLD:
        # predict_and_submit uses the Multi-View Ensemble (Even + Odd slices)
        predict_and_submit(model, test_loader, test_ids, device, SUBMISSION_PATH)
    else:
        print(f"Validation metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
