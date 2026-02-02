import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.utils import set_seed, get_device
from library.data_loader import get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import train_one_epoch, validate, predict_tta

# Constants
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
INPUT_DIR = "./input"

# Hyperparameters for Fast Baseline
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.6254545454545455


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")

    # 3. Prepare Data Loaders
    # Using full dataset as it is small (~500 total), ensuring fast execution within minutes
    train_loader = get_dataloader(
        train_df,
        root_dir=INPUT_DIR,
        phase="train",
        batch_size=BATCH_SIZE,
        load_cached_data=True,
    )
    val_loader = get_dataloader(
        val_df,
        root_dir=INPUT_DIR,
        phase="val",
        batch_size=BATCH_SIZE,
        load_cached_data=True,
    )

    # 4. Initialize Model
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.2, pretrained=True)
    model = model.to(device)

    # 5. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 6. Training Loop
    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)

    # 7. Final Validation Assessment
    # Load best model for accurate metric calculation
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()
    all_targets = []
    all_probs = []

    # Re-run validation inference to get predictions for analysis
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    final_auc = roc_auc_score(all_targets, all_probs)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_df_analysis = val_df.copy()
    val_df_analysis["prob"] = all_probs
    val_df_analysis["error"] = np.abs(
        val_df_analysis["MGMT_value"] - val_df_analysis["prob"]
    )

    # Extract feature: FLAIR Slice Count (Volume Depth)
    # We count files in the directory to check if volume size correlates with error
    slice_counts = []
    for _, row in val_df_analysis.iterrows():
        flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
        try:
            count = len(os.listdir(flair_path))
        except Exception:
            count = 0
        slice_counts.append(count)

    val_df_analysis["flair_count"] = slice_counts

    # Calculate correlation
    if val_df_analysis["flair_count"].std() > 0:
        corr = val_df_analysis["error"].corr(val_df_analysis["flair_count"])
        print(f"Correlation between Error and FLAIR Slice Count: {corr}")
    else:
        print("Could not calculate correlation (constant slice count).")

    # 9. Submission
    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_df = pd.read_csv("./metadata/test.csv")
        test_loader = get_dataloader(
            test_df,
            root_dir=INPUT_DIR,
            phase="test",
            batch_size=BATCH_SIZE,
            load_cached_data=True,
        )

        # Use Test-Time Augmentation
        test_probs = predict_tta(model, test_loader, device)

        submission_df = pd.DataFrame(
            {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": test_probs}
        )

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
