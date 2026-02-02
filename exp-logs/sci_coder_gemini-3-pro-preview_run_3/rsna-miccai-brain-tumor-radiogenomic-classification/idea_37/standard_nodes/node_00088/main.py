import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import model
from library import data_loader


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load training data (shuffled)
    train_loader = data_loader.get_dataloader(
        "train", shuffle=True, load_cached_data=True
    )
    # Load validation data (not shuffled, for consistent evaluation)
    val_loader = data_loader.get_dataloader("val", shuffle=False, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Siamese SNR-Net...")
    net = model.SiameseSNRNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    print(f"Starting training for {config.N_EPOCHS} epochs...")
    net.train()

    for epoch in range(config.N_EPOCHS):
        running_loss = 0.0
        batch_count = 0

        for x_even, x_odd, labels in train_loader:
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)
            labels = labels.to(device).unsqueeze(1)  # Shape: (B, 1)

            optimizer.zero_grad()

            # Forward pass
            outputs = net(x_even, x_odd)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}/{config.N_EPOCHS} - Loss: {avg_loss:.6f}")

    # 5. Validation
    print("Performing validation...")
    net.eval()
    val_preds = []
    val_targets = []
    val_ids = (
        val_loader.dataset.ids
    )  # Retrieve IDs from dataset to map back to metadata

    with torch.no_grad():
        for x_even, x_odd, labels in val_loader:
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)

            outputs = net(x_even, x_odd)
            probs = torch.sigmoid(outputs).cpu().numpy()

            val_preds.extend(probs.flatten())
            val_targets.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Metric
    if len(np.unique(val_targets)) > 1:
        val_auc = roc_auc_score(val_targets, val_preds)
    else:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Running failure analysis...")
    # Load metadata to get features
    val_meta_df = pd.read_parquet(config.VAL_META_PATH)

    # Calculate absolute errors
    errors = np.abs(val_preds - val_targets)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "BraTS21ID": val_ids,
            "error": errors,
            "target": val_targets,
            "prediction": val_preds,
        }
    )

    # Merge with metadata to get file paths/counts
    # Ensure IDs match type (string in parquet, likely string/int in dataset)
    # The dataset IDs are derived from the parquet, so they should match format.
    analysis_df = analysis_df.merge(val_meta_df, on="BraTS21ID", how="left")

    # Compute slice counts per modality as features
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate length of the list of paths
        analysis_df[f"{mod}_count"] = analysis_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )

    # Calculate correlation between error and slice counts
    print("Correlation between Error Magnitude and Modality Slice Counts:")
    for mod in modalities:
        feat = f"{mod}_count"
        if feat in analysis_df.columns:
            # Use pandas corr()
            corr = analysis_df["error"].corr(analysis_df[feat])
            print(f"  {feat}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"Validation metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_loader = data_loader.get_dataloader(
            "test", shuffle=False, load_cached_data=True
        )
        test_ids = test_loader.dataset.ids
        test_preds = []

        # Inference
        net.eval()
        with torch.no_grad():
            for x_even, x_odd in test_loader:
                x_even = x_even.to(device)
                x_odd = x_odd.to(device)

                outputs = net(x_even, x_odd)
                probs = torch.sigmoid(outputs).cpu().numpy()
                test_preds.extend(probs.flatten())

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        # Format BraTS21ID as integer for submission (as per sample_submission.csv info)
        # Note: The IDs in test folder are strings like "00001", but sample_submission.csv usually expects ints.
        # The task description sample shows: 00001,0.5 in text, but the dataframe view shows ints.
        # We will convert to int to be safe and consistent with typical BraTS submissions.
        submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
