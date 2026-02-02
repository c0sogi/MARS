import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import EfficientNet25D
from library.train import train_one_epoch, validate, predict_and_submit


def analyze_failures(val_probs, val_targets, val_meta_path="./metadata/val.parquet"):
    """
    Performs failure analysis by correlating prediction errors with input data properties.
    """
    print("Performing failure analysis...")
    if not os.path.exists(val_meta_path):
        print(f"Metadata file not found at {val_meta_path}. Skipping failure analysis.")
        return

    df = pd.read_parquet(val_meta_path)

    # Ensure lengths match (in case of dropped batches or mismatches, though unlikely with shuffle=False)
    if len(df) != len(val_probs):
        print(
            f"Warning: Metadata length ({len(df)}) != Predictions length ({len(val_probs)})"
        )
        min_len = min(len(df), len(val_probs))
        df = df.iloc[:min_len]
        val_probs = val_probs[:min_len]
        val_targets = val_targets[:min_len]

    # Calculate Absolute Error
    df["pred"] = val_probs
    df["target"] = val_targets
    df["error"] = np.abs(df["target"] - df["pred"])

    # Extract Features for Correlation (e.g., Slice Counts)
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    feature_cols = []
    for mod in modalities:
        col_name = f"{mod}_paths"
        count_col = f"{mod}_count"
        # Count number of files in the list
        df[count_col] = df[col_name].apply(lambda x: len(x) if x is not None else 0)
        feature_cols.append(count_col)

    # Calculate correlations
    print("Correlation between Error and Input Features:")
    for col in feature_cols:
        if df[col].std() > 0:  # Avoid correlation with constant columns
            corr = df["error"].corr(df[col])
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: NaN (Constant value)")


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    working_dir = "./working/idea_opt"
    os.makedirs(working_dir, exist_ok=True)
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # Configuration
    BATCH_SIZE = 8
    EPOCHS = 15
    LR = 1e-4
    THRESHOLD = 0.6978181818181817

    print(f"Running on device: {device}")

    # 2. Data Loading
    # debug=False to use full dataset for best performance
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=True,
        debug=False,
        input_dir="./input",
    )

    # 3. Model Initialization
    # in_channels=64 (4 modalities * 16 slices)
    model = EfficientNet25D(in_channels=64, num_classes=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")

    # 5. Final Validation & Failure Analysis
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run inference on validation set
    val_targets_list = []
    val_probs_list = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            val_targets_list.append(targets.numpy())
            val_probs_list.append(probs.cpu().numpy().flatten())

    val_targets = np.concatenate(val_targets_list)
    val_probs = np.concatenate(val_probs_list)

    final_auc = roc_auc_score(val_targets, val_probs)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # REQUIRED: Failure Analysis
    analyze_failures(val_probs, val_targets)

    # 6. Submission
    if final_auc > THRESHOLD:
        print(
            f"Validation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(
            model, test_loader, device, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"Validation metric ({final_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
