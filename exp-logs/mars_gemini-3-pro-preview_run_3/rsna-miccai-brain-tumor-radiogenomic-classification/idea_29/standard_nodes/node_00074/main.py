import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import get_device
from library.data import get_datasets
from library.model import VAMSHDNet
from library.train import train_epoch, evaluate, predict_test_set


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # Using load_cached_data=True to utilize preprocessed .npy files if available
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = VAMSHDNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model_runfile.pth")

    # Limit epochs to Config.EPOCHS (15) which is sufficient for a fast baseline
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation
    # Load the best model for final assessment
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, final_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()
    val_probs = []
    val_targets = []

    # Collect predictions
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            targets = target.numpy().flatten()
            val_probs.extend(probs)
            val_targets.extend(targets)

    # Create DataFrame for analysis
    df_val_results = pd.DataFrame(
        {"BraTS21ID": val_dataset.ids, "target": val_targets, "prob": val_probs}
    )

    # Calculate Error Magnitude
    df_val_results["error"] = np.abs(df_val_results["target"] - df_val_results["prob"])

    # Load metadata to get input features (slice counts)
    try:
        df_meta = pd.read_parquet(Config.VAL_META_PATH)
        # Ensure ID format consistency
        df_meta["BraTS21ID"] = df_meta["BraTS21ID"].astype(str)

        # Merge results with metadata
        df_analysis = pd.merge(df_val_results, df_meta, on="BraTS21ID", how="left")

        # Extract features for correlation
        analysis_features = {}
        analysis_features["target_class"] = df_analysis["target"]

        modalities = ["flair", "t1w", "t1wce", "t2w"]
        for mod in modalities:
            col_name = f"{mod}_paths"
            if col_name in df_analysis.columns:
                # Count number of files/slices
                analysis_features[f"{mod}_slice_count"] = df_analysis[col_name].apply(
                    lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
                )

        print("Correlation between Error Magnitude and Input Features:")
        for name, values in analysis_features.items():
            if len(values) == len(df_analysis):
                # Using numpy for correlation to avoid scipy dependency issues if any
                if np.std(values) > 0 and np.std(df_analysis["error"]) > 0:
                    corr = np.corrcoef(df_analysis["error"], values)[0, 1]
                    print(f" - {name}: {corr:.4f}")
                else:
                    print(f" - {name}: N/A (Constant value)")

    except Exception as e:
        print(f"Failure analysis skipped due to error: {e}")

    # 7. Submission
    THRESHOLD = 0.6978181818181817

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for test set
        test_probs = predict_test_set(model, test_loader, device)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_dataset.ids, "MGMT_value": test_probs}
        )

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
