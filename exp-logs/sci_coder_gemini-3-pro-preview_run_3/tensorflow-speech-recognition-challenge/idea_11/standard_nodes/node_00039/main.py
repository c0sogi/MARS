import os
import sys
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score

# Import from the provided library files
from library.config import train_cfg, path_cfg, model_cfg, set_seed
from library.preprocessing import run_preprocessing
from library.dataset import get_dataloaders, IDX2LABEL
from library.trainer import Trainer
from library.model import MultiScaleHierarchicalSKResNet


def main():
    # 1. Setup and Configuration
    # Set fixed seed for reproducibility
    set_seed(train_cfg.seed)

    # Fast Baseline: Limit epochs to ensure execution within time limits
    # The A100 is fast, but we want to ensure we don't overstay.
    train_cfg.epochs = 12

    print("=== Starting Runfile ===")
    print(f"Device: {train_cfg.device}")

    # 2. Preprocessing
    # Generates/Loads cache and returns paths to metadata CSVs with cache paths
    print("Running Preprocessing...")
    csv_paths = run_preprocessing(load_cached_data=True)

    # 3. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_paths["train"], csv_paths["val"], csv_paths["test"]
    )

    # 4. Training
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.fit(train_loader, val_loader)

    # 5. Final Validation
    print("=== Final Validation ===")

    # Load the best model saved by EarlyStopping
    best_model_path = path_cfg.model_save_path
    model = MultiScaleHierarchicalSKResNet(model_cfg)

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=train_cfg.device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")
        model = trainer.model

    model.to(train_cfg.device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(train_cfg.device)
            # targets are on cpu or gpu, move to cpu for storage

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Calculate Metric
    val_acc = accuracy_score(all_targets, all_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    print("=== Failure Analysis ===")

    # Load validation metadata to correlate with features
    val_meta = pd.read_csv(path_cfg.val_meta)

    # Ensure lengths match (just in case)
    if len(val_meta) != len(all_preds):
        print(
            f"Warning: Metadata rows ({len(val_meta)}) != Prediction count ({len(all_preds)})"
        )
        min_len = min(len(val_meta), len(all_preds))
        val_meta = val_meta.iloc[:min_len]
        all_preds = all_preds[:min_len]
        all_targets = all_targets[:min_len]

    # Create Error Vector (1 = Error, 0 = Correct)
    errors = (np.array(all_preds) != np.array(all_targets)).astype(int)
    val_meta["error"] = errors
    val_meta["pred_idx"] = all_preds

    # Analysis 1: Error Rate by Class
    print("\nError Rate by Class (Top 5):")
    class_error = val_meta.groupby("label")["error"].mean().sort_values(ascending=False)
    print(class_error.head(5))

    # Analysis 2: Correlation with Input Features (Duration)
    # We sample a subset to avoid excessive I/O overhead during analysis
    print("\nCalculating correlation between Error and Audio Duration...")
    sample_size = min(2000, len(val_meta))
    subset_indices = np.random.choice(len(val_meta), size=sample_size, replace=False)
    subset_meta = val_meta.iloc[subset_indices].copy()

    durations = []
    for idx, row in subset_meta.iterrows():
        try:
            full_path = os.path.join(path_cfg.input_dir, row["filepath"])
            info = sf.info(full_path)
            durations.append(info.duration)
        except Exception:
            durations.append(1.0)  # Default fallback

    subset_meta["duration"] = durations

    # Point-Biserial Correlation
    if len(subset_meta) > 0:
        corr = subset_meta["error"].corr(subset_meta["duration"])
        print(f"Correlation between Error and Duration: {corr}")

    # 7. Submission
    threshold = 0.9832324978392394

    if val_acc > threshold:
        print(
            f"\nValidation Metric ({val_acc}) > Threshold ({threshold}). Generating Submission..."
        )

        test_preds = []

        # Test Inference
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(train_cfg.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())

        # Map indices to labels
        pred_labels = [IDX2LABEL[idx] for idx in test_preds]

        # Prepare Submission DataFrame
        test_meta = pd.read_csv(path_cfg.test_meta)

        # Extract filename from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
        fnames = test_meta["filepath"].apply(lambda x: os.path.basename(x)).tolist()

        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save
        save_path = path_cfg.submission_path
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation Metric ({val_acc}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
