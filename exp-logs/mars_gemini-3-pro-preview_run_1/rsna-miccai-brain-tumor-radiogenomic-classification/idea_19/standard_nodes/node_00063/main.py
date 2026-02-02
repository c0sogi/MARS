import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model as model_lib
from library import train as train_lib


def main():
    # ==========================================
    # 1. Setup
    # ==========================================
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    # Run training using the provided library function.
    # This handles data loading, model init, training loop, and saving the best model.
    # We use load_cached_data=True to speed up if data was already processed.
    train_lib.run_training(debug=False, load_cached_data=True)

    # ==========================================
    # 3. Validation & Metric
    # ==========================================
    print("\n=== Starting Validation Phase ===")

    # Load the best model saved during training
    best_model_path = os.path.join(config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    net = model_lib.WIVENet()
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.to(device)
    net.eval()

    # Get Validation Loader
    # We reload the dataset to ensure we have the exact validation set used in metadata
    _, val_dataset = data.get_train_val_datasets(load_cached_data=True, debug=False)
    _, val_loader = data.get_dataloaders(
        None, val_dataset, batch_size=config.BATCH_SIZE
    )

    all_targets = []
    all_probs = []

    # Inference Loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = net(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_targets.extend(targets.numpy().flatten())
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute Metric
    final_metric = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")

    # Load validation metadata to link predictions with subject info
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Ensure lengths match
    if len(df_val) != len(all_probs):
        print(
            f"Warning: Metadata length ({len(df_val)}) does not match prediction length ({len(all_probs)}). Skipping detailed analysis."
        )
    else:
        df_val["pred"] = all_probs
        df_val["target"] = all_targets
        df_val["error"] = np.abs(df_val["target"] - df_val["pred"])

        # Helper to count files in a directory
        def count_files(rel_path):
            full_path = os.path.join(config.INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                return len([f for f in os.listdir(full_path) if f.endswith(".dcm")])
            return 0

        # Calculate correlations between error and file counts (proxy for volume/info quantity)
        modalities = ["flair", "t1w", "t1wce", "t2w"]
        print("Correlation between Error Magnitude and Input Features (Slice Counts):")

        for mod in modalities:
            col_name = f"{mod}_path"
            if col_name in df_val.columns:
                # Count files for this modality
                counts = df_val[col_name].apply(count_files)

                # Calculate correlation if variance exists
                if counts.std() > 0:
                    corr, _ = pearsonr(df_val["error"], counts)
                    print(f"  {mod.upper()} Slice Count: {corr:.6f}")
                else:
                    print(f"  {mod.upper()} Slice Count: NaN (No variance)")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = data.get_test_dataset(load_cached_data=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE == "cuda" else False,
        )

        test_ids = []
        test_probs = []

        # Inference on Test Set
        with torch.no_grad():
            for inputs, ids in test_loader:
                inputs = inputs.to(device)

                outputs = net(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                test_ids.extend(ids.numpy().flatten())
                test_probs.extend(probs)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})

        # Save
        submission_path = config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
