import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import pearsonr
import importlib

# Force reload of libraries to pick up changes in persistent environment
# Cite debug_lesson_2: Force Module Reloads in Persistent Runtimes
import library.config
import library.utils
import library.data
import library.models
import library.engine

importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.models)
importlib.reload(library.engine)

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import (
    get_fold_dataloaders,
    get_test_dataloader,
    BirdDataset,
    get_transforms,
)
from library.models import BirdClassifier
from library.engine import run_fold


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Prepare Data for Cross-Validation & OOF Collection
    # -------------------------------------------------------------------------
    # We need to replicate the exact data splitting logic used in library.data
    # to correctly map predictions back to the validation set for the global metric.

    df_train_part = pd.read_csv(Config.TRAIN_CSV)
    df_val_part = pd.read_csv(Config.VAL_CSV)
    df_dev = pd.concat([df_train_part, df_val_part], ignore_index=True)

    # Replicate the shuffling done in get_fold_dataloaders
    df_dev = df_dev.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    # Prepare OOF (Out-Of-Fold) prediction arrays
    label_cols = [c for c in df_dev.columns if c.startswith("species_")]
    y_dev = df_dev[label_cols].values
    X_dev = df_dev["rec_id"].values.reshape(-1, 1)

    # Array to store the aggregated ensemble predictions for every sample in dev set
    oof_preds = np.zeros_like(y_dev, dtype=np.float32)

    # Replicate the Iterative Stratified Split
    k_fold = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)
    splits = list(k_fold.split(X_dev, y_dev))

    # Define the heterogeneous ensemble
    model_names = ["resnet18", "efficientnet_b0", "densenet121"]

    # -------------------------------------------------------------------------
    # 3. Training & Inference Loop (5 Folds x 3 Models)
    # -------------------------------------------------------------------------
    print(f"Starting Training: {Config.NUM_FOLDS} Folds x {len(model_names)} Models")

    for fold_idx in range(Config.NUM_FOLDS):
        train_indices, val_indices = splits[fold_idx]

        # We will accumulate predictions from all 3 models for this fold here
        fold_ensemble_preds = np.zeros(
            (len(val_indices), Config.NUM_SPECIES), dtype=np.float32
        )

        for model_name in model_names:
            print(f"\n--- Processing Fold {fold_idx} | Model: {model_name} ---")

            # A. Train the model
            # run_fold handles the training loop, early stopping, and saving the best checkpoint
            run_fold(fold_idx, model_name)

            # B. Inference on Validation Set
            # Load the best checkpoint we just saved
            model_path = os.path.join(
                Config.WORKING_DIR, f"model_{model_name}_fold_{fold_idx}.pth"
            )
            model = BirdClassifier(model_name, Config.NUM_SPECIES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Create a Validation DataLoader specifically for this fold's validation indices
            # We construct it manually to ensure we predict on the exact subset corresponding to val_indices
            resolution = Config.MODEL_SPECS[model_name]["resolution"]
            df_val_fold = df_dev.iloc[val_indices].reset_index(drop=True)

            val_dataset = BirdDataset(
                df_val_fold,
                phase="val",
                resolution=resolution,
                transform=get_transforms("val", resolution),
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Generate Predictions
            preds_list = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    preds_list.append(probs.cpu().numpy())

            model_preds = np.concatenate(preds_list, axis=0)

            # Add to fold ensemble
            fold_ensemble_preds += model_preds

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()

        # Average predictions across the 3 architectures
        fold_ensemble_preds /= len(model_names)

        # Store in the global OOF array
        oof_preds[val_indices] = fold_ensemble_preds

    # -------------------------------------------------------------------------
    # 4. Global Validation Metric
    # -------------------------------------------------------------------------
    # Calculate ROC AUC on the full development set using OOF predictions
    final_metric = calculate_roc_auc(y_dev, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Mean Absolute Error per sample (averaged across all 19 species)
    sample_errors = np.mean(np.abs(y_dev - oof_preds), axis=1)

    # Load supplemental tabular features for correlation analysis
    hist_path = os.path.join(
        Config.INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )

    if os.path.exists(hist_path):
        # The file has a header line "rec_id,[histogram...]" which has fewer commas than data lines
        # We skip the header and read data, then manually assign column names
        try:
            # Check if header exists
            with open(hist_path, "r") as f:
                first_line = f.readline()

            skip_rows = 1 if "rec_id" in first_line else 0

            # Read data
            df_feats = pd.read_csv(hist_path, skiprows=skip_rows, header=None)

            # Rename first column to rec_id
            df_feats.rename(columns={0: "rec_id"}, inplace=True)

            # Rename feature columns
            feature_cols = [f"feat_{i}" for i in range(df_feats.shape[1] - 1)]
            df_feats.columns = ["rec_id"] + feature_cols

            # Merge with error data
            df_analysis = df_dev[["rec_id"]].copy()
            df_analysis["error"] = sample_errors
            df_analysis = df_analysis.merge(df_feats, on="rec_id", how="inner")

            # Compute correlations
            correlations = {}
            for col in feature_cols:
                if df_analysis[col].std() > 1e-6:  # Skip constant columns
                    corr, _ = pearsonr(df_analysis["error"], df_analysis[col])
                    correlations[col] = corr

            # Sort and print top correlations
            sorted_corr = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )
            print("Top 5 Input Features correlated with Prediction Error:")
            for name, val in sorted_corr[:5]:
                print(f"  {name}: {val:.4f}")

        except Exception as e:
            print(f"Error during failure analysis: {e}")
    else:
        print("Supplemental feature file not found. Skipping failure analysis.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9129501920716607

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Metadata
        df_test = pd.read_csv(Config.TEST_CSV)
        test_rec_ids = df_test["rec_id"].values
        num_test_samples = len(df_test)

        # Initialize accumulator for ensemble predictions
        ensemble_test_preds = np.zeros(
            (num_test_samples, Config.NUM_SPECIES), dtype=np.float32
        )
        total_models = 0

        # Iterate over all trained models (5 Folds * 3 Architectures)
        for fold_idx in range(Config.NUM_FOLDS):
            for model_name in model_names:
                model_path = os.path.join(
                    Config.WORKING_DIR, f"model_{model_name}_fold_{fold_idx}.pth"
                )

                # Load Model
                model = BirdClassifier(model_name, Config.NUM_SPECIES, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                # Get Test Loader (returns img, label, rec_id)
                test_loader = get_test_dataloader(model_name)

                # Predict
                preds_list = []
                with torch.no_grad():
                    for images, _, _ in test_loader:
                        images = images.to(device)
                        outputs = model(images)
                        probs = torch.sigmoid(outputs)
                        preds_list.append(probs.cpu().numpy())

                ensemble_test_preds += np.concatenate(preds_list, axis=0)
                total_models += 1

                del model
                torch.cuda.empty_cache()

        # Average predictions
        ensemble_test_preds /= total_models

        # Format Submission: Id,Probability
        # Id is constructed as rec_id * 100 + species_id
        submission_rows = []
        for i, rec_id in enumerate(test_rec_ids):
            probs = ensemble_test_preds[i]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
