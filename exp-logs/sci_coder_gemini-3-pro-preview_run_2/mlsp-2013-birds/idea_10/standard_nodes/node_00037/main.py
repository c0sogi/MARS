import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Add current directory to sys.path to ensure library imports work
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data import get_loaders, get_test_loader
from library.models import get_model
from library.training import train_fold


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Monkey-patch Config to ensure execution within time limits
    # 15 models * 400 steps is manageable within 2 hours on A100
    Config.MAX_STEPS = 400
    Config.VAL_CHECK_INTERVAL = 100

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    device = torch.device(Config.DEVICE)
    architectures = Config.ARCHITECTURES
    n_folds = Config.N_FOLDS

    print("Starting Tri-Architecture Heterogeneous Ensemble Workflow...")

    # 2. Training Phase
    # Train 3 architectures across 5 folds = 15 models
    for arch in architectures:
        for fold in range(n_folds):
            print(f"\nTraining {arch} - Fold {fold}")
            # train_fold saves the best model to disk automatically
            train_fold(fold, arch)

    # 3. Validation Phase (OOF Inference)
    print("\nGenerating Out-Of-Fold (OOF) Predictions...")

    oof_rec_ids = []
    oof_preds = []
    oof_targets = []

    # Iterate over folds to generate predictions on the validation part of each fold
    for fold in range(n_folds):
        _, val_loader = get_loaders(fold, batch_size=Config.BATCH_SIZE * 2, debug=False)

        # Store predictions for this fold from all architectures
        fold_preds_accum = None
        fold_targets = []
        fold_ids = []

        # Collect ground truth and IDs once per fold
        with torch.no_grad():
            for _, labels, rec_ids in val_loader:
                fold_targets.append(labels.numpy())
                fold_ids.append(rec_ids.numpy())

        fold_targets = np.concatenate(fold_targets)
        fold_ids = np.concatenate(fold_ids)

        # Ensemble predictions from all architectures for this fold
        for arch in architectures:
            model_path = os.path.join(
                Config.WORKING_DIR, f"model_{arch}_fold_{fold}.pth"
            )
            model = get_model(arch, num_classes=Config.NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            arch_preds = []
            with torch.no_grad():
                for inputs, _, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    arch_preds.append(probs.cpu().numpy())

            arch_preds = np.concatenate(arch_preds)

            if fold_preds_accum is None:
                fold_preds_accum = arch_preds
            else:
                fold_preds_accum += arch_preds

        # Average predictions
        fold_preds_avg = fold_preds_accum / len(architectures)

        oof_rec_ids.append(fold_ids)
        oof_preds.append(fold_preds_avg)
        oof_targets.append(fold_targets)

    # Concatenate all OOF results
    oof_rec_ids = np.concatenate(oof_rec_ids)
    oof_preds = np.concatenate(oof_preds)
    oof_targets = np.concatenate(oof_targets)

    # 4. Metric Calculation
    # Calculate Macro-Averaged ROC AUC on the full OOF set
    final_metric = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    # error shape: (N_samples,)
    per_sample_error = np.mean(np.abs(oof_targets - oof_preds), axis=1)

    # Load tabular features
    hist_path = os.path.join(
        Config.INPUT_ROOT, "supplemental_data", "histogram_of_segments.txt"
    )
    if os.path.exists(hist_path):
        # Read tabular data
        # The file format is rec_id,feat_0,feat_1,...
        # We need to handle the header line: "rec_id,[histogram of segment features]"
        # and then data lines.
        try:
            with open(hist_path, "r") as f:
                header = f.readline()

            # If header looks like the description, we can just read csv skipping header or using it if standard
            # The description says: rec_id,[histogram...]
            # Let's read with pandas, assuming standard CSV structure after header cleaning or just use 'header=0'
            # Based on file inspection in prompt: "rec_id,[histogram of segment features]" is line 1
            # "0,0.00..." is line 2.
            # So we can treat line 1 as header.
            df_features = pd.read_csv(hist_path)

            # Rename columns: first is rec_id, rest are features
            feature_cols = [f"feat_{i}" for i in range(len(df_features.columns) - 1)]
            df_features.columns = ["rec_id"] + feature_cols

            # Create DataFrame for errors
            df_error = pd.DataFrame({"rec_id": oof_rec_ids, "error": per_sample_error})

            # Merge
            df_analysis = df_error.merge(df_features, on="rec_id", how="inner")

            if not df_analysis.empty:
                # Calculate correlation
                correlations = df_analysis[feature_cols].corrwith(df_analysis["error"])

                # Get top 5 correlated features (absolute correlation)
                top_corr = correlations.abs().sort_values(ascending=False).head(5)

                print("Top 5 Features correlated with Error Magnitude:")
                for feat, corr_val in top_corr.items():
                    # Get original sign
                    sign = correlations[feat]
                    print(f"  {feat}: {sign:.4f}")
            else:
                print("Merge resulted in empty dataframe. Check rec_id matching.")
        except Exception as e:
            print(f"Failure analysis failed: {e}")
    else:
        print("Tabular feature file not found.")

    # 6. Submission Generation
    threshold = 0.9065740624675196

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        test_loader = get_test_loader(batch_size=Config.BATCH_SIZE * 2)
        test_preds_accum = None
        test_rec_ids = []

        # Collect test rec_ids once
        with torch.no_grad():
            for _, _, rec_ids in test_loader:
                test_rec_ids.append(rec_ids.numpy())
        test_rec_ids = np.concatenate(test_rec_ids)

        # Grand Ensemble Inference: 15 Models
        model_count = 0
        for arch in architectures:
            for fold in range(n_folds):
                model_path = os.path.join(
                    Config.WORKING_DIR, f"model_{arch}_fold_{fold}.pth"
                )

                model = get_model(
                    arch, num_classes=Config.NUM_CLASSES, pretrained=False
                )
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                fold_preds = []
                with torch.no_grad():
                    for inputs, _, _ in test_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        probs = torch.sigmoid(outputs)
                        fold_preds.append(probs.cpu().numpy())

                fold_preds = np.concatenate(fold_preds)

                if test_preds_accum is None:
                    test_preds_accum = fold_preds
                else:
                    test_preds_accum += fold_preds

                model_count += 1

        # Average predictions
        avg_test_preds = test_preds_accum / model_count

        # Format Submission
        # Format: Id,Probability
        # Id = rec_id * 100 + species_number
        submission_rows = []
        for i, rec_id in enumerate(test_rec_ids):
            sample_probs = avg_test_preds[i]  # Shape (19,)
            for species_idx, prob in enumerate(sample_probs):
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
