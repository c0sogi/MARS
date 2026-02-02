import os
import sys
import gc
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataframes, get_dataloader
from library.pipeline import train_model_fold, generate_calibrated_submission
from library.engine import predict


def main():
    # 1. Setup & Configuration
    # Enable debug mode for fast execution (reduces folds to 2 and epochs to 1 per phase)
    Config.setup(debug=True)
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Folds: {Config.N_FOLDS}")
    print(f"  Models: {list(Config.MODELS.keys())}")

    # 2. Data Loading
    # Load metadata with caching
    train_df_part, val_df_part, test_df = get_dataframes(load_cached_data=True)
    full_train_df = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    # SUBSAMPLE for Fast Baseline Requirement
    # Limit to 1500 samples to ensure execution within time limits
    # Stratified split to maintain class balance
    full_train_df, _ = train_test_split(
        full_train_df,
        train_size=1500,
        stratify=full_train_df["label"],
        random_state=Config.SEED,
    )
    full_train_df = full_train_df.reset_index(drop=True)
    print(f"Training on subset of {len(full_train_df)} samples.")

    # 3. Pipeline Execution
    # Initialize storage compatible with library functions
    oof_data = {}
    test_predictions = {}
    test_ids = None

    # Track indices for Failure Analysis mapping
    fold_indices_map = {}  # model -> list of fold validation indices

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for model_name in Config.MODELS:
        print(f"\n{'='*20} Processing Model: {model_name} {'='*20}")

        oof_data[model_name] = {"y_true": [], "y_pred": []}
        test_predictions[model_name] = []
        fold_indices_map[model_name] = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["label"])
        ):
            print(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            # Store indices for later mapping
            fold_indices_map[model_name].append(val_idx)

            # Train Model
            oof_preds, oof_targets, model = train_model_fold(
                model_name, fold, fold_train_df, fold_val_df, device
            )

            # Store OOF Predictions
            oof_data[model_name]["y_pred"].append(oof_preds)
            oof_data[model_name]["y_true"].append(oof_targets)

            # Inference on Test Set
            final_img_size = Config.MODELS[model_name]["phases"][-1]["img_size"]
            test_loader = get_dataloader(
                test_df,
                final_img_size,
                Config.MODELS[model_name]["batch_size"],
                mode="test",
            )

            fold_test_preds, fold_test_ids = predict(
                model, test_loader, device, use_tta=Config.USE_TTA
            )

            test_predictions[model_name].append(fold_test_preds)

            if test_ids is None:
                test_ids = fold_test_ids

            # Cleanup to save memory
            del model, oof_preds, oof_targets, fold_test_preds
            torch.cuda.empty_cache()
            gc.collect()

    # 4. Metric Calculation (Ensemble OOF)
    print("\nCalculating Final Validation Metric...")

    # Reconstruct OOF dataframe to handle ensemble averaging correctly
    ensemble_oof_preds = np.zeros(len(full_train_df))
    model_count = 0

    for model_name in Config.MODELS:
        # Flatten predictions and indices for this model
        all_preds = np.concatenate(oof_data[model_name]["y_pred"]).flatten()
        all_indices = np.concatenate(fold_indices_map[model_name])
        all_targets = np.concatenate(oof_data[model_name]["y_true"]).flatten()

        # Check model quality
        loss = log_loss(all_targets, all_preds, labels=[0, 1])
        print(f"Model {model_name} OOF Log Loss: {loss:.6f}")

        if loss < Config.OOF_THRESHOLD:
            # Accumulate predictions at the correct indices
            ensemble_oof_preds[all_indices] += all_preds
            model_count += 1
        else:
            print(
                f"Model {model_name} excluded from ensemble (Loss > {Config.OOF_THRESHOLD})"
            )

    if model_count > 0:
        ensemble_oof_preds /= model_count
    else:
        # Fallback if all models fail
        print("Warning: No models passed quality check. Using random baseline.")
        ensemble_oof_preds[:] = 0.5

    final_metric = log_loss(full_train_df["label"], ensemble_oof_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude
    errors = np.abs(full_train_df["label"] - ensemble_oof_preds)

    # Extract features for correlation
    feature_data = []

    print("Extracting image features for correlation analysis...")
    for idx, row in full_train_df.iterrows():
        fpath = os.path.join(Config.INPUT_DIR, row["filepath"])
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            try:
                # Fast header read for dimensions
                img = cv2.imread(fpath)
                if img is not None:
                    h, w, _ = img.shape
                    ar = w / h if h > 0 else 0
                else:
                    h, w, ar = 0, 0, 0
            except:
                h, w, ar = 0, 0, 0

            feature_data.append(
                {
                    "file_size": size,
                    "width": w,
                    "height": h,
                    "aspect_ratio": ar,
                    "error": errors[idx],
                }
            )

    if feature_data:
        analysis_df = pd.DataFrame(feature_data)
        # Calculate correlation with error
        correlations = analysis_df.corr()["error"].drop("error")
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Could not extract features for analysis.")

    # 6. Submission Generation
    # Strict threshold as per task requirements
    SUBMISSION_THRESHOLD = 0.009074434935821756

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) < Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_calibrated_submission(oof_data, test_predictions, test_ids)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) >= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
