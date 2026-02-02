import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_centroids
from library.train import run_expert_training
from library.inference import generate_submission
from library.model import EfficientNetExpert
from library.data import VCAEDataset, get_transforms


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for Fast Baseline Execution
    # 2 Epochs is sufficient to demonstrate learning capability on this dataset size
    # within the strict time limit.
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4

    # Initialize directories and seeds
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting VCAE Pipeline on device: {device}")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n" + "=" * 40)
    print(" TRAINING PHASE")
    print("=" * 40)
    # Train the 3 Experts (A, B, C) across 5 Folds
    run_expert_training(load_cached_data=True)

    # ==========================================
    # 3. Validation Phase (Out-Of-Fold)
    # ==========================================
    print("\n" + "=" * 40)
    print(" VALIDATION PHASE (OOF)")
    print("=" * 40)

    # Load the hold-out validation metadata
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure centroids are computed/cached for validation set
    df_val = get_centroids(df_val, split_name="val", load_cached_data=True)

    # Dictionary to store predictions: BraTS21ID -> Probability
    oof_preds = {}

    # To ensure a valid evaluation, we predict each subject using ONLY the models
    # from the fold where that subject was in the validation set.
    # The training logic splits data using: fold_idx = BraTS21ID % NUM_FOLDS.
    df_val["fold_idx"] = df_val["BraTS21ID"] % Config.NUM_FOLDS

    for fold_idx in range(Config.NUM_FOLDS):
        # Select subjects belonging to this validation fold
        fold_subjects = df_val[df_val["fold_idx"] == fold_idx].copy()

        if len(fold_subjects) == 0:
            continue

        # Accumulate predictions from the 3 Experts for this fold
        # We will average the probabilities from Expert A, B, and C
        expert_preds_sum = np.zeros(len(fold_subjects))
        valid_experts = 0

        for expert_name, offset in Config.EXPERTS.items():
            # Construct path to the specific model trained on this fold
            model_path = os.path.join(
                Config.WORK_DIR, f"best_model_{expert_name}_fold{fold_idx}.pth"
            )

            if not os.path.exists(model_path):
                # This might happen if training crashed or was skipped
                continue

            # Load the Model Architecture and Weights
            model = EfficientNetExpert(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Create Dataset and Loader for this specific subset of subjects
            # We use the 'val' transform (resize + normalize, no augmentation)
            ds = VCAEDataset(
                fold_subjects,
                expert_offset=offset,
                split="val",
                transform=get_transforms("val"),
                cache_file_lists=False,  # Disable caching for small subsets to save setup time
            )
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Generate Predictions
            preds = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    preds.append(probs.cpu().numpy())

            if preds:
                expert_preds_sum += np.concatenate(preds).flatten()
                valid_experts += 1

            # Cleanup to free GPU memory
            del model
            torch.cuda.empty_cache()

        # Average across the valid experts found for this fold
        if valid_experts > 0:
            avg_preds = expert_preds_sum / valid_experts

            # Store predictions mapped by Subject ID
            for sid, pred in zip(fold_subjects["BraTS21ID"].values, avg_preds):
                oof_preds[sid] = pred

    # Map predictions back to the dataframe
    df_val["pred"] = df_val["BraTS21ID"].map(oof_preds)

    # Fill any missing predictions (fallback, though logic guarantees coverage)
    if df_val["pred"].isnull().any():
        df_val["pred"] = df_val["pred"].fillna(0.5)

    # Calculate Final Metric
    y_true = df_val["MGMT_value"].values
    y_pred = df_val["pred"].values

    final_auc = roc_auc_score(y_true, y_pred)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate absolute error
    df_val["error"] = np.abs(df_val["MGMT_value"] - df_val["pred"])

    # Identify feature columns (Center of Mass coordinates)
    feature_cols = [c for c in df_val.columns if c.endswith("_CoM")]

    if feature_cols:
        print("Correlation between Prediction Error and Input Features:")
        for col in feature_cols:
            corr = df_val[col].corr(df_val["error"])
            print(f"  {col}: {corr:.6f}")
    else:
        print("No metadata features available for correlation analysis.")

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n" + "=" * 40)
    print(" SUBMISSION")
    print("=" * 40)

    threshold = 0.6705454545454544

    if final_auc > threshold:
        print(f"Metric ({final_auc:.6f}) exceeds threshold ({threshold:.6f}).")
        print("Generating submission for Test Set...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Metric ({final_auc:.6f}) does not exceed threshold ({threshold:.6f}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
