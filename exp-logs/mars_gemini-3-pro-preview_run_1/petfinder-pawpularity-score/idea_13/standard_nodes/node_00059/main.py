import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, rmse_score, load_array
from library.feature_extraction import run_feature_extraction
from library.level0_experts import train_level0_experts
from library.level1_meta import train_meta_learner


def main():
    # =========================================================================
    # 1. Setup
    # =========================================================================
    set_seed(Config.SEED)
    print("Starting End-to-End Pipeline...")

    # =========================================================================
    # 2. Pipeline Execution
    # =========================================================================
    # Step 1: Feature Extraction
    # Extracts features from images using SigLIP, DINOv2, and ConvNeXt
    print("\n=== Step 1: Feature Extraction ===")
    run_feature_extraction(debug=False, load_cached_data=True)

    # Step 2: Level-0 Experts
    # Trains 12 experts (3 backbones * 4 models) using Stratified CV
    print("\n=== Step 2: Level-0 Experts Training ===")
    train_level0_experts(debug=False, load_cached_data=True)

    # Step 3: Level-1 Meta-Learner
    # Trains the Interaction-Aware Meta-Learner and generates submission.csv
    print("\n=== Step 3: Level-1 Meta-Learner Training ===")
    train_meta_learner(debug=False, load_cached_data=True)

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\n=== Step 4: Validation Assessment ===")

    # Load hold-out validation metadata
    val_df = pd.read_csv(Config.VAL_METADATA)
    val_ids_set = set(val_df[Config.ID_COL].values)

    # Load Level-1 OOF Data and Merged IDs
    # We need to map the OOF predictions back to the original Validation IDs
    try:
        X_oof = load_array("level1_simple_X_oof.npy")
        y_oof = load_array("level1_simple_y_oof.npy")

        # Load merged IDs and Meta from one of the backbones (they are all aligned)
        ref_backbone = list(Config.BACKBONES.keys())[0]
        merged_ids = load_array(f"{ref_backbone}_merged_ids.npy")
        merged_meta = load_array(f"{ref_backbone}_merged_meta.npy")
    except FileNotFoundError as e:
        print(f"Critical Error during validation: {e}")
        return

    # Reproduce OOF Predictions via CV
    # We re-run the exact same CV split to get predictions for the OOF set
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    oof_preds = np.zeros(len(y_oof))

    for train_idx, val_idx in kf.split(X_oof, y_oof):
        X_train, X_val = X_oof[train_idx], X_oof[val_idx]
        y_train = y_oof[train_idx]

        model = BayesianRidge(**Config.META_MODEL_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Clip predictions to valid range
        preds = np.clip(preds, 1.0, 100.0)
        oof_preds[val_idx] = preds

    # Filter for Hold-out Validation Set
    # Identify indices in the merged set that correspond to the validation set
    val_indices = [i for i, uid in enumerate(merged_ids) if uid in val_ids_set]

    if len(val_indices) == 0:
        print("Error: Could not match validation IDs to OOF predictions.")
        return

    y_true_val = y_oof[val_indices]
    y_pred_val = oof_preds[val_indices]

    # Calculate Metric
    final_metric = rmse_score(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Step 5: Failure Analysis on Validation Set ===")

    # Calculate absolute errors
    errors = np.abs(y_true_val - y_pred_val)

    # Extract Metadata from Merged Meta Array
    # Since X_oof no longer contains metadata (Simple Stacking), we use the loaded merged_meta
    metadata_val = merged_meta[val_indices]

    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<10}")
    print("-" * 48)

    for i, feature_name in enumerate(Config.META_FEATURES):
        feat_values = metadata_val[:, i]
        # Calculate Pearson correlation between binary feature and continuous error
        corr, p_val = pearsonr(feat_values, errors)
        print(f"{feature_name:<20} | {corr:<12.4f} | {p_val:<10.4g}")

    # =========================================================================
    # 5. Submission Logic
    # =========================================================================
    print("\n=== Step 6: Submission Verification ===")
    threshold = 16.95541414729265

    if final_metric < threshold:
        print(
            f"Metric ({final_metric}) < Threshold ({threshold}). Submission accepted."
        )
        print(f"Submission file available at: {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric ({final_metric}) >= Threshold ({threshold}). Submission rejected."
        )
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
            print("Submission file deleted.")


if __name__ == "__main__":
    main()
