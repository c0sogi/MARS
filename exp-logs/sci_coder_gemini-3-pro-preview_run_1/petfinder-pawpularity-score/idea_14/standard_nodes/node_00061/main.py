import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge

# Import library modules
from library.config import Config
from library.utils import seed_everything, rmse_score, save_to_cache
from library.data import load_metadata, PetDataset, get_processor
from library.extractors import process_and_cache_features
from library.ensemble import Level0Trainer

# =============================================================================
# 1. Configuration & Setup
# =============================================================================

# Fast Baseline Overrides to meet time constraints
Config.DEBUG = False
Config.N_FOLDS = 3
Config.ET_PARAMS["n_estimators"] = 50
Config.LGBM_PARAMS["n_estimators"] = 100
Config.LGBM_PARAMS["early_stopping_rounds"] = 20
# Adjusted for raw targets (0-100 range)
Config.SVR_GRID = {
    "kernel": ["rbf"],
    "C": [1.0, 10.0],
    "epsilon": [1.0, 5.0],
    "gamma": ["scale"],
}
Config.RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 500.0]

# Ensure output directories exist
os.makedirs(Config.IDEA_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

seed_everything(Config.SEED)


def main():
    print("Starting Fast Baseline Pipeline...")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading Metadata...")
    # Load separate train and val sets
    train_df, val_df, test_df = load_metadata(merge_train_val=False, debug=False)

    # Subsample for speed (Time limit constraint)
    # We keep full Test set for submission.
    # We subsample Train and Val to ensure feature extraction and training fit in time.

    N_TRAIN = 2200
    N_VAL = 500

    if len(train_df) > N_TRAIN:
        train_df = train_df.sample(n=N_TRAIN, random_state=Config.SEED).reset_index(
            drop=True
        )
    if len(val_df) > N_VAL:
        val_df = val_df.sample(n=N_VAL, random_state=Config.SEED).reset_index(drop=True)

    print(f"Data Shapes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # =========================================================================
    # 3. Feature Extraction
    # =========================================================================
    # Define backbones with optimized batch sizes
    backbones = [
        ("SigLIP", Config.MODEL_SIGLIP, 64, Config.CACHE_FEATURES_SIGLIP),
        ("DINOv2", Config.MODEL_DINOV2, 32, Config.CACHE_FEATURES_DINOV2),
        ("ConvNeXt", Config.MODEL_CONVNEXTV2, 64, Config.CACHE_FEATURES_CONVNEXT),
    ]

    feature_data_train = {}
    feature_data_val = {}
    feature_data_test = {}

    train_targets = None
    val_targets = None
    test_ids = None

    for name, model_path, batch_size, cache_base_path in backbones:
        print(f"\nProcessing {name}...")
        processor = get_processor(model_path)

        # Helper to process a dataframe
        def process_subset(df, subset_name, include_target=True):
            # Unique cache path for this run's subset to avoid conflicts
            base, ext = os.path.splitext(cache_base_path)
            cache_paths = {
                "embeddings": f"{base}_{subset_name}_fast{ext}",
                "ids": f"{base}_{subset_name}_ids_fast{ext}",
                "metadata": f"{base}_{subset_name}_meta_fast{ext}",
                "targets": (
                    f"{base}_{subset_name}_tgt_fast{ext}" if include_target else None
                ),
            }

            ds = PetDataset(
                df, processor, return_flipped=True, include_target=include_target
            )
            return process_and_cache_features(
                ds, model_path, batch_size, cache_paths, load_cached_data=True
            )

        # Process Train
        res_train = process_subset(train_df, "train", include_target=True)
        feature_data_train[name] = {
            "embeddings": res_train["embeddings"],
            "metadata": res_train["metadata"],
        }
        if train_targets is None:
            train_targets = res_train["targets"]

        # Process Val
        res_val = process_subset(val_df, "val", include_target=True)
        feature_data_val[name] = {
            "embeddings": res_val["embeddings"],
            "metadata": res_val["metadata"],
        }
        if val_targets is None:
            val_targets = res_val["targets"]

        # Process Test
        res_test = process_subset(test_df, "test", include_target=False)
        feature_data_test[name] = {
            "embeddings": res_test["embeddings"],
            "metadata": res_test["metadata"],
        }
        if test_ids is None:
            test_ids = res_test["ids"]

        # Cleanup
        del processor, res_train, res_val, res_test
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # 4. Level 0 Training
    # =========================================================================
    print("\nTraining Level 0 Experts...")

    # Combine Val and Test into one 'combined_test' dictionary to pass to run_cv.
    # This allows us to get predictions for both sets in one go.
    combined_test_data = {}
    n_val = len(val_df)

    for name in feature_data_train.keys():
        combined_test_data[name] = {
            "embeddings": np.vstack(
                [
                    feature_data_val[name]["embeddings"],
                    feature_data_test[name]["embeddings"],
                ]
            ),
            "metadata": np.vstack(
                [
                    feature_data_val[name]["metadata"],
                    feature_data_test[name]["metadata"],
                ]
            ),
        }

    l0_trainer = Level0Trainer(n_folds=Config.N_FOLDS, seed=Config.SEED)

    # Run CV on Train. Pass combined Val+Test as the 'test' set.
    # We disable cache loading for L0 to ensure we use the current subsampled data.
    oof_train, combined_preds = l0_trainer.run_cv(
        feature_data_train, train_targets, combined_test_data, load_cached=False
    )

    # Split combined predictions back to Val and Test
    l0_val_preds = combined_preds[:n_val]
    l0_test_preds = combined_preds[n_val:]

    # =========================================================================
    # 5. Level 1 Training (Meta-Learner)
    # =========================================================================
    print("\nTraining Level 1 Meta-Learner...")

    meta_model = BayesianRidge(**Config.META_MODEL_PARAMS)

    # Fit on OOF predictions from Train
    meta_model.fit(oof_train, train_targets)

    # Predict on Val
    val_final_preds = meta_model.predict(l0_val_preds)

    # Predict on Test
    test_final_preds = meta_model.predict(l0_test_preds)

    # =========================================================================
    # 6. Validation & Failure Analysis
    # =========================================================================
    print("\n--- Validation Results ---")

    # Calculate RMSE
    val_rmse = rmse_score(val_targets, val_final_preds)
    print(f"Final Validation Metric: {val_rmse}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    residuals = np.abs(val_targets - val_final_preds)

    # Correlate residuals with metadata features
    meta_cols = [
        "Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
    # Handle column name variation if present
    if "Subject Focus" in val_df.columns:
        val_df.rename(columns={"Subject Focus": "Focus"}, inplace=True)

    correlations = {}
    for col in meta_cols:
        if col in val_df.columns:
            # Point-biserial correlation
            corr = val_df[col].corr(pd.Series(residuals, index=val_df.index))
            correlations[col] = corr

    print("Correlation of Error Magnitude with Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"{feat}: {corr:.4f}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    THRESHOLD = 16.95541414729265

    if val_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({val_rmse:.4f}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        submission = pd.DataFrame({"Id": test_ids, "Pawpularity": test_final_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation RMSE ({val_rmse:.4f}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
