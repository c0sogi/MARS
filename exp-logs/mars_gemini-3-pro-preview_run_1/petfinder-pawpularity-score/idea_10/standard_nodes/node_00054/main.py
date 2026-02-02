import os
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.backbone_extractor import BackboneExtractor
from library.stacking_engine import Level0Trainer, Level1MetaLearner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    print("Starting Orchestration Pipeline...")

    # 2. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Cite solution_lesson_node_00033: Full-Dataset Cross-Validation
    # Combine Train and Val for Level-0 training to maximize data utility
    n_train = len(train_df)
    n_val = len(val_df)

    # Ensure targets are available
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values
    y_full = np.concatenate([y_train, y_val], axis=0)

    # 3. Feature Extraction
    # We extract features for Train, Val, and Test using all 3 backbones.
    extractor = BackboneExtractor()
    backbones = ["siglip", "dinov2", "convnext"]

    full_data_map = {}
    test_data_map = {}

    print("\n=== Phase 1: Feature Extraction ===")
    for bb in backbones:
        # Extract (or load cached) features
        train_data = extractor.extract(train_df, bb, "train", load_cached_data=True)
        val_data = extractor.extract(val_df, bb, "val", load_cached_data=True)

        # Combine Train and Val features for Level-0 training
        full_data_map[bb] = {
            "features": np.concatenate(
                [train_data["features"], val_data["features"]], axis=0
            ),
            "meta": np.concatenate([train_data["meta"], val_data["meta"]], axis=0),
            "targets": np.concatenate(
                [train_data["targets"], val_data["targets"]], axis=0
            ),
        }

        test_data_map[bb] = extractor.extract(
            test_df, bb, "test", load_cached_data=True
        )

    # 4. Level-0 Expert Training
    # We train 3 experts (Ridge, SVR, ET) for each of the 3 backbones.
    l0_trainer = Level0Trainer()
    models = ["ridge", "svr", "et"]

    oof_dict = {}  # Stores OOF predictions for Full (Train+Val) set
    test_pred_dict = {}  # Stores predictions for Test set

    print("\n=== Phase 2: Level-0 Expert Training (Full Dataset) ===")
    for bb in backbones:
        for model_type in models:
            expert_key = f"{bb}_{model_type}"
            print(f"Processing Expert: {expert_key}")

            # Run Expert: Trains on Full (CV), Predicts on Test
            # Cite solution_lesson_node_00033: Use combined dataset for training
            oof_preds, test_preds = l0_trainer.run_expert(
                backbone_name=bb,
                model_type=model_type,
                train_data=full_data_map[bb],
                test_data=test_data_map[bb],
                load_cached_data=True,
                cache_suffix="_full",  # Use distinct cache for full-dataset models
            )

            # Store OOF and Test predictions
            oof_dict[expert_key] = oof_preds
            test_pred_dict[expert_key] = test_preds

    # 5. Level-1 Meta-Learner & Validation
    print("\n=== Phase 3: Level-1 Meta-Learning & Validation ===")
    l1_learner = Level1MetaLearner()

    # Step A: Validate
    # We use the Meta-Learner to train on the OOF predictions (Full Set).
    # To get validation metrics, we ask it to return the Meta-OOF predictions
    # and we slice out the part corresponding to the validation set.
    print("Evaluating on Hold-out Validation Set (via OOF)...")

    # We pass test_ids as dummy here because we are primarily interested in the OOF return
    # The actual submission file generation happens in the final step if threshold is met
    _, meta_oof_preds = l1_learner.train_and_predict(
        oof_dict=oof_dict,
        test_pred_dict=test_pred_dict,
        y_true=y_full,
        test_ids=test_df[Config.ID_COL].values,
        return_oof=True,
    )

    # Extract predictions corresponding to the Validation set
    # Since we concatenated [Train, Val], the Val predictions are the last n_val elements
    final_val_preds = meta_oof_preds[n_train:]

    # Calculate RMSE on the Validation subset
    val_rmse = compute_rmse(y_val, final_val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_rmse}")

    # Clean up the temporary submission file created during validation
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # 6. Failure Analysis
    print("\n=== Phase 4: Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_preds)

    # Calculate correlation with metadata features
    print("Correlation between Error Magnitude and Metadata Features:")
    correlations = []

    for col in Config.METADATA_COLS:
        if col in val_df.columns:
            # Point-Biserial Correlation (since features are binary) is equivalent to Pearson here
            corr = np.corrcoef(val_df[col].values, errors)[0, 1]
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, corr in correlations:
        print(f"{col}: {corr:.4f}")

    # 7. Submission Generation
    print("\n=== Phase 5: Submission Generation ===")
    THRESHOLD = 17.07053899184464

    if val_rmse < THRESHOLD:
        print(f"Validation Metric ({val_rmse}) is better than threshold ({THRESHOLD}).")
        print("Generating final submission for Test Set...")

        # Generate predictions for the actual Test Set
        l1_learner.train_and_predict(
            oof_dict=oof_dict,
            test_pred_dict=test_pred_dict,
            y_true=y_train,
            test_ids=test_df[Config.ID_COL].values,
        )
        print("Submission generation complete.")
    else:
        print(f"Validation Metric ({val_rmse}) did not meet threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
