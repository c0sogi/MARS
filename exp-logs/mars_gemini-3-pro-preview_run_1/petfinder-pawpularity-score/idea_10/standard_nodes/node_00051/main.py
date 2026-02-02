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

    # Ensure targets are available
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # 3. Feature Extraction
    # We extract features for Train, Val, and Test using all 3 backbones.
    extractor = BackboneExtractor()
    backbones = ["siglip", "dinov2", "convnext"]

    train_data_map = {}
    val_data_map = {}
    test_data_map = {}

    print("\n=== Phase 1: Feature Extraction ===")
    for bb in backbones:
        # Extract (or load cached) features
        train_data_map[bb] = extractor.extract(
            train_df, bb, "train", load_cached_data=True
        )
        val_data_map[bb] = extractor.extract(val_df, bb, "val", load_cached_data=True)
        test_data_map[bb] = extractor.extract(
            test_df, bb, "test", load_cached_data=True
        )

    # 4. Level-0 Expert Training
    # We train 3 experts (Ridge, SVR, ET) for each of the 3 backbones.
    # To optimize runtime, we combine Val and Test sets for the inference pass of the experts.

    l0_trainer = Level0Trainer()
    models = ["ridge", "svr", "et"]

    oof_dict = {}  # Stores OOF predictions for Train set
    val_pred_dict = {}  # Stores predictions for Val set
    test_pred_dict = {}  # Stores predictions for Test set

    n_val = len(val_df)

    print("\n=== Phase 2: Level-0 Expert Training ===")
    for bb in backbones:
        # Prepare Combined Inference Data (Val + Test)
        # We concatenate features and metadata to pass as 'test_data' to the trainer
        combined_features = np.concatenate(
            [val_data_map[bb]["features"], test_data_map[bb]["features"]], axis=0
        )
        combined_meta = np.concatenate(
            [val_data_map[bb]["meta"], test_data_map[bb]["meta"]], axis=0
        )

        combined_inference_data = {"features": combined_features, "meta": combined_meta}

        for model_type in models:
            expert_key = f"{bb}_{model_type}"
            print(f"Processing Expert: {expert_key}")

            # Run Expert: Trains on Train (CV), Predicts on Combined Inference Data
            oof_preds, combined_preds = l0_trainer.run_expert(
                backbone_name=bb,
                model_type=model_type,
                train_data=train_data_map[bb],
                test_data=combined_inference_data,
                load_cached_data=True,
            )

            # Store OOF
            oof_dict[expert_key] = oof_preds

            # Split Combined Predictions back into Val and Test
            val_pred_dict[expert_key] = combined_preds[:n_val]
            test_pred_dict[expert_key] = combined_preds[n_val:]

    # 5. Level-1 Meta-Learner & Validation
    print("\n=== Phase 3: Level-1 Meta-Learning & Validation ===")
    l1_learner = Level1MetaLearner()

    # Step A: Validate
    # We use the Meta-Learner to train on the OOF predictions and predict on the Validation set predictions.
    # Note: train_and_predict saves a submission file. We pass dummy IDs for now (or Val IDs),
    # and we will clean it up or overwrite it later.
    print("Evaluating on Hold-out Validation Set...")

    # We pass val_pred_dict as the 'test' input to get predictions on the validation set
    final_val_preds = l1_learner.train_and_predict(
        oof_dict=oof_dict,
        test_pred_dict=val_pred_dict,
        y_true=y_train,
        test_ids=val_df[Config.ID_COL].values,
    )

    # Calculate RMSE
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
