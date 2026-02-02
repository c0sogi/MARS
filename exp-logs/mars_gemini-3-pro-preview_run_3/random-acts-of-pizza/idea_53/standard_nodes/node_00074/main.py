import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library components
from library.config import (
    RANDOM_SEED,
    TARGET_COL,
    ID_COL,
    SUBMISSION_PATH,
    METADATA_FEATURES,
)
from library.data_loader import load_dataset
from library.feature_extraction import FeaturePipeline
from library.training_engine import HybridTrainer
from library.inference_engine import HybridPredictor


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    print("Initialized runfile.py execution...")

    # 2. Load Data
    # We load all splits. library.data_loader handles reading parquet and basic cleaning.
    print("Loading datasets...")
    df_train = load_dataset("train", load_cached_data=True)
    df_val = load_dataset("val", load_cached_data=True)
    df_test = load_dataset("test", load_cached_data=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    # 3. Initialize Feature Pipeline
    # The pipeline manages vectorizers and scalers.
    # It will be fitted inside the trainer on df_train.
    print("Initializing Feature Pipeline...")
    feature_pipeline = FeaturePipeline(load_cached_data=True)

    # 4. Train Models
    # We use debug=False to ensure we meet the performance threshold.
    # The dataset is small enough (~2.3k rows) that full training is fast.
    print("Starting Model Training...")
    trainer = HybridTrainer(debug=False)

    # Note: trainer.train calls feature_pipeline.fit_transform(df_train)
    # This fits the vectorizers/scalers and caches the train features.
    trainer.train(df_train, feature_pipeline)

    # 5. Validation Inference
    print("Performing Validation Inference...")
    predictor = HybridPredictor()

    # Predict on validation set
    # The pipeline is already fitted from the training step.
    val_preds_df = predictor.predict(df_val, feature_pipeline)

    # Align predictions with ground truth
    # Both dfs should be in the same order, but we merge on ID to be safe
    val_merged = df_val[[ID_COL, TARGET_COL]].merge(
        val_preds_df, on=ID_COL, suffixes=("_true", "_pred")
    )

    # The predictor returns the probability in a column named TARGET_COL
    y_true = val_merged[f"{TARGET_COL}_true"].values
    y_pred = val_merged[f"{TARGET_COL}_pred"].values

    # 6. Metric Calculation
    val_auc = roc_auc_score(y_true, y_pred)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    print("Correlation between Error Magnitude and Metadata Features:")
    # We analyze correlations with the numerical metadata features used in the model
    analysis_results = []
    for feature in METADATA_FEATURES:
        if feature in df_val.columns:
            # Ensure numeric and handle NaNs (though pipeline handles them, raw df might have them)
            feat_values = df_val[feature].fillna(df_val[feature].median())

            # Align indices (merge might have reordered, though unlikely)
            # We use the values corresponding to the merged dataframe rows
            # To do this correctly, we merge the feature back or rely on index if preserved.
            # Let's use the ID map.
            feat_values_aligned = df_val.set_index(ID_COL).loc[val_merged[ID_COL]][
                feature
            ]

            if len(np.unique(feat_values_aligned)) > 1:
                corr, _ = pearsonr(errors, feat_values_aligned)
                analysis_results.append((feature, corr))
                print(f"  {feature}: {corr:.4f}")
            else:
                print(f"  {feature}: N/A (Constant)")

    # 8. Submission Generation
    THRESHOLD = 0.7222984867326668

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for test set
        submission_df = predictor.predict(df_test, feature_pipeline)

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {val_auc} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
