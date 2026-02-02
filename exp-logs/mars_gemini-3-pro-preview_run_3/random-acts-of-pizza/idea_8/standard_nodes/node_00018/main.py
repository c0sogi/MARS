import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library import config
from library import utils
from library import feature_pipeline
from library import ensemble_trainer


def main():
    # 1. Setup
    utils.set_seed(config.SEED)

    with utils.Timer("Total Pipeline Execution"):

        # 2. Data Loading
        print("\n[Main] Loading Data...")
        df_train = utils.load_data("train")
        df_val = utils.load_data("val")
        df_test = utils.load_data("test")

        y_train = df_train[config.TARGET_COL].values
        y_val = df_val[config.TARGET_COL].values

        print(f"  Train shape: {df_train.shape}")
        print(f"  Val shape:   {df_val.shape}")
        print(f"  Test shape:  {df_test.shape}")

        # 3. Feature Engineering
        print("\n[Main] Running Feature Pipeline...")
        pipeline = feature_pipeline.FeaturePipeline()

        # Fit and transform training data
        # load_cached_data=True allows using pre-computed features if available
        features_train = pipeline.fit_transform(
            df_train, split_name="train", load_cached_data=True
        )

        # Transform validation and test data
        features_val = pipeline.transform(
            df_val, split_name="val", load_cached_data=True
        )

        features_test = pipeline.transform(
            df_test, split_name="test", load_cached_data=True
        )

        # 4. Model Training
        print("\n[Main] Training Tri-View Stacking Ensemble...")
        model = ensemble_trainer.TriViewStackingEnsemble()
        model.fit(features_train, y_train)

        # 5. Validation
        print("\n[Main] Validating...")
        val_preds = model.predict(features_val)
        val_auc = roc_auc_score(y_val, val_preds)

        # REQUIRED OUTPUT FORMAT
        print(f"Final Validation Metric: {val_auc}")

        # 6. Failure Analysis
        print("\n[Main] Performing Failure Analysis on Validation Set...")
        errors = np.abs(y_val - val_preds)

        # We correlate errors with numerical columns available in the raw dataframe
        # to see which original features are associated with higher error.
        # We filter for numerical columns that are not IDs or targets.
        numerical_cols = df_val.select_dtypes(include=["number"]).columns
        exclude_cols = [
            config.TARGET_COL,
            "unix_timestamp_of_request",
            "unix_timestamp_of_request_utc",
        ]
        analysis_cols = [
            c
            for c in numerical_cols
            if c not in exclude_cols and not c.endswith("_at_retrieval")
        ]

        correlations = []
        for col in analysis_cols:
            # Handle potential NaNs in raw data by filling with median for correlation check
            col_data = df_val[col].fillna(df_val[col].median())
            # Ensure constant columns don't crash pearsonr
            if col_data.std() > 0:
                corr, _ = pearsonr(errors, col_data)
                correlations.append((col, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("  Top 5 Features correlated with Error Magnitude:")
        for name, corr in correlations[:5]:
            print(f"    - {name}: {corr:.4f}")

        # 7. Submission
        threshold = 0.6913548345419015
        if val_auc > threshold:
            print(
                f"\n[Main] Validation AUC ({val_auc}) > Threshold ({threshold}). Generating Submission..."
            )

            test_preds = model.predict(features_test)

            submission_df = pd.DataFrame(
                {
                    "request_id": df_test[config.ID_COL],
                    "requester_received_pizza": test_preds,
                }
            )

            # Ensure output directory exists (handled in config but good to be safe)
            os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

            submission_df.to_csv(config.SUBMISSION_PATH, index=False)
            print(f"  Submission saved to {config.SUBMISSION_PATH}")
            print(f"  Submission head:\n{submission_df.head()}")
        else:
            print(
                f"\n[Main] Validation AUC ({val_auc}) <= Threshold ({threshold}). Skipping Submission."
            )


if __name__ == "__main__":
    main()
