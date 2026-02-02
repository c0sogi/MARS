import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

# Import from provided libraries
from library.config import (
    SEED,
    TARGET_COL,
    USER_ID_COL,
    ITEM_ID_COL,
    SUBMISSION_PATH,
    RANKER_FEATURES,
)
from library.data_loader import get_time_split_data, load_test_customers
from library.feature_builder import RankerDatasetBuilder
from library.ranker import LGBMRanker
from library.utils import calculate_map12, create_submission

# Setup configuration
warnings.filterwarnings("ignore")
np.random.seed(SEED)


def main():
    print("=== Starting Runfile Execution ===")

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("Loading time-split data...")
    # train_df: History (Weeks 0 to T-2)
    # val_df: Validation Target (Week T-1)
    train_df, val_df = get_time_split_data(load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Build Ranker Dataset
    # ---------------------------------------------------------
    print("Building Ranker Dataset...")
    builder = RankerDatasetBuilder()

    # Generates candidates for users in val_df using history from train_df
    # Labels are derived from val_df
    ranker_dataset = builder.build_train_set(train_df, val_df, load_cached_data=False)

    # ---------------------------------------------------------
    # 3. Train/Val Split for Ranker
    # ---------------------------------------------------------
    print("Splitting Ranker Dataset (80/20 by Customer)...")
    # We split the available labeled data to train the ranker and evaluate the pipeline
    unique_customers = ranker_dataset[USER_ID_COL].unique()

    # Use a fixed random state for reproducibility
    train_cust, val_cust = train_test_split(
        unique_customers, test_size=0.2, random_state=SEED
    )

    # Create DataFrames
    X_train = ranker_dataset[ranker_dataset[USER_ID_COL].isin(train_cust)].copy()
    X_val = ranker_dataset[ranker_dataset[USER_ID_COL].isin(val_cust)].copy()

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # ---------------------------------------------------------
    # 4. Train Ranker
    # ---------------------------------------------------------
    print("Training LightGBM Ranker...")
    ranker = LGBMRanker()
    # load_cached_data=False ensures we train on this specific split
    ranker.fit(X_train, val_df=X_val, load_cached_data=False)

    # ---------------------------------------------------------
    # 5. Validation & Metrics
    # ---------------------------------------------------------
    print("Generating predictions for validation set...")
    preds = ranker.predict(X_val)
    X_val["score"] = preds

    # Select Top 12 per customer
    print("Selecting top 12 candidates...")
    top_preds = (
        X_val.sort_values([USER_ID_COL, "score"], ascending=[True, False])
        .groupby(USER_ID_COL)
        .head(12)
    )

    # Convert to dict for metric calculation
    pred_dict = top_preds.groupby(USER_ID_COL)[ITEM_ID_COL].apply(list).to_dict()

    # Get Ground Truth (subset of val_df matching our validation customers)
    val_df_subset = val_df[val_df[USER_ID_COL].isin(val_cust)]
    truth_dict = val_df_subset.groupby(USER_ID_COL)[ITEM_ID_COL].apply(list).to_dict()

    print("Calculating MAP@12...")
    metric = calculate_map12(pred_dict, truth_dict)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    X_val["error"] = np.abs(X_val[TARGET_COL] - X_val["score"])

    # Correlate error with features
    print("Correlation between Model Error and Input Features:")

    # Identify numerical columns
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns
    exclude_cols = [TARGET_COL, "score", "error"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in feature_cols:
        if X_val[col].std() > 1e-6:
            corr = X_val["error"].corr(X_val[col])
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for feat, corr in sorted_corrs[:10]:
        print(f"  {feat}: {corr:.6f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.0096263154326182

    if metric > THRESHOLD:
        print(
            f"\nMetric ({metric}) exceeds threshold ({THRESHOLD}). Proceeding to submission..."
        )

        # 1. Prepare Data
        print("Loading test customers...")
        test_customers_df = load_test_customers()
        test_customer_ids = test_customers_df[USER_ID_COL].unique()

        print("Combining history for retrieval retraining...")
        full_history = pd.concat([train_df, val_df], ignore_index=True)

        # 2. Build Test Candidates
        # load_cached_data=False forces the retrieval model to retrain on full_history
        # This is crucial to capture the most recent trends (the validation week)
        print("Building test candidates (this may take a while)...")
        test_candidates = builder.build_test_set(
            full_history, test_customer_ids, load_cached_data=False
        )

        # 3. Predict
        print("Predicting on test candidates...")
        test_preds = ranker.predict(test_candidates)
        test_candidates["score"] = test_preds

        # 4. Select Top 12
        print("Selecting top 12 predictions...")
        submission_top = (
            test_candidates.sort_values([USER_ID_COL, "score"], ascending=[True, False])
            .groupby(USER_ID_COL)
            .head(12)
        )

        # 5. Format
        submission_dict = (
            submission_top.groupby(USER_ID_COL)[ITEM_ID_COL].apply(list).to_dict()
        )

        # 6. Save
        create_submission(submission_dict, output_path=SUBMISSION_PATH)

    else:
        print(
            f"\nMetric ({metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )

    print("=== Execution Complete ===")


if __name__ == "__main__":
    main()
