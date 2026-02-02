import pandas as pd
import numpy as np
import warnings
import sys
import os
from pathlib import Path

# Import library components
from library.config import WORKING_DIR
from library.data_loader import get_time_split_data, load_test_customers
from library.retrieval import CooccurrenceRecommender
from library.feature_builder import RankerDatasetBuilder
from library.ranker import LGBMRanker
from library.utils import calculate_map12, create_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)


def main():
    print("=== H&M Recommendation Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Data Loading & Subsampling
    # ---------------------------------------------------------
    print("\n[1] Loading and Subsampling Data...")

    # Load raw time splits (ignoring cache to ensure we control the data flow for this demo)
    # We use load_cached_data=False to force loading from metadata parquet files
    train_df, val_df = get_time_split_data(load_cached_data=False)

    # Subsample for speed: Limit to 2000 unique customers for training context
    # and 500 customers for validation/testing.
    unique_train_users = train_df["customer_id"].unique()
    unique_val_users = val_df["customer_id"].unique()

    sampled_train_users = np.random.choice(
        unique_train_users, size=min(len(unique_train_users), 2000), replace=False
    )
    sampled_val_users = np.random.choice(
        unique_val_users, size=min(len(unique_val_users), 500), replace=False
    )

    # Filter DataFrames
    train_df_small = train_df[train_df["customer_id"].isin(sampled_train_users)].copy()
    val_df_small = val_df[val_df["customer_id"].isin(sampled_val_users)].copy()

    print(f"  Subsampled Train Rows: {len(train_df_small)}")
    print(f"  Subsampled Val Rows:   {len(val_df_small)}")

    # ---------------------------------------------------------
    # 2. Stage 1: Retrieval (Co-occurrence)
    # ---------------------------------------------------------
    print("\n[2] Stage 1: Retrieval (CooccurrenceRecommender)...")

    # Instantiate Recommender
    rec = CooccurrenceRecommender()

    # Fit on the subsampled training data
    # We set load_cached_data=False to avoid loading the full dataset matrix from disk
    rec.fit(train_df_small, load_cached_data=False)

    # Demonstrate candidate generation for validation users
    # We retrieve top 12 candidates for a quick check
    demo_candidates = rec.generate_candidates(sampled_val_users[:5], k=12)

    # Validation
    assert not demo_candidates.empty, "Candidate generation returned empty DataFrame"
    assert "cooccurrence_score" in demo_candidates.columns
    print("  Retrieval sanity check passed.")

    # ---------------------------------------------------------
    # 3. Stage 2: Feature Engineering
    # ---------------------------------------------------------
    print("\n[3] Stage 2: Feature Engineering (RankerDatasetBuilder)...")

    builder = RankerDatasetBuilder()

    # Build the labeled training dataset for the ranker
    # This process:
    #   1. Fits a recommender on train_df_small
    #   2. Generates candidates for val_df_small users
    #   3. Labels candidates (1 if bought in val_df_small, 0 otherwise)
    #   4. Computes features (User, Item, and Visual Embeddings)
    # Note: This will trigger image embedding extraction (using ResNet)
    ranker_train_df = builder.build_train_set(
        train_df_small, val_df_small, load_cached_data=False
    )

    # Validation
    expected_cols = ["visual_similarity", "age", "product_type_no", "purchased"]
    for col in expected_cols:
        if col not in ranker_train_df.columns:
            raise AssertionError(f"Missing expected feature: {col}")

    print(f"  Ranker Train Data Shape: {ranker_train_df.shape}")
    print(f"  Positive Samples: {ranker_train_df['purchased'].sum()}")

    # ---------------------------------------------------------
    # 4. Stage 3: Ranking (LightGBM)
    # ---------------------------------------------------------
    print("\n[4] Stage 3: Ranking (LGBMRanker)...")

    ranker = LGBMRanker()

    # Optimize hyperparameters for this quick demo
    ranker.params["n_estimators"] = 50
    ranker.params["early_stopping_rounds"] = 10

    # Train the model
    ranker.fit(ranker_train_df, load_cached_data=False)

    # Validation
    if ranker.model is None:
        raise AssertionError("Model failed to train.")
    print("  Ranker trained successfully.")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[5] Inference & Submission Generation...")

    # Load test customers and subsample
    test_customers = load_test_customers()
    sampled_test_ids = test_customers["customer_id"].iloc[:100].values

    # Combine history for test context (Train + Val)
    full_history_small = pd.concat([train_df_small, val_df_small], ignore_index=True)

    # Build test dataset (Candidates + Features)
    ranker_test_df = builder.build_test_set(
        full_history_small, sampled_test_ids, load_cached_data=False
    )

    # Predict scores
    scores = ranker.predict(ranker_test_df)
    ranker_test_df["score"] = scores

    # Select Top 12 items per customer
    # Sort by customer and score descending
    ranker_test_df.sort_values(
        ["customer_id", "score"], ascending=[True, False], inplace=True
    )

    # Group and take top 12
    final_preds = ranker_test_df.groupby("customer_id")["article_id"].apply(list)
    final_preds = final_preds.apply(lambda x: x[:12])

    # Create submission file
    submission_path = Path("./working/demo_submission.csv")
    create_submission(final_preds, output_path=submission_path)

    if not submission_path.exists():
        raise AssertionError("Submission file was not created.")

    # ---------------------------------------------------------
    # 6. Metric Calculation (MAP@12)
    # ---------------------------------------------------------
    print("\n[6] Metric Validation (MAP@12)...")

    # We will evaluate the model on the validation set we created earlier
    # 1. Predict on validation candidates
    val_scores = ranker.predict(ranker_train_df)
    ranker_train_df["score"] = val_scores

    # 2. Format predictions (Top 12)
    val_preds_df = ranker_train_df.sort_values(
        ["customer_id", "score"], ascending=[True, False]
    )
    val_predictions = val_preds_df.groupby("customer_id")["article_id"].apply(list)
    val_predictions = val_predictions.apply(lambda x: x[:12])

    # 3. Format Ground Truth
    ground_truth = val_df_small.groupby("customer_id")["article_id"].apply(list)

    # 4. Filter for common customers (intersection of ground truth and our predictions)
    common_users = ground_truth.index.intersection(val_predictions.index)

    if len(common_users) > 0:
        gt_subset = ground_truth.loc[common_users]
        pred_subset = val_predictions.loc[common_users]

        map_score = calculate_map12(pred_subset, gt_subset)
        print(f"  Calculated MAP@12 on subsample: {map_score:.6f}")
    else:
        print(
            "  No overlapping customers found for metric calculation in this subsample."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
