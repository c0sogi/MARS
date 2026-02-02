import sys
import os
import numpy as np
import pandas as pd
import random

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library import config
from library import utils
from library.adipc_model import ADIPCRecommender


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(valid_df, preds_df, history_df, user_map):
    """
    Analyzes the correlation between model performance (AP) and user history length.
    """
    print("Performing failure analysis...")

    # 1. Calculate AP for each user
    # Group ground truth
    ground_truth = valid_df.groupby("customer_id")["article_id"].apply(list).to_dict()

    # Parse predictions
    if "customer_id" in preds_df.columns:
        predictions_map = preds_df.set_index("customer_id")["prediction"].to_dict()
    else:
        predictions_map = preds_df["prediction"].to_dict()

    user_scores = []
    user_ids = []

    for customer_id, actual_items in ground_truth.items():
        pred_str = predictions_map.get(customer_id, "")
        if pd.isna(pred_str) or pred_str == "":
            predicted_items = []
        else:
            try:
                predicted_items = [int(x) for x in pred_str.split()]
            except ValueError:
                predicted_items = []

        score = utils.apk(actual_items, predicted_items, k=12)
        user_scores.append(score)
        user_ids.append(customer_id)

    metrics_df = pd.DataFrame({"customer_id": user_ids, "ap_score": user_scores})

    # 2. Calculate History Length for these users
    # We need to map customer_id to user_idx to query history_df efficiently,
    # or just group history_df by user_idx and map back.

    # Get user_idx for the validation users
    target_users_map = user_map[user_map["customer_id"].isin(user_ids)].copy()

    # Count transactions in history per user
    # history_df has 'user_idx'
    history_counts = (
        history_df.groupby("user_idx").size().reset_index(name="history_len")
    )

    # Merge map and counts
    analysis_df = target_users_map.merge(history_counts, on="user_idx", how="left")
    analysis_df["history_len"] = analysis_df["history_len"].fillna(0)

    # Merge with scores
    analysis_df = analysis_df.merge(metrics_df, on="customer_id", how="inner")

    # 3. Calculate Correlation
    correlation = analysis_df["ap_score"].corr(analysis_df["history_len"])

    print(f"Correlation between User History Length and AP Score: {correlation:.4f}")

    # Additional Insight: Average AP for Cold Start vs Heavy Users
    cold_start = analysis_df[analysis_df["history_len"] <= 5]["ap_score"].mean()
    heavy_users = analysis_df[analysis_df["history_len"] > 50]["ap_score"].mean()
    print(f"Mean AP - Cold Start (<=5 txns): {cold_start:.4f}")
    print(f"Mean AP - Heavy Users (>50 txns): {heavy_users:.4f}")


def main():
    # 1. Setup
    set_seed(config.RANDOM_SEED)
    print("Initializing ADIPC Pipeline...")

    recommender = ADIPCRecommender()

    # 2. Validation Phase
    print("\n" + "=" * 30)
    print("PHASE 1: VALIDATION")
    print("=" * 30)

    # Fit on training portion (T-10 weeks to T-1 week)
    recommender.fit(mode="validation", load_cached_data=True)

    # Load validation data structures
    val_data = recommender.data_handler.load_dataset(
        mode="validation", load_cached_data=True
    )

    # Predict on validation set
    # Note: target_users are indices, prediction returns dataframe with customer_id
    val_preds = recommender.predict(
        target_user_indices=val_data["target_users"],
        history_df=val_data["history_df"],
        cutoff_date=val_data["cutoff_date"],
    )

    # Restore string identifiers for evaluation
    # Cite debug_lesson_6: Standardize Identifier Formatting Across Inference and Evaluation Interfaces
    validation_truth = (
        val_data["future_df"]
        .merge(val_data["user_map"], on="user_idx", how="left")
        .merge(val_data["item_map"], on="article_idx", how="left")
    )

    # Compute Metric
    # future_df contains the ground truth for the validation period
    map_score = utils.calculate_map12(validation_truth, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {map_score:.10f}")

    # Failure Analysis
    perform_failure_analysis(
        valid_df=validation_truth,
        preds_df=val_preds,
        history_df=val_data["history_df"],
        user_map=val_data["user_map"],
    )

    # 3. Submission Phase
    print("\n" + "=" * 30)
    print("PHASE 2: SUBMISSION")
    print("=" * 30)

    THRESHOLD = 0.0265060791

    if map_score > THRESHOLD:
        print(
            f"Validation score {map_score:.6f} exceeds threshold {THRESHOLD}. Proceeding to submission."
        )

        # Refit on FULL dataset
        # This rebuilds the similarity matrix using the most recent data
        recommender.fit(mode="submission", load_cached_data=True)

        # Load submission targets (sample submission users)
        sub_data = recommender.data_handler.load_dataset(
            mode="submission", load_cached_data=True
        )

        # Predict
        sub_preds = recommender.predict(
            target_user_indices=sub_data["target_users"],
            history_df=sub_data["history_df"],
            cutoff_date=sub_data["cutoff_date"],
        )

        # Save
        save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        sub_preds.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"Validation score {map_score:.6f} does not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
