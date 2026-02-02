import pandas as pd
import numpy as np
import os
import sys
import gc
import random
from scipy import stats

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_processor import (
    load_and_filter_data,
    create_mappings,
    process_customer_cohorts,
)
from library.matrix_factory import MatrixFactory
from library.trend_analyzer import TrendAnalyzer
from library.inference_engine import StratifiedRecommender
from library.utils import calculate_map12, format_submission, apk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(val_df, val_preds_dict, train_df):
    print("\n--- Failure Analysis ---")

    # 1. Calculate AP per user
    val_grouped = val_df.groupby("customer_id")["article_id"].apply(list)

    user_scores = []
    for cust_id, actual_items in val_grouped.items():
        pred_items = val_preds_dict.get(cust_id, [])
        score = apk(actual_items, pred_items, k=12)
        user_scores.append({"customer_id": cust_id, "ap": score})

    analysis_df = pd.DataFrame(user_scores)

    # 2. Get User Features
    # Feature A: History Length (from train)
    print("Aggregating training history length...")
    history_len = train_df.groupby("customer_id").size().reset_index(name="history_len")

    # Feature B: Age (from customers.csv)
    print("Loading customer metadata...")
    customers = pd.read_csv(os.path.join(Config.INPUT_DIR, "customers.csv"))
    # Fill NaN age with median (simple imputation for analysis)
    customers["age"] = customers["age"].fillna(customers["age"].median())

    # 3. Merge
    analysis_df = analysis_df.merge(history_len, on="customer_id", how="left")
    analysis_df = analysis_df.merge(
        customers[["customer_id", "age"]], on="customer_id", how="left"
    )

    # Fill missing history with 0
    analysis_df["history_len"] = analysis_df["history_len"].fillna(0)

    # 4. Correlation
    print("Calculating correlations with Error (1 - AP)...")
    # We correlate with performance (AP). Negative correlation means feature helps (higher feature -> higher AP).
    # To frame it as "Error Magnitude", we can correlate with (1 - AP).
    analysis_df["error"] = 1.0 - analysis_df["ap"]

    cols_to_analyze = ["age", "history_len"]
    correlations = (
        analysis_df[["error"] + cols_to_analyze].corr()["error"].drop("error")
    )

    print("Correlation between Model Error and Features:")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing SDCC Orchestration...")

    # 2. Data Loading
    # Using load_cached_data=True as requested to utilize any existing preprocessed files
    train_df, val_df, test_df = load_and_filter_data(load_cached_data=True)

    # 3. Mappings
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = create_mappings(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Cohort Processing
    user_cohort_map = process_customer_cohorts(user_to_idx, load_cached_data=True)

    # 5. Matrix Construction
    print("\n--- Building Model Components ---")

    # Stratum 1: User History
    U = MatrixFactory.build_user_history_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=True
    )

    # Stratum 2: Hybrid Similarity
    S_sym = MatrixFactory.build_symmetric_similarity(
        train_df, user_to_idx, item_to_idx, load_cached_data=True
    )
    S_fwd = MatrixFactory.build_transition_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=True
    )
    S_hybrid = MatrixFactory.get_hybrid_matrix(S_sym, S_fwd)

    # Clean up intermediate matrices to save RAM
    del S_sym, S_fwd
    gc.collect()

    # Stratum 3 & 4: Trends
    global_trends = TrendAnalyzer.compute_global_trends(
        train_df, item_to_idx, load_cached_data=True
    )
    cohort_trends = TrendAnalyzer.compute_cohort_trends(
        train_df, user_cohort_map, user_to_idx, item_to_idx, load_cached_data=True
    )

    # 6. Inference Engine Initialization
    print("\n--- Initializing Recommender Engine ---")
    recommender = StratifiedRecommender(
        user_history_matrix=U,
        hybrid_matrix=S_hybrid,
        cohort_trends=cohort_trends,
        global_trends=global_trends,
        user_cohort_map=user_cohort_map,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )

    # 7. Validation
    print("\n--- Starting Validation ---")
    val_customers = val_df["customer_id"].unique()
    print(f"Predicting for {len(val_customers)} validation customers...")

    # Predict returns indices
    val_preds_matrix = recommender.predict(val_customers)

    # Convert indices to article_ids for scoring
    val_preds_dict = {}
    for i, cid in enumerate(val_customers):
        # Map indices back to article IDs
        # Note: idx_to_item returns the original article_id (int)
        pred_items = [idx_to_item[idx] for idx in val_preds_matrix[i]]
        val_preds_dict[cid] = pred_items

    # Calculate Metric
    score = calculate_map12(val_df, val_preds_dict)
    print(f"Final Validation Metric: {score}")

    # 8. Failure Analysis
    perform_failure_analysis(val_df, val_preds_dict, train_df)

    # Clean up validation data
    del val_preds_matrix, val_preds_dict
    gc.collect()

    # 9. Submission
    THRESHOLD = 0.0265060791
    if score > THRESHOLD:
        print(
            f"\nValidation score {score} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_customers = test_df["customer_id"].unique()
        print(f"Predicting for {len(test_customers)} test customers...")

        test_preds_matrix = recommender.predict(test_customers)

        format_submission(test_preds_matrix, test_customers, idx_to_item)
    else:
        print(
            f"\nValidation score {score} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
