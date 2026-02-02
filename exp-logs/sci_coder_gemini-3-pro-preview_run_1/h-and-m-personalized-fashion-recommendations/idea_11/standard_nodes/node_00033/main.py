import pandas as pd
import numpy as np
import os
import shutil
import gc
import warnings
from library import config, data_utils, sparse_engine, stratified_inference, evaluation

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clear_cache():
    """
    Clears specific matrix cache files to ensure we don't load
    stale data (e.g., loading train-split matrices during full-train phase).
    """
    files_to_remove = [
        "interaction_matrix.npz",
        "similarity_matrix.npz",
        "habit_matrix.npz",
        "global_trend.npy",
    ]
    for f in files_to_remove:
        path = os.path.join(config.CACHE_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleared cache file: {path}")


def run_failure_analysis(val_df, preds_df, customers_df):
    print("\n=== Failure Analysis ===")

    # Calculate AP@12 for each user
    ground_truth = val_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    preds_map = preds_df.set_index("customer_id")["prediction"].to_dict()

    user_scores = []
    for uid, actual in ground_truth.items():
        pred_str = preds_map.get(uid, "")
        if pred_str:
            pred_list = [int(x) for x in pred_str.split()]
        else:
            pred_list = []

        score = evaluation.apk(actual, pred_list, k=12)
        user_scores.append({"customer_id": uid, "score": score})

    scores_df = pd.DataFrame(user_scores)

    # Merge with customer metadata
    analysis_df = scores_df.merge(customers_df, on="customer_id", how="left")

    # 1. Correlation with Age
    if "age" in analysis_df.columns:
        corr_age = analysis_df["score"].corr(analysis_df["age"])
        print(f"Correlation (Score vs Age): {corr_age:.4f}")

    # 2. Correlation with Activity (if available, usually derived from FN/Active)
    # We'll check 'Active' column from customers.csv if it exists and is numeric
    if "Active" in analysis_df.columns:
        # Fill NaN with 0
        analysis_df["Active"] = analysis_df["Active"].fillna(0)
        corr_active = analysis_df["score"].corr(analysis_df["Active"])
        print(f"Correlation (Score vs Active Status): {corr_active:.4f}")

    # 3. Correlation with History Length (using val_df count as proxy for activity in this period)
    user_activity = val_df.groupby("customer_id").size().reset_index(name="tx_count")
    analysis_df = analysis_df.merge(user_activity, on="customer_id", how="left")
    corr_len = analysis_df["score"].corr(analysis_df["tx_count"])
    print(f"Correlation (Score vs Validation Transaction Count): {corr_len:.4f}")

    print("========================\n")


def main():
    set_seed(config.RANDOM_SEED)

    print("Starting Time-Decayed Graph Cascade (TDGC) Pipeline...")

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    # We use the train.csv from metadata which represents the main history
    # However, for the purpose of this pipeline, we need to simulate the split carefully.
    # The task description says "transactions_train.csv" is the training data.
    # We will load that to be consistent with the standard flow, or use the library loader.

    # Load full transactions
    transactions_df = data_utils.load_transactions(
        os.path.join(config.INPUT_DIR, "transactions_train.csv")
    )

    # Load metadata
    customers_df = pd.read_csv(config.CUSTOMERS_PATH)
    articles_df = data_utils.load_articles()

    # =========================================================================
    # 2. Validation Phase
    # =========================================================================
    print("\n--- Validation Phase ---")

    # Filter to last 20 weeks (TRAIN_WEEKS)
    # This window includes the validation week.
    # We need to split this window into (Train History) and (Validation Ground Truth)

    # First, filter to the window of interest
    recent_transactions = data_utils.filter_date_window(
        transactions_df, weeks=config.TRAIN_WEEKS
    )

    # Split into Train (19 weeks) and Val (1 week)
    train_split, val_split = data_utils.get_time_split(
        recent_transactions, val_weeks=config.VAL_WEEKS
    )

    # Clear cache to ensure we build matrices on train_split ONLY
    clear_cache()

    # Generate Mappings
    # We must include validation users in the mapping to make predictions for them
    val_customers = val_split[["customer_id"]].drop_duplicates()
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = data_utils.generate_mappings(
        train_split, customers_df=val_customers, articles_df=articles_df
    )

    # Build Matrices (Train Split)
    interaction_matrix = sparse_engine.build_decayed_interaction_matrix(
        train_split, user_to_idx, item_to_idx, load_cached_data=False
    )

    similarity_matrix = sparse_engine.compute_similarity_matrix(
        interaction_matrix, top_k=config.SIMILARITY_TOP_K, load_cached_data=False
    )

    # Initialize and Fit Recommender
    recommender = stratified_inference.StratifiedRecommender()
    recommender.fit(
        train_split,
        interaction_matrix,
        similarity_matrix,
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
        load_cached_data=False,  # Force recompute of habit/trend
    )

    # Predict
    val_customer_ids = val_split["customer_id"].unique()
    val_preds_df = recommender.predict(val_customer_ids)

    # Evaluate
    map12 = evaluation.calculate_map12(val_split, val_preds_df)
    print(f"Final Validation Metric: {map12:.16f}")

    # Failure Analysis
    run_failure_analysis(val_split, val_preds_df, customers_df)

    # Cleanup to free memory
    del (
        train_split,
        val_split,
        interaction_matrix,
        similarity_matrix,
        recommender,
        val_preds_df,
    )
    gc.collect()

    # =========================================================================
    # 3. Full Training & Submission
    # =========================================================================
    threshold = 0.0265060791

    if map12 > threshold:
        print("\n--- Generating Submission ---")

        # Clear cache again to build matrices on FULL data
        clear_cache()

        # Use the full recent_transactions (Train + Val weeks) as history
        # Load sample submission to get target customers
        sample_sub = data_utils.load_submission_template()
        test_customers = sample_sub[["customer_id"]]

        # Generate Mappings (Full Data + Test Users)
        user_to_idx, idx_to_user, item_to_idx, idx_to_item = (
            data_utils.generate_mappings(
                recent_transactions,
                customers_df=test_customers,
                articles_df=articles_df,
            )
        )

        # Build Matrices (Full Data)
        interaction_matrix = sparse_engine.build_decayed_interaction_matrix(
            recent_transactions, user_to_idx, item_to_idx, load_cached_data=False
        )

        similarity_matrix = sparse_engine.compute_similarity_matrix(
            interaction_matrix, top_k=config.SIMILARITY_TOP_K, load_cached_data=False
        )

        # Fit Recommender
        recommender = stratified_inference.StratifiedRecommender()
        recommender.fit(
            recent_transactions,
            interaction_matrix,
            similarity_matrix,
            user_to_idx,
            idx_to_user,
            item_to_idx,
            idx_to_item,
            load_cached_data=False,
        )

        # Predict for all test customers
        target_ids = sample_sub["customer_id"].values
        submission_df = recommender.predict(target_ids)

        # Save
        print(f"Saving submission to {config.SUBMISSION_PATH}...")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"Validation score {map12} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
