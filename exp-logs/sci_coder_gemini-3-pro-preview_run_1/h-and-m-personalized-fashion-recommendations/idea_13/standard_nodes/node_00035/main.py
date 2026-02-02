import pandas as pd
import numpy as np
import os
import gc
import logging
import shutil
from datetime import timedelta
from library.config import Config
from library.recommender import MSGRecommender
from library.data_utils import compute_decay_weights, get_active_inventory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def run_failure_analysis(val_users, val_actual, val_predicted, customers_df):
    logger.info("Running Failure Analysis...")

    # Calculate AP per user
    ap_scores = []
    for a, p in zip(val_actual, val_predicted):
        ap_scores.append(apk(a, p, k=12))

    analysis_df = pd.DataFrame({"customer_id": val_users, "ap": ap_scores})

    # Merge with customer metadata
    analysis_df = analysis_df.merge(
        customers_df[["customer_id", "age"]], on="customer_id", how="left"
    )

    # Calculate correlation
    # Handle NaNs in age
    analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].median())

    corr_age = analysis_df["ap"].corr(analysis_df["age"])
    logger.info(f"Correlation between Error (AP) and Age: {corr_age:.4f}")

    # Bin analysis
    analysis_df["age_bin"] = pd.cut(
        analysis_df["age"], bins=[10, 20, 30, 40, 50, 60, 100]
    )
    bin_perf = analysis_df.groupby("age_bin", observed=True)["ap"].mean()
    logger.info(f"Performance by Age Bin:\n{bin_perf}")


def main():
    config = Config()
    set_seed(config.SEED)

    # --------------------------------------------------------------------------
    # 1. Load Data for Validation
    # --------------------------------------------------------------------------
    logger.info("Loading metadata datasets...")
    # Load optimized types
    dtypes = {"article_id": "int32", "price": "float32", "sales_channel_id": "int8"}

    train_df = pd.read_csv(config.TRAIN_DATA_PATH, dtype=dtypes)
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])

    val_df = pd.read_csv(config.VAL_DATA_PATH, dtype=dtypes)
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    customers_df = pd.read_csv(config.CUSTOMERS_PATH)
    articles_df = pd.read_csv(config.ARTICLES_PATH, dtype={"article_id": "int32"})

    # --------------------------------------------------------------------------
    # 2. Prepare Validation Split
    # --------------------------------------------------------------------------
    logger.info("Preparing validation split...")
    # Split val_df into history and target
    # Target is last 7 days of the validation set
    val_max_date = val_df["t_dat"].max()
    split_date = val_max_date - timedelta(days=7)

    val_history = val_df[val_df["t_dat"] <= split_date].copy()
    val_target = val_df[val_df["t_dat"] > split_date].copy()

    # Ground Truth
    val_target_grouped = (
        val_target.groupby("customer_id")["article_id"].apply(list).reset_index()
    )
    val_target_map = dict(
        zip(val_target_grouped["customer_id"], val_target_grouped["article_id"])
    )

    # Identify validation users (only those who are in the target set)
    val_user_ids = val_target_grouped["customer_id"].values

    logger.info(f"Validation Users: {len(val_user_ids)}")

    # --------------------------------------------------------------------------
    # 3. Build Model (Graphs) on Training Data
    # --------------------------------------------------------------------------
    recommender = MSGRecommender(config)

    # Preprocess Train Data
    train_df = compute_decay_weights(train_df, config)
    active_items_train = get_active_inventory(train_df, config)

    # Initialize Mappings
    recommender.graph_builder.fit_mappings(customers_df, articles_df)

    # Build Graphs (S_fast, S_slow) using Train Data
    # We force run to ensure we have the objects
    X_fast, S_fast, X_slow, S_slow = recommender.graph_builder.run(
        train_df, customers_df, articles_df, active_items_train, load_cached=True
    )

    # Compute Global Trend from Train
    global_trend = recommender._compute_global_trend(
        train_df, recommender.graph_builder.item_map
    )

    # --------------------------------------------------------------------------
    # 4. Validation Inference
    # --------------------------------------------------------------------------
    logger.info("Building validation user vectors...")

    # Preprocess Validation History
    val_history = compute_decay_weights(val_history, config)
    val_history = recommender._prepare_habit_weights(val_history)

    # Build User Vectors (X matrices) for Validation Users
    # Note: graph_builder uses the global mappings established in fit_mappings
    X_val_fast = recommender.graph_builder.build_interaction_matrix(
        val_history, "weight_fast"
    )
    X_val_slow = recommender.graph_builder.build_interaction_matrix(
        val_history, "weight_slow"
    )
    X_val_habit = recommender.graph_builder.build_interaction_matrix(
        val_history, "weight_habit"
    )

    # Map validation user IDs to indices
    val_user_indices = []
    valid_val_users = []

    for uid in val_user_ids:
        if uid in recommender.graph_builder.user_map:
            val_user_indices.append(recommender.graph_builder.user_map[uid])
            valid_val_users.append(uid)

    val_user_indices = np.array(val_user_indices)

    # Batch Prediction
    batch_size = 5000
    all_preds_str = []

    logger.info(f"Predicting for {len(val_user_indices)} validation users...")

    for start_idx in range(0, len(val_user_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(val_user_indices))
        batch_indices = val_user_indices[start_idx:end_idx]

        batch_preds = recommender._get_batch_predictions(
            batch_indices,
            X_val_fast,
            S_fast,
            X_val_slow,
            S_slow,
            X_val_habit,
            global_trend,
        )
        all_preds_str.extend(batch_preds)

        if start_idx % (batch_size * 5) == 0:
            gc.collect()

    # Convert predictions to list of ints for metric calculation
    all_preds_list = []
    for p_str in all_preds_str:
        # p_str is "id1 id2 ..."
        if p_str:
            all_preds_list.append([int(x) for x in p_str.split()])
        else:
            all_preds_list.append([])

    # Get actuals
    all_actuals_list = [val_target_map[uid] for uid in valid_val_users]

    # --------------------------------------------------------------------------
    # 5. Metric & Analysis
    # --------------------------------------------------------------------------
    score = mapk(all_actuals_list, all_preds_list, k=12)
    print(f"Final Validation Metric: {score:.10f}")

    run_failure_analysis(
        valid_val_users, all_actuals_list, all_preds_list, customers_df
    )

    # Clean up memory
    del train_df, val_df, val_history, val_target, X_fast, S_fast, X_slow, S_slow
    del X_val_fast, X_val_slow, X_val_habit
    gc.collect()

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.0265060791

    if score > THRESHOLD:
        logger.info("Validation score passed threshold. Generating submission...")

        # Clear cache to force retraining on full dataset
        # We delete the specific matrix files to ensure graph_builder rebuilds them
        cache_files = [
            "X_fast.npz",
            "S_fast.npz",
            "X_slow.npz",
            "S_slow.npz",
            "X_habit.npz",
        ]
        for f in cache_files:
            path = os.path.join(config.WORKING_DIR, f)
            if os.path.exists(path):
                os.remove(path)

        # Load Full Dataset
        # Note: We use the raw transactions_train.csv
        logger.info("Loading full transaction dataset...")
        full_df = pd.read_csv(
            os.path.join(config.INPUT_DIR, "transactions_train.csv"), dtype=dtypes
        )
        full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])

        # Preprocess Full Data
        full_df = compute_decay_weights(full_df, config)
        active_items_full = get_active_inventory(full_df, config)

        # Load Test Customers (Sample Submission)
        test_customers_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

        # Re-initialize Recommender (clean state)
        recommender_full = MSGRecommender(config)

        # Generate Submission
        recommender_full.generate_submission(
            full_df,
            test_customers_df,
            articles_df,
            active_items_full,
            load_cached=True,  # It will try to load, fail (since we deleted), and rebuild
        )

    else:
        logger.info(
            f"Validation score {score:.6f} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
