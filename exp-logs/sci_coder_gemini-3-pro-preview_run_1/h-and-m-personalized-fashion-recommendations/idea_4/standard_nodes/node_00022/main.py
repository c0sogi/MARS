import pandas as pd
import numpy as np
import os
import sys
import torch
import gc
from scipy.stats import pearsonr

# Import library modules
from library.config import (
    SEED,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    SUBMISSION_PATH,
    TRANSACTIONS_CACHE_PATH,
    USER_HISTORY_PATH,
    BEHAVIOR_MATRIX_PATH,
    GLOBAL_TRENDS_PATH,
    ITEM_MAP_PATH,
)
from library.utils import calculate_map12, apk
from library.data_loader import TransactionLoader, IndexMapper, UserHistoryBuilder
from library.behavioral_engine import CooccurrenceBuilder
from library.smmc_recommender import SMMCModel


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cache_files():
    """
    Removes cached matrix files to ensure fresh build for different phases
    (Validation vs Submission) since item indices might shift.
    """
    paths = [
        USER_HISTORY_PATH,
        BEHAVIOR_MATRIX_PATH,
        GLOBAL_TRENDS_PATH,
        ITEM_MAP_PATH,
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def run_failure_analysis(val_df, preds_df, transactions_history):
    """
    Analyzes model performance on validation set.
    """
    print("\n=== Failure Analysis ===")

    # 1. Prepare Ground Truth
    val_df["customer_id"] = val_df["customer_id"].astype(str)
    ground_truth = (
        val_df.groupby("customer_id")["article_id"]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )
    ground_truth.columns = ["customer_id", "actual"]

    # 2. Merge with Predictions
    preds_df["customer_id"] = preds_df["customer_id"].astype(str)
    merged = ground_truth.merge(preds_df, on="customer_id", how="left")
    merged["prediction"] = merged["prediction"].fillna("")

    # 3. Calculate AP per user
    def get_ap(row):
        actuals = [f"{int(x):010d}" for x in row["actual"]]
        preds = row["prediction"].strip().split()
        return apk(actuals, preds, k=12)

    merged["ap"] = merged.apply(get_ap, axis=1)

    # 4. Feature: History Length (Activity level)
    # Count transactions per user in the input history
    user_activity = transactions_history["customer_id"].value_counts().reset_index()
    user_activity.columns = ["customer_id", "history_len"]
    user_activity["customer_id"] = user_activity["customer_id"].astype(str)

    analysis_df = merged.merge(user_activity, on="customer_id", how="left")
    analysis_df["history_len"] = analysis_df["history_len"].fillna(0)

    # 5. Correlation
    corr, p_val = pearsonr(analysis_df["ap"], analysis_df["history_len"])
    print(f"Correlation (AP vs History Length): {corr:.4f} (p={p_val:.4e})")

    # 6. Binning for insight
    analysis_df["hist_bin"] = pd.cut(
        analysis_df["history_len"],
        bins=[-1, 0, 5, 20, 100, 9999],
        labels=["0", "1-5", "6-20", "21-100", "100+"],
    )
    print("\nMean AP by History Length:")
    print(analysis_df.groupby("hist_bin", observed=True)["ap"].mean())
    print("========================\n")


def main():
    set_seed(SEED)
    print("Initializing SMMC Pipeline...")

    # =========================================================================
    # 1. Data Loading
    # =========================================================================
    # We use TransactionLoader to get the raw dataframe of recent transactions.
    # Note: We will filter this manually for validation to ensure strict separation.
    loader = TransactionLoader()
    # Load full recent history (Train + Val periods)
    df_full_transactions = loader.load_transactions(load_cached_data=True)

    # Load user sets
    df_val_users = pd.read_csv(VAL_PATH)
    df_test_users = pd.read_csv(TEST_PATH)

    # =========================================================================
    # 2. Validation Phase
    # =========================================================================
    print("\n" + "=" * 40)
    print("PHASE 1: VALIDATION")
    print("=" * 40)

    # Clear cache to ensure we build matrices specifically for the validation split
    clear_cache_files()

    # Define Split: Last 7 days of the available data is the "Target", rest is "History"
    max_date = df_full_transactions["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=7)

    print(f"Validation Split Date: {split_date}")

    # Split transactions
    # Validation Input: History for all users up to split date
    val_input_df = df_full_transactions[
        df_full_transactions["t_dat"] <= split_date
    ].copy()

    # Validation Target: Transactions after split date (Ground Truth)
    # We only care about users in the provided val.csv list for scoring
    val_target_df = df_full_transactions[
        (df_full_transactions["t_dat"] > split_date)
        & (df_full_transactions["customer_id"].isin(df_val_users["customer_id"]))
    ].copy()

    # Recalculate days_elapsed for the input set relative to the split date
    # (The loader calculated it relative to max_date, we need it relative to split_date for correct decay)
    val_input_df["days_elapsed"] = (split_date - val_input_df["t_dat"]).dt.days
    val_input_df["days_elapsed"] = val_input_df["days_elapsed"].astype("int16")

    # Initialize Components
    mapper = IndexMapper()
    history_builder = UserHistoryBuilder()
    beh_builder = CooccurrenceBuilder()
    model = SMMCModel()

    # Fit Mapper (Rows: Val Users, Cols: Items in Val Input)
    mapper.fit(val_input_df, df_val_users)

    # Build Matrices
    # 1. User History (Val Users x Items)
    u_hist = history_builder.build_history(val_input_df, mapper, load_cached_data=False)

    # 2. Behavior Matrix (Items x Items)
    beh_matrix = beh_builder.build_similarity_matrix(u_hist, load_cached_data=False)

    # Predict
    val_preds = model.predict(
        u_hist, beh_matrix, mapper, val_input_df, load_cached_data=False
    )

    # Score
    print("Calculating MAP@12...")
    score = calculate_map12(val_target_df, val_preds)
    print(f"Final Validation Metric: {score:.10f}")

    # Failure Analysis
    run_failure_analysis(val_target_df, val_preds, val_input_df)

    # Cleanup to free memory
    del u_hist, beh_matrix, val_preds, val_input_df, val_target_df
    gc.collect()

    # =========================================================================
    # 3. Submission Phase
    # =========================================================================
    THRESHOLD = 0.0265060791

    if score > THRESHOLD:
        print("\n" + "=" * 40)
        print("PHASE 2: SUBMISSION")
        print("=" * 40)

        # Clear cache again to rebuild matrices with FULL data
        clear_cache_files()

        # Use the full transaction set (Train + Val) for maximum signal
        # Loader already returned this in df_full_transactions
        # days_elapsed is already correct relative to the absolute max date

        # Fit Mapper (Rows: Test Users, Cols: All Active Items)
        mapper.fit(df_full_transactions, df_test_users)

        # Build Matrices
        # 1. User History (Test Users x Items)
        u_hist = history_builder.build_history(
            df_full_transactions, mapper, load_cached_data=False
        )

        # 2. Visual Matrix
        embeds = visual_embedder.extract_embeddings(mapper, load_cached_data=False)
        vis_matrix = visual_builder.build_similarity_matrix(
            embeds, load_cached_data=False
        )

        # 3. Behavior Matrix
        beh_matrix = beh_builder.build_similarity_matrix(u_hist, load_cached_data=False)

        # Predict
        model.predict(
            u_hist,
            beh_matrix,
            vis_matrix,
            mapper,
            df_full_transactions,
            load_cached_data=False,
        )

        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"Validation score {score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
