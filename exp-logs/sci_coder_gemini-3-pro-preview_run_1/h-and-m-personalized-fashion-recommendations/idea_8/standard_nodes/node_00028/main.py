import pandas as pd
import numpy as np
import os
import gc
import sys

# Ensure library is in path
sys.path.append(os.getcwd())

from library.utils import set_seed, calculate_map12, apk, Timer
from library.recommender import StratifiedRecommender


def main():
    # 1. Setup
    set_seed(42)
    rec = StratifiedRecommender(working_dir="./working/idea_8")

    # Configuration
    TRAIN_WEEKS = 10
    VAL_DAYS = 7
    THRESHOLD = 0.0265060791

    # ==========================================
    # PHASE 1: Validation & Failure Analysis
    # ==========================================
    with Timer("Validation Phase"):
        # Load Data (Validation Split)
        train_df, val_df, _ = rec.loader.load_transactions(
            train_weeks=TRAIN_WEEKS,
            val_days=VAL_DAYS,
            validation=True,
            load_cached_data=True,
        )

        # Build Matrices
        X, user_map, item_map = rec.matrix_builder.build(
            train_df,
            pd.DataFrame(
                {"customer_id": val_df["customer_id"].unique()}
            ),  # Dummy test set for map building
            load_cached_data=True,
            suffix="_val",
        )

        # Compute Similarity
        S = rec.sim_engine.compute_similarity(X, top_k=100, load_cached_data=True)

        # Compute Global Trend
        # Logic copied from StratifiedRecommender.run to ensure consistency
        global_trend = np.array(X.sum(axis=0)).flatten()
        max_trend = global_trend.max()
        if max_trend > 0:
            global_trend = 10.0 * (global_trend / max_trend)
        global_trend = global_trend.astype(np.float32)

        # Predict
        target_users = val_df["customer_id"].unique()
        preds = rec._predict_stratified(
            X, S, global_trend, user_map, item_map, target_users
        )

        # Calculate Metric
        val_metric = calculate_map12(preds, val_df)
        print(f"Final Validation Metric: {val_metric:.10f}")

        # --- Failure Analysis ---
        print("\n[Failure Analysis] Analyzing error patterns...")

        # 1. Prepare Ground Truth for per-user calculation
        # Group by customer and convert to list of strings (zfilled)
        val_grouped = (
            val_df.groupby("customer_id")["article_id"].apply(list).reset_index()
        )

        def process_ids(x):
            return [str(i).zfill(10) for i in x]

        # Merge predictions and actuals
        analysis_df = pd.merge(val_grouped, preds, on="customer_id", how="inner")

        # Calculate AP per user
        user_aps = []
        for _, row in analysis_df.iterrows():
            actual = process_ids(row["article_id"])
            predicted = row["prediction"].split()
            score = apk(actual, predicted, k=12)
            user_aps.append(score)

        analysis_df["ap_score"] = user_aps

        # 2. Load Customer Metadata
        cust_df = rec.loader.load_customers()

        # 3. Merge and Correlate
        # Select features and handle missing values
        cust_features = cust_df[["customer_id", "age", "FN", "Active"]].copy()
        cust_features["age"] = cust_features["age"].fillna(
            cust_features["age"].median()
        )
        cust_features["FN"] = cust_features["FN"].fillna(0)
        cust_features["Active"] = cust_features["Active"].fillna(0)

        merged_analysis = pd.merge(
            analysis_df[["customer_id", "ap_score"]],
            cust_features,
            on="customer_id",
            how="left",
        )

        # Calculate correlations
        correlations = merged_analysis[["ap_score", "age", "FN", "Active"]].corr()[
            "ap_score"
        ]
        print("Correlation between AP Score and User Features:")
        print(correlations.drop("ap_score"))

        # Cleanup to free memory for full training
        del (
            train_df,
            val_df,
            X,
            S,
            global_trend,
            preds,
            analysis_df,
            merged_analysis,
            cust_df,
        )
        gc.collect()

    # ==========================================
    # PHASE 2: Submission
    # ==========================================
    if val_metric > THRESHOLD:
        print(
            f"\n[Submission] Metric {val_metric:.6f} > {THRESHOLD}. Generating submission..."
        )

        with Timer("Submission Phase"):
            # Load Full Data (No Validation Split)
            # Note: We pass load_cached_data=False or ensure cache keys differ.
            # The loader handles cache keys based on parameters (val=False vs True), so it's safe.
            train_df, _, test_customers = rec.loader.load_transactions(
                train_weeks=TRAIN_WEEKS,
                val_days=VAL_DAYS,
                validation=False,
                load_cached_data=True,
            )

            # Build Matrices (Full Data)
            X, user_map, item_map = rec.matrix_builder.build(
                train_df, test_customers, load_cached_data=True, suffix="_full"
            )

            # Compute Similarity (Full Data)
            # We force recompute or use a different cache file if the library supports it.
            # The library saves as 'similarity_matrix_k100.npz'.
            # To avoid using the validation matrix, we should ideally clear the cache or rely on the fact
            # that X is different. However, the library sim_engine checks for file existence.
            # We will force compute by setting load_cached_data=False for the sim engine here
            # to ensure we use the full-data matrix.
            S = rec.sim_engine.compute_similarity(X, top_k=100, load_cached_data=False)

            # Compute Global Trend
            global_trend = np.array(X.sum(axis=0)).flatten()
            max_trend = global_trend.max()
            if max_trend > 0:
                global_trend = 10.0 * (global_trend / max_trend)
            global_trend = global_trend.astype(np.float32)

            # Predict for Test Users
            target_users = test_customers["customer_id"].unique()
            preds = rec._predict_stratified(
                X, S, global_trend, user_map, item_map, target_users
            )

            # Save
            sub_dir = "./submission"
            os.makedirs(sub_dir, exist_ok=True)
            sub_path = os.path.join(sub_dir, "submission.csv")
            preds.to_csv(sub_path, index=False)
            print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\n[Submission] Metric {val_metric:.6f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
