import pandas as pd
import numpy as np
import gc
import sys
import warnings
from library.config import Config
from library.utils import Timer
from library.data import DataManager
from library.retrieval import HybridRetrieval
from library.features import FeatureEngineer
from library.ranker import Ranker
from library.evaluation import Evaluator, apk

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------------------------
    print("=== Starting Orchestration Pipeline ===")
    # Set seeds for reproducibility
    np.random.seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    dm = DataManager()
    # Load cached data (Train/Val split is already done by DataManager based on last 7 days)
    data = dm.load_data(load_cached_data=True)

    train_df = data["train"]
    val_df = data["val"]
    test_df = data["test"]
    articles_df = data["articles"]
    customers_df = data["customers"]

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")

    # -------------------------------------------------------------------------
    # 2. Prepare Ranker Training Data (Internal Time Split)
    # -------------------------------------------------------------------------
    # We need to simulate the test scenario to train the ranker.
    # We split the provided 'train_df' into a 'History' and a 'Local Target'.
    with Timer("Ranker Data Prep"):
        max_train_date = train_df["t_dat"].max()
        split_date = max_train_date - pd.Timedelta(days=7)

        print(f"Creating Ranker Training Split (Cutoff: {split_date})...")

        # History for Ranker Training (T-4 weeks to T-1 week)
        train_hist = train_df[train_df["t_dat"] <= split_date].reset_index(drop=True)

        # Ground Truth for Ranker Training (Last week of train_df)
        train_target = train_df[train_df["t_dat"] > split_date].reset_index(drop=True)

        # Sampling: Select N active customers to train the ranker efficiently
        SAMPLE_SIZE = 200000
        active_users = train_target["customer_id_idx"].unique()

        if len(active_users) > SAMPLE_SIZE:
            print(
                f"Sampling {SAMPLE_SIZE} users from {len(active_users)} active users for training..."
            )
            train_users = np.random.choice(
                active_users, size=SAMPLE_SIZE, replace=False
            )
        else:
            train_users = active_users

        # Retrieval for Ranker Train
        retriever = HybridRetrieval()
        cand_train = retriever.generate_candidates(
            train_hist, train_users, mode="ranker_train", load_cached_data=True
        )

        # Features for Ranker Train
        engineer = FeatureEngineer()
        feat_train = engineer.generate_features(
            cand_train,
            train_hist,
            articles_df,
            customers_df,
            mode="ranker_train",
            labeled_data=train_target,
            load_cached_data=False,
        )

        # Cleanup
        del train_hist, train_target, cand_train
        gc.collect()

    # -------------------------------------------------------------------------
    # 3. Prepare Validation Data (Full Hold-out)
    # -------------------------------------------------------------------------
    # For validation, we use the full provided 'val_df' as target and full 'train_df' as history.
    with Timer("Validation Data Prep"):
        val_users = val_df["customer_id_idx"].unique()

        # Retrieval for Val
        cand_val = retriever.generate_candidates(
            train_df, val_users, mode="val", load_cached_data=True
        )

        # Features for Val
        # Note: We pass labeled_data=val_df to generate labels for evaluation (AUC) during training
        feat_val = engineer.generate_features(
            cand_val,
            train_df,
            articles_df,
            customers_df,
            mode="val",
            labeled_data=val_df,
            load_cached_data=False,
        )

        # Cleanup
        del cand_val
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Train Ranker
    # -------------------------------------------------------------------------
    ranker = Ranker()

    # Define features (exclude metadata and labels)
    exclude_cols = [
        "customer_id_idx",
        "article_id_idx",
        "label",
        "t_dat",
        "score",
        "prediction",
    ]
    feature_cols = [c for c in feat_train.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    ranker.train(feat_train, feat_val, feature_cols)

    # Cleanup training data to free memory for inference
    del feat_train
    gc.collect()

    # -------------------------------------------------------------------------
    # 5. Evaluation (MAP@12)
    # -------------------------------------------------------------------------
    print("Generating validation predictions...")
    # ranker.predict returns a DataFrame with [customer_id, prediction]
    # It also saves to submission.csv, but we will overwrite that later if needed.
    val_preds_df = ranker.predict(feat_val, feature_cols)

    evaluator = Evaluator()
    map12_score = evaluator.calculate_map12(val_df, val_preds_df)

    print(f"Final Validation Metric: {map12_score}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    with Timer("Failure Analysis"):
        # 1. Calculate Per-User Precision (AP)
        # Prepare Ground Truth Dictionary
        val_df_str = val_df.copy()
        if val_df_str["article_id"].dtype != object:
            val_df_str["article_id"] = val_df_str["article_id"].astype(str)

        gt_dict = val_df_str.groupby("customer_id")["article_id"].apply(list).to_dict()

        # Prepare Predictions Dictionary
        pred_dict = dict(
            zip(val_preds_df["customer_id"], val_preds_df["prediction"].str.split())
        )

        # Calculate AP for each user in validation set
        user_scores = []
        for cust_id, actual_items in gt_dict.items():
            pred_items = pred_dict.get(cust_id, [])
            score = apk(actual_items, pred_items, k=12)
            user_scores.append({"customer_id": cust_id, "ap_score": score})

        scores_df = pd.DataFrame(user_scores)

        # 2. Merge with Customer Metadata
        # We need to merge with customers_df to get features like Age
        analysis_df = scores_df.merge(customers_df, on="customer_id", how="left")

        # 3. Calculate Correlations
        # Select numeric columns for correlation
        corr_cols = ["ap_score", "age"]
        if "FN" in analysis_df.columns:
            analysis_df["FN"] = analysis_df["FN"].fillna(0)
            corr_cols.append("FN")
        if "Active" in analysis_df.columns:
            analysis_df["Active"] = analysis_df["Active"].fillna(0)
            corr_cols.append("Active")
        if "club_member_status" in analysis_df.columns:
            # If categorical, we might have encoded it or it's object.
            # If it's object, skip for simple correlation or encode.
            if pd.api.types.is_numeric_dtype(analysis_df["club_member_status"]):
                corr_cols.append("club_member_status")

        print("Correlation between Error (AP) and Features:")
        corr_matrix = analysis_df[corr_cols].corr()
        print(corr_matrix["ap_score"].sort_values(ascending=False))

        # Interpretation
        print(
            "Note: Positive correlation with AP means the feature is associated with BETTER performance."
        )

        del val_df_str, gt_dict, pred_dict, scores_df, analysis_df
        gc.collect()

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.022018351021215375

    if map12_score > THRESHOLD:
        print(
            f"\nScore ({map12_score}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        with Timer("Submission Generation"):
            # 1. Combine History (Train + Val)
            full_history = pd.concat([train_df, val_df], ignore_index=True)

            # 2. Identify Test Users
            test_users = test_df["customer_id_idx"].unique()

            # 3. Retrieval
            cand_test = retriever.generate_candidates(
                full_history, test_users, mode="test", load_cached_data=True
            )

            # 4. Features
            feat_test = engineer.generate_features(
                cand_test,
                full_history,
                articles_df,
                customers_df,
                mode="test",
                load_cached_data=True,
            )

            # 5. Predict & Save
            # This method automatically saves to Config.PATH_SUBMISSION
            ranker.predict(feat_test, feature_cols)

            print("Submission file generated successfully.")
    else:
        print(
            f"\nScore ({map12_score}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )

    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
