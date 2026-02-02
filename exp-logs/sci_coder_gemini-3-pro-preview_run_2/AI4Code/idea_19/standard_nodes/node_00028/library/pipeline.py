import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import kendall_tau_metric
from library.data_loader import load_data
from library.feature_engineering import MultiViewExtractor
from library.model_zoo import Stage1Ridge, Stage2LGBM


class RankingPipeline:
    """
    Orchestrates the Stacked Hybrid Ranking Pipeline with Multi-View Instance-Based Anchoring.
    Manages data loading, feature generation, two-stage model training, and submission generation.
    """

    def __init__(self):
        self.extractor = MultiViewExtractor()
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

    def run_train(self, load_cached_data=True):
        """
        Executes the training pipeline:
        1. Loads Train/Val data.
        2. Generates Multi-View Features.
        3. Trains Stage 1 (Ridge) to get OOF predictions.
        4. Trains Stage 2 (LightGBM) using stacked features.
        5. Validates and prints the Kendall Tau score.

        Args:
            load_cached_data (bool): Whether to load intermediate data from parquet cache.
        """
        print("--- Starting Training Pipeline ---")

        # 1. Load Data
        df_train = load_data(split="train", load_cached_data=load_cached_data)
        df_val = load_data(split="val", load_cached_data=load_cached_data)

        # 2. Generate Features
        # The extractor handles fitting the vectorizer on the first call (usually train)
        feats_train = self.extractor.generate_features(
            df_train, split="train", load_cached_data=load_cached_data
        )
        feats_val = self.extractor.generate_features(
            df_val, split="val", load_cached_data=load_cached_data
        )

        # 3. Stage 1: Ridge Regression
        # Generate OOF predictions for training data (to prevent leakage in Stage 2)
        print("Running Stage 1: Ridge Regression (OOF Generation)...")
        ridge_oof_train = self.stage1_model.fit_oof(
            df_train, load_cached_data=load_cached_data
        )
        # Generate predictions for validation data using the final fitted Ridge model
        ridge_pred_val = self.stage1_model.predict(df_val)

        # 4. Stage 2: LightGBM
        print("Running Stage 2: LightGBM Training...")
        self.stage2_model.train(
            df_train,
            ridge_oof_train,
            feats_train,
            df_val,
            ridge_pred_val,
            feats_val,
        )

        # 5. Validation
        print("--- Validating ---")
        # Predict ranks on validation set
        lgbm_pred_val = self.stage2_model.predict(df_val, ridge_pred_val, feats_val)

        # Post-process to get final cell orders
        val_submission = self._post_process_sorting(df_val, lgbm_pred_val)

        # Construct Ground Truth for Validation
        # We need to reconstruct the 'cell_order' string from the 'rank' column in df_val
        print("Calculating Kendall Tau Metric...")
        df_val_sorted = df_val.sort_values(["id", "rank"])
        gt_series = df_val_sorted.groupby("id", observed=True)["cell_id"].apply(
            lambda x: " ".join(x)
        )
        df_val_gt = gt_series.reset_index()
        df_val_gt.columns = ["id", "cell_order"]

        score = kendall_tau_metric(df_val_gt, val_submission)
        print(f"Validation Kendall Tau: {score}")

    def run_inference(self, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Loads Test data.
        2. Generates Features.
        3. Predicts using Stage 1 and Stage 2 models.
        4. Generates and saves the submission file.

        Args:
            load_cached_data (bool): Whether to load intermediate data from parquet cache.
        """
        print("--- Starting Inference Pipeline ---")

        # 1. Load Data
        df_test = load_data(split="test", load_cached_data=load_cached_data)

        # 2. Generate Features
        feats_test = self.extractor.generate_features(
            df_test, split="test", load_cached_data=load_cached_data
        )

        # 3. Predict
        print("Predicting Stage 1 (Ridge)...")
        ridge_pred_test = self.stage1_model.predict(df_test)

        print("Predicting Stage 2 (LightGBM)...")
        lgbm_pred_test = self.stage2_model.predict(df_test, ridge_pred_test, feats_test)

        # 4. Submission
        print("Generating Submission...")
        submission_df = self._post_process_sorting(df_test, lgbm_pred_test)

        submission_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    def _post_process_sorting(self, df_base, df_pred):
        """
        Combines fixed code cell anchors with predicted markdown ranks to determine final order.

        Args:
            df_base (pd.DataFrame): Base dataframe containing all cells (code + markdown).
            df_pred (pd.DataFrame): Predictions for markdown cells ['id', 'cell_id', 'pred_rank'].

        Returns:
            pd.DataFrame: Submission format ['id', 'cell_order'].
        """
        # 1. Handle Code Cells (Anchors)
        # Filter code cells. We assume df_base preserves the relative order of code cells
        # as they appear in the source JSON (which is correct/fixed).
        df_code = df_base[df_base["cell_type"] == "code"].copy()

        # Assign equidistant ranks (0.0 to 1.0) to code cells
        # Calculate position index (0, 1, 2...) within each notebook
        df_code["pos"] = df_code.groupby("id", observed=True).cumcount()
        # Calculate total count of code cells per notebook
        df_code["count"] = df_code.groupby("id", observed=True)["cell_id"].transform(
            "count"
        )

        # Rank = pos / (count - 1).
        # If count=1, rank=0.0.
        denom = df_code["count"] - 1
        denom = denom.replace(0, 1)  # Avoid division by zero
        df_code["rank"] = df_code["pos"] / denom

        # Select relevant columns for merging
        code_ranks = df_code[["id", "cell_id", "rank"]]

        # 2. Handle Markdown Cells (Predictions)
        # Rename pred_rank to rank
        md_ranks = df_pred.rename(columns={"pred_rank": "rank"})[
            ["id", "cell_id", "rank"]
        ]

        # 3. Combine All Cells
        all_ranks = pd.concat([code_ranks, md_ranks], ignore_index=True)

        # 4. Sort
        # Sort primarily by Notebook ID, secondarily by Rank
        all_ranks = all_ranks.sort_values(["id", "rank"])

        # 5. Aggregate to String
        # Group by ID and join cell_ids with space
        submission_series = all_ranks.groupby("id", observed=True)["cell_id"].apply(
            lambda x: " ".join(x)
        )

        # Format as DataFrame
        submission_df = submission_series.reset_index()
        submission_df.columns = ["id", "cell_order"]

        return submission_df
