import os
import pandas as pd
import numpy as np
from typing import List

from library.config import Config
from library.data_manager import DataManager
from library.feature_engine import FeatureEngine
from library.stage1_model import RidgeStacker
from library.stage2_model import LGBMRanker


class SubmissionGenerator:
    """
    Handles the generation of the submission file for the notebook cell ordering task.
    Orchestrates data loading, feature engineering, model inference, and order reconstruction.
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        self.stage1_model = RidgeStacker()
        self.stage2_model = LGBMRanker()

        self.submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    def _get_feature_columns(self) -> List[str]:
        """
        Defines the feature columns used by the Stage 2 LightGBM model.
        Must match the columns used during training.
        """
        # Base anchor features
        cols = [
            "lex_mean",
            "lex_std",
            "lat_mean",
            "lat_std",
            "sym_mean",
            "sym_std",
            "md_ratio",
            "total_code",
            "pred_ridge",  # Output from Stage 1
        ]

        # SVD features
        for i in range(Config.SVD_COMPONENTS):
            cols.append(f"svd_{i}")

        return cols

    def generate_submission(self, load_cached_data: bool = True):
        """
        Main method to generate the submission file.

        Args:
            load_cached_data: Whether to use cached intermediate data (test features).
        """
        print("Starting submission generation...")
        Config.setup()

        # ----------------------------------------------------------------------
        # 1. Load Test Data
        # ----------------------------------------------------------------------
        print("Loading test data...")
        df_test = self.data_manager.get_test_data(load_cached_data=load_cached_data)

        # ----------------------------------------------------------------------
        # 2. Feature Engineering
        # ----------------------------------------------------------------------
        print("Transforming test features...")
        # transform returns sparse matrix (for Ridge) and dense dataframe (for LGBM)
        # We use name="test" to cache test-specific features
        X_sparse_test, df_features_test = self.feature_engine.transform(
            df_test, name="test", load_cached_data=load_cached_data
        )

        # ----------------------------------------------------------------------
        # 3. Stage 1 Inference (Ridge)
        # ----------------------------------------------------------------------
        print("Running Stage 1 (Ridge) inference...")
        # Predict using the sparse TF-IDF matrix
        ridge_preds = self.stage1_model.predict(X_sparse_test)

        # Add predictions to the dense dataframe for Stage 2
        df_features_test["pred_ridge"] = ridge_preds

        # ----------------------------------------------------------------------
        # 4. Stage 2 Inference (LightGBM)
        # ----------------------------------------------------------------------
        print("Running Stage 2 (LightGBM) inference...")
        feature_cols = self._get_feature_columns()

        # Predict final ranks for markdown cells
        lgbm_preds = self.stage2_model.predict(df_features_test, feature_cols)

        # Store predicted ranks
        df_features_test["pred_rank"] = lgbm_preds

        # ----------------------------------------------------------------------
        # 5. Reconstruct Cell Order
        # ----------------------------------------------------------------------
        print("Reconstructing cell orders...")

        # Prepare Markdown predictions: Select relevant columns
        df_md = df_features_test[["id", "cell_id", "pred_rank"]].copy()

        # Prepare Code cells:
        # We need to assign them ranks based on their original order in the notebook.
        # Filter code cells from the raw test dataframe
        df_code = df_test[df_test["cell_type"] == "code"].copy()

        # Calculate ranks for code cells: 0.0 to 1.0 based on position
        # We assume df_test preserves the JSON order for code cells (which is correct/ground truth)
        # Use groupby to handle calculation per notebook

        # Get the count of code cells per notebook
        code_counts = df_code.groupby("id")["cell_id"].transform("count")

        # Get the cumulative count (0-based index) of code cells per notebook
        code_cumcounts = df_code.groupby("id").cumcount()

        # Calculate rank: index / (count - 1). Handle division by zero/single cell case.
        # If count > 1: rank = index / (count - 1)
        # If count == 1: rank = 0.0
        df_code["pred_rank"] = np.where(
            code_counts > 1, code_cumcounts / (code_counts - 1), 0.0
        )

        # Select relevant columns for code
        df_code = df_code[["id", "cell_id", "pred_rank"]]

        # Combine Markdown and Code
        df_all = pd.concat([df_md, df_code], ignore_index=True)

        # Sort: First by Notebook ID, then by Predicted Rank
        df_all.sort_values(by=["id", "pred_rank"], inplace=True)

        # Group by Notebook ID and aggregate cell_ids into a space-delimited string
        submission_series = df_all.groupby("id")["cell_id"].apply(" ".join)

        # Convert to DataFrame
        submission_df = submission_series.reset_index()
        submission_df.columns = ["id", "cell_order"]

        # ----------------------------------------------------------------------
        # 6. Save Submission
        # ----------------------------------------------------------------------
        print(f"Saving submission to {self.submission_path}...")

        # Ensure we include all IDs from the sample submission (test metadata)
        # The groupby should cover all IDs present in df_test.
        _, _, df_test_meta = self.data_manager.load_metadata()
        final_submission = df_test_meta[["id"]].merge(
            submission_df, on="id", how="left"
        )

        # Fill missing values (if any) with empty string or handle appropriately
        final_submission["cell_order"] = final_submission["cell_order"].fillna("")

        final_submission.to_csv(self.submission_path, index=False)
        print("Submission generation complete.")
