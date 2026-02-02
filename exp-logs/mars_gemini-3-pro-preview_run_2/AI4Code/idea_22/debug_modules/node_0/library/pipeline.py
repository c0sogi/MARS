import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import kendall_tau_metric, save_artifacts
from library.dataset import NotebookLoader
from library.vectorization import TextPipeline
from library.feature_engineering import FeatureExtractor
from library.modeling import Stage1Ridge, Stage2LGBM


class RankingPipeline:
    def __init__(self):
        self.loader = NotebookLoader()
        self.text_pipeline = TextPipeline()
        self.extractor = FeatureExtractor()
        self.stage1 = Stage1Ridge()
        self.stage2 = Stage2LGBM()

        # Ensure working and submission directories exist
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    def train(self, load_cached_data=True):
        """
        Executes the full training pipeline:
        1. Load Data
        2. Vectorize Text
        3. Stage 1 (Ridge) OOF & Training
        4. Feature Engineering (Content-Aware Neighbors)
        5. Stage 2 (LightGBM) Stacking
        6. Validation Evaluation
        """
        print("=== Starting Training Pipeline ===")

        # 1. Load Data
        df_train = self.loader.load_train_data(load_cached_data=load_cached_data)
        df_val = self.loader.load_val_data(load_cached_data=load_cached_data)

        # 2. Vectorize Text (Fit on Train Markdown)
        # This fits TF-IDF and SVD models
        self.text_pipeline.fit_transform_corpus(
            df_train, load_cached_model=load_cached_data
        )

        # 3. Stage 1: Ridge Regression
        # Generate OOF predictions for Train (used as features for Stage 2)
        train_s1_preds = self.stage1.get_oof_predictions(
            df_train, self.text_pipeline, load_cached_data=load_cached_data
        )

        # Generate predictions for Validation
        val_s1_preds = self.stage1.predict(df_val, self.text_pipeline)

        # 4. Feature Engineering
        # Extract content-aware neighbor features
        train_features = self.extractor.extract_features(
            df_train,
            self.text_pipeline,
            mode="train",
            load_cached_data=load_cached_data,
        )
        val_features = self.extractor.extract_features(
            df_val, self.text_pipeline, mode="val", load_cached_data=load_cached_data
        )

        # 5. Stage 2: LightGBM Stacking
        self.stage2.train(train_features, val_features, train_s1_preds, val_s1_preds)

        # 6. Evaluation on Validation Set
        print("=== Evaluating on Validation Set ===")
        # Generate final ranks for validation markdown cells
        val_final_preds = self.stage2.predict(val_features, val_s1_preds)

        # Reconstruct full notebook orders and calculate metric
        score = self._evaluate_kendall(df_val, val_final_preds)
        print(f"Validation Kendall Tau: {score}")

    def predict(self, load_cached_data=True):
        """
        Executes the inference pipeline:
        1. Load Test Data
        2. Stage 1 Predictions
        3. Feature Extraction
        4. Stage 2 Predictions
        5. Sort and Generate Submission
        """
        print("=== Starting Inference Pipeline ===")

        # 1. Load Test Data
        df_test = self.loader.load_test_data(load_cached_data=load_cached_data)

        # 2. Ensure Vectorizer is ready
        if not self.text_pipeline.is_fitted:
            # Attempt to load from cache if not explicitly trained in this session
            self.text_pipeline.fit_transform_corpus(None, load_cached_model=True)

        # 3. Stage 1 Predictions
        test_s1_preds = self.stage1.predict(df_test, self.text_pipeline)

        # 4. Feature Extraction
        test_features = self.extractor.extract_features(
            df_test, self.text_pipeline, mode="test", load_cached_data=load_cached_data
        )

        # 5. Stage 2 Predictions
        test_final_preds = self.stage2.predict(test_features, test_s1_preds)

        # 6. Generate Submission
        self._generate_submission(df_test, test_final_preds)

    def _evaluate_kendall(self, df_val, pred_df):
        """
        Internal helper to evaluate Kendall Tau on validation set.
        Merges predicted markdown ranks with ground truth code ranks.
        """
        # 1. Prepare Ground Truth Orders
        # df_val contains 'cell_id' and 'rank' (integer index)
        gt_dict = {}
        val_grouped = df_val.groupby("id")
        for nb_id, group in val_grouped:
            # Sort by ground truth rank
            sorted_group = group.sort_values("rank")
            cell_order_str = " ".join(sorted_group["cell_id"].tolist())
            gt_dict[nb_id] = cell_order_str

        df_gt = pd.DataFrame([{"id": k, "cell_order": v} for k, v in gt_dict.items()])

        # 2. Prepare Predicted Orders
        # Get Code Cells with their Ground Truth Normalized Ranks (pct_rank)
        mask_code = df_val["cell_type"] == "code"
        code_cells = df_val[mask_code][["id", "cell_id", "pct_rank"]].copy()
        code_cells.rename(columns={"pct_rank": "rank"}, inplace=True)

        # Get Markdown Cells with Predicted Ranks
        md_preds = pred_df[["id", "cell_id", "pred_rank"]].copy()
        md_preds.rename(columns={"pred_rank": "rank"}, inplace=True)

        # Concatenate and Sort
        all_cells = pd.concat([code_cells, md_preds], axis=0)

        preds_dict = {}
        grouped = all_cells.groupby("id")
        for nb_id, group in grouped:
            sorted_group = group.sort_values("rank")
            preds_dict[nb_id] = sorted_group["cell_id"].tolist()

        # 3. Compute Metric
        return kendall_tau_metric(df_gt, preds_dict)

    def _generate_submission(self, df_test, pred_df):
        """
        Generates the submission.csv file.
        Merges predicted markdown ranks with implicit code ranks (equidistant).
        """
        print("Generating submission file...")

        # 1. Process Code Cells
        # In test set, code cells are provided in correct order.
        # We assign them equidistant ranks 0..1 to form the skeleton.
        mask_code = df_test["cell_type"] == "code"
        code_cells = df_test[mask_code].copy()

        code_list = []
        for nb_id, group in code_cells.groupby("id"):
            n = len(group)
            if n > 1:
                ranks = np.arange(n) / (n - 1.0)
            else:
                ranks = np.zeros(n)

            # Preserve original order (guaranteed by input file read order)
            temp = group[["id", "cell_id"]].copy()
            temp["rank"] = ranks
            code_list.append(temp)

        if code_list:
            df_code = pd.concat(code_list)
        else:
            df_code = pd.DataFrame(columns=["id", "cell_id", "rank"])

        # 2. Process Markdown Predictions
        df_md = pred_df[["id", "cell_id", "pred_rank"]].copy()
        df_md.rename(columns={"pred_rank": "rank"}, inplace=True)

        # 3. Merge and Sort
        all_cells = pd.concat([df_code, df_md], axis=0)

        submission_rows = []
        for nb_id, group in all_cells.groupby("id"):
            sorted_ids = group.sort_values("rank")["cell_id"].tolist()
            submission_rows.append({"id": nb_id, "cell_order": " ".join(sorted_ids)})

        df_submission = pd.DataFrame(submission_rows)

        # Save
        save_path = Config.SUBMISSION_PATH
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
