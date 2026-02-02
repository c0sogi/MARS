import os
import numpy as np
import pandas as pd
from scipy import sparse
from library.config import Config
from library.utils import kendall_tau
from library.data_processing import NotebookLoader
from library.feature_engineering import AnchorFeatureGenerator
from library.models import Stage1Ridge, Stage2LGBM


class HybridRankingPipeline:
    """
    Orchestrates the Stacked Hybrid Ranking pipeline with Multi-View Anchoring.
    """

    def __init__(self):
        Config.set_seed(Config.RANDOM_SEED)
        self.loader = NotebookLoader()
        self.feature_gen = AnchorFeatureGenerator()
        self.stage1 = Stage1Ridge()
        self.stage2 = Stage2LGBM()

    def _get_tfidf_matrix(self, df, split_name="train"):
        """
        Extracts TF-IDF matrix for markdown cells in the dataframe.
        Assumes vectorizer is already fit (handled by feature_gen).
        """
        # Filter for markdown cells to ensure alignment with feature generation
        md_mask = df["cell_type"] == "markdown"
        md_sources = df.loc[md_mask, "source_clean"].fillna("").tolist()

        # Transform using the pipeline inside feature_gen
        # We ignore the SVD output here as we only need TF-IDF for Ridge
        tfidf, _ = self.feature_gen.pipeline.transform(md_sources)
        return tfidf

    def _prepare_stacking_features(self, ridge_preds, anchor_features):
        """
        Combines Stage 1 predictions with Anchor/SVD features for Stage 2.
        """
        # Drop metadata columns to keep only numerical features
        meta_cols = ["notebook_id", "cell_id", "rank", "pct_rank"]
        feat_cols = [c for c in anchor_features.columns if c not in meta_cols]

        X_anchors = anchor_features[feat_cols].values

        # Reshape Ridge predictions to (N, 1)
        X_ridge = ridge_preds.reshape(-1, 1)

        # Stack horizontally
        return np.hstack([X_ridge, X_anchors])

    def train(self, load_cached_data=True):
        """
        Executes the full training loop: Data Loading -> Feature Gen -> Stage 1 -> Stage 2 -> Validation.
        """
        print("Starting Pipeline Training...")

        # 1. Load Data
        df_train = self.loader.get_flattened_data(
            "train", load_cached_data=load_cached_data
        )
        df_val = self.loader.get_flattened_data(
            "val", load_cached_data=load_cached_data
        )

        # 2. Feature Generation
        # This step fits the vectorizers on 'train' and computes anchor features
        print("Generating Training Features...")
        feats_train = self.feature_gen.generate_features(
            df_train, "train", load_cached_data=load_cached_data
        )

        print("Generating Validation Features...")
        feats_val = self.feature_gen.generate_features(
            df_val, "val", load_cached_data=load_cached_data
        )

        # 3. Prepare Stage 1 Inputs (TF-IDF)
        print("Preparing TF-IDF Matrices...")
        X_train_tfidf = self._get_tfidf_matrix(df_train, "train")
        X_val_tfidf = self._get_tfidf_matrix(df_val, "val")

        # Extract Targets (Normalized Ranks)
        y_train = feats_train["pct_rank"].values
        y_val = feats_val["pct_rank"].values

        # 4. Stage 1: Ridge Regression
        print("Generating Stage 1 OOF Predictions...")
        # Get unbiased predictions for stacking
        train_ridge_preds = self.stage1.get_oof_predictions(
            X_train_tfidf, y_train, n_splits=5, load_cached_data=load_cached_data
        )

        # Fit Ridge on full training data for inference/validation usage
        self.stage1.fit(X_train_tfidf, y_train)
        self.stage1.save()

        # Predict on Validation set
        val_ridge_preds = self.stage1.predict(X_val_tfidf)

        # 5. Stage 2: LightGBM
        print("Preparing Stage 2 Features...")
        X_stack_train = self._prepare_stacking_features(train_ridge_preds, feats_train)
        X_stack_val = self._prepare_stacking_features(val_ridge_preds, feats_val)

        # Train with Early Stopping
        self.stage2.fit(X_stack_train, y_train, X_val=X_stack_val, y_val=y_val)
        self.stage2.save()

        # 6. Validation Metric (Kendall Tau)
        print("Computing Validation Metrics...")

        # Predict ranks using the trained Stage 2 model
        val_lgbm_preds = self.stage2.predict(X_stack_val)

        # Create a map for fast lookup: cell_id -> predicted_rank
        pred_map = dict(zip(feats_val["cell_id"], val_lgbm_preds))

        val_scores = []

        # Group by notebook to reconstruct order
        grouped = df_val.groupby("notebook_id")

        for nb_id, group in grouped:
            # Ground Truth Order
            gt_order = group.sort_values("rank")["cell_id"].tolist()

            # Reconstruct Predicted Order
            cells = []

            # Code cells: Fixed equidistant ranks [0.0, 1.0]
            # They are assumed to be in correct relative order in the dataframe
            code_cells = group[group["cell_type"] == "code"]
            n_code = len(code_cells)
            if n_code > 0:
                code_ranks = np.linspace(0.0, 1.0, n_code)
                for (idx, row), r in zip(code_cells.iterrows(), code_ranks):
                    cells.append((row["cell_id"], r))

            # Markdown cells: Predicted ranks
            md_cells = group[group["cell_type"] == "markdown"]
            for idx, row in md_cells.iterrows():
                cid = row["cell_id"]
                # Default to 0.5 if for some reason missing (should not happen)
                r = pred_map.get(cid, 0.5)
                cells.append((cid, r))

            # Sort by rank
            cells.sort(key=lambda x: x[1])
            pred_order = [c[0] for c in cells]

            # Compute Metric
            score = kendall_tau(gt_order, pred_order)
            val_scores.append(score)

        avg_score = np.mean(val_scores)
        print(f"Validation Kendall Tau: {avg_score}")

        return avg_score

    def predict(self, load_cached_data=True):
        """
        Executes the inference pipeline on the test set and generates submission.
        """
        print("Starting Inference...")

        # 1. Load Test Data
        df_test = self.loader.get_flattened_data(
            "test", load_cached_data=load_cached_data
        )

        # 2. Feature Generation
        print("Generating Test Features...")
        # Uses vectorizers fitted during training
        feats_test = self.feature_gen.generate_features(
            df_test, "test", load_cached_data=load_cached_data
        )

        # 3. Stage 1 Inference
        print("Stage 1 Inference...")
        X_test_tfidf = self._get_tfidf_matrix(df_test, "test")

        # Ensure model is loaded
        if not hasattr(self.stage1.model, "coef_"):
            self.stage1.load()

        test_ridge_preds = self.stage1.predict(X_test_tfidf)

        # 4. Stage 2 Inference
        print("Stage 2 Inference...")
        X_stack_test = self._prepare_stacking_features(test_ridge_preds, feats_test)

        # Ensure model is loaded
        try:
            # Check for booster attribute to verify fit
            self.stage2.model.booster_
        except:
            self.stage2.load()

        test_lgbm_preds = self.stage2.predict(X_stack_test)

        # 5. Post-processing & Submission
        print("Generating Submission...")

        pred_map = dict(zip(feats_test["cell_id"], test_lgbm_preds))
        submission_rows = []

        grouped = df_test.groupby("notebook_id")

        for nb_id, group in grouped:
            cells = []

            # Code Cells: Fixed Ranks
            code_cells = group[group["cell_type"] == "code"]
            n_code = len(code_cells)
            if n_code > 0:
                code_ranks = np.linspace(0.0, 1.0, n_code)
                for (idx, row), r in zip(code_cells.iterrows(), code_ranks):
                    cells.append((row["cell_id"], r))

            # Markdown Cells: Predicted Ranks
            md_cells = group[group["cell_type"] == "markdown"]
            for idx, row in md_cells.iterrows():
                cid = row["cell_id"]
                r = pred_map.get(cid, 0.5)
                cells.append((cid, r))

            # Sort and format
            cells.sort(key=lambda x: x[1])
            pred_order = " ".join([c[0] for c in cells])

            submission_rows.append({"id": nb_id, "cell_order": pred_order})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
