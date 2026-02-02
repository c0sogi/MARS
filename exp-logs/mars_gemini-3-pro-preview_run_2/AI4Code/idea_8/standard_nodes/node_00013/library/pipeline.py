import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed, kendall_tau_metric
from library.data_loader import DataLoader
from library.feature_engine import DualViewFeatureGenerator
from library.models import Stage1Ridge, Stage2LGBM


class OrderingPipeline:
    """
    Orchestrates the end-to-end training and inference workflow for the
    Dual-View Stacked Ranking model.
    """

    def __init__(self):
        self.feature_gen = DualViewFeatureGenerator()
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

    def fit(self, load_cached_data=True, debug=False, num_debug_samples=100):
        """
        Executes the training pipeline:
        1. Loads data.
        2. Generates features (TF-IDF, LSA, Anchors).
        3. Trains Stage 1 (Ridge) and generates OOF predictions.
        4. Trains Stage 2 (LightGBM) with Stacking.
        5. Evaluates on Validation set.
        """
        Config.setup()
        set_seed(Config.SEED)

        print("Pipeline: Loading Training and Validation Data...")
        train_md, train_code = DataLoader.load_data(
            "train",
            load_cached_data=load_cached_data,
            debug=debug,
            num_debug_samples=num_debug_samples,
        )
        val_md, val_code = DataLoader.load_data(
            "val",
            load_cached_data=load_cached_data,
            debug=debug,
            num_debug_samples=num_debug_samples,
        )

        print("Pipeline: Generating Features (LSA, Anchors, Metadata)...")
        # process_data handles caching of the dense feature dataframe
        train_feats = self.feature_gen.process_data(
            train_md, train_code, "train", load_cached_data=load_cached_data
        )
        val_feats = self.feature_gen.process_data(
            val_md, val_code, "val", load_cached_data=load_cached_data
        )

        # --- Stage 1 Preparation ---
        print("Pipeline: Preparing Stage 1 (TF-IDF) Data...")
        # Ensure vectorizer is loaded. If process_data loaded from cache, vectorizer might be None.
        if self.feature_gen.tfidf_vectorizer is None:
            # We use the training corpus to load/fit the vectorizer
            train_corpus = train_md["source"].fillna("").astype(str).tolist()
            self.feature_gen.tfidf_vectorizer = self.feature_gen._get_tfidf_model(
                corpus=train_corpus, load_cached=True
            )

        # Transform text to sparse matrices for Ridge
        train_text = train_md["source"].fillna("").astype(str).tolist()
        val_text = val_md["source"].fillna("").astype(str).tolist()

        X_train_tfidf = self.feature_gen.tfidf_vectorizer.transform(train_text)
        X_val_tfidf = self.feature_gen.tfidf_vectorizer.transform(val_text)

        y_train = train_md["norm_rank"].values
        # y_val is needed for metric calculation later, but not for Stage 1 prediction

        # --- Stage 1 Training ---
        print("Pipeline: Training Stage 1 (Ridge) & Generating OOF...")
        self.stage1_model.fit(X_train_tfidf, y_train)

        # Generate OOF predictions for stacking
        oof_preds = self.stage1_model.get_oof_predictions(X_train_tfidf, y_train)
        train_feats["ridge_pred"] = oof_preds

        # Generate predictions for validation set
        val_ridge_preds = self.stage1_model.predict(X_val_tfidf)
        val_feats["ridge_pred"] = val_ridge_preds

        # --- Stage 2 Training ---
        print("Pipeline: Training Stage 2 (LightGBM)...")

        # Select features
        exclude_cols = ["notebook_id", "cell_id", "source", "rank", "norm_rank"]
        feature_cols = [c for c in train_feats.columns if c not in exclude_cols]
        # Filter for numeric columns only
        feature_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(train_feats[c])
        ]

        print(f"Selected Features for Stage 2: {feature_cols}")

        self.stage2_model.fit(
            train_df=train_feats,
            val_df=val_feats,
            feature_cols=feature_cols,
            target_col="norm_rank",
        )

        # --- Validation Evaluation ---
        print("Pipeline: Evaluating on Validation Set...")
        # Predict final ranks
        val_final_preds = self.stage2_model.predict(val_feats, feature_cols)
        val_feats["pred_rank"] = val_final_preds

        # Post-process to get cell orders
        val_pred_df = self._post_process_sorting(val_feats, val_code)

        # Load Ground Truth for Validation
        df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)[["id", "cell_order"]]

        # Compute Metric
        score = kendall_tau_metric(val_pred_df, df_val_gt)
        print(f"Validation Kendall Tau Score: {score}")

    def predict(self, load_cached_data=True, debug=False, num_debug_samples=100):
        """
        Executes the inference pipeline:
        1. Loads Test data.
        2. Generates features.
        3. Predicts using Stage 1 & Stage 2 models.
        4. Sorts cells and generates submission file.
        """
        Config.setup()
        set_seed(Config.SEED)

        print("Pipeline: Loading Test Data...")
        test_md, test_code = DataLoader.load_data(
            "test",
            load_cached_data=load_cached_data,
            debug=debug,
            num_debug_samples=num_debug_samples,
        )

        print("Pipeline: Generating Test Features...")
        test_feats = self.feature_gen.process_data(
            test_md, test_code, "test", load_cached_data=load_cached_data
        )

        # --- Stage 1 Inference ---
        print("Pipeline: Stage 1 Inference...")
        if self.feature_gen.tfidf_vectorizer is None:
            self.feature_gen.tfidf_vectorizer = self.feature_gen._get_tfidf_model(
                corpus=None, load_cached=True
            )

        test_text = test_md["source"].fillna("").astype(str).tolist()
        X_test_tfidf = self.feature_gen.tfidf_vectorizer.transform(test_text)

        test_ridge_preds = self.stage1_model.predict(X_test_tfidf)
        test_feats["ridge_pred"] = test_ridge_preds

        # --- Stage 2 Inference ---
        print("Pipeline: Stage 2 Inference...")
        exclude_cols = ["notebook_id", "cell_id", "source", "rank", "norm_rank"]
        feature_cols = [c for c in test_feats.columns if c not in exclude_cols]
        feature_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(test_feats[c])
        ]

        final_ranks = self.stage2_model.predict(test_feats, feature_cols)
        test_feats["pred_rank"] = final_ranks

        # --- Submission Generation ---
        print("Pipeline: Post-processing and Saving Submission...")
        submission_df = self._post_process_sorting(test_feats, test_code)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    def _post_process_sorting(self, df_md_pred, df_code):
        """
        Combines code cells (fixed ranks) and markdown cells (predicted ranks)
        to produce the final sorted order string for each notebook.

        Args:
            df_md_pred (pd.DataFrame): Markdown cells with 'pred_rank' (0..1).
            df_code (pd.DataFrame): Code cells with 'rank' (integer 0..N).

        Returns:
            pd.DataFrame: Submission format ['id', 'cell_order'].
        """
        submission_rows = []

        # Group by notebook for efficient processing
        md_groups = df_md_pred.groupby("notebook_id")
        code_groups = df_code.groupby("notebook_id")

        all_ids = set(md_groups.groups.keys()) | set(code_groups.groups.keys())

        for nb_id in tqdm(all_ids, desc="Sorting Notebooks"):
            cells = []

            # Get counts to normalize code ranks
            n_code = (
                len(code_groups.get_group(nb_id)) if nb_id in code_groups.groups else 0
            )
            n_md = len(md_groups.get_group(nb_id)) if nb_id in md_groups.groups else 0
            total_cells = n_code + n_md

            if total_cells == 0:
                submission_rows.append({"id": nb_id, "cell_order": ""})
                continue

            # Process Code Cells
            if n_code > 0:
                c_df = code_groups.get_group(nb_id)
                for _, row in c_df.iterrows():
                    # Normalize integer rank to 0..1 range
                    # Logic matches training target generation: rank / (total - 1)
                    norm_rank = (
                        row["rank"] / (total_cells - 1) if total_cells > 1 else 0.0
                    )
                    cells.append((row["cell_id"], norm_rank))

            # Process Markdown Cells
            if n_md > 0:
                m_df = md_groups.get_group(nb_id)
                for _, row in m_df.iterrows():
                    # Use predicted rank directly
                    cells.append((row["cell_id"], row["pred_rank"]))

            # Sort by rank
            cells.sort(key=lambda x: x[1])

            # Create space-delimited string
            cell_order = " ".join([c[0] for c in cells])
            submission_rows.append({"id": nb_id, "cell_order": cell_order})

        return pd.DataFrame(submission_rows)
