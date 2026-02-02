import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from library.config import Config
from library.utils import kendall_tau, format_submission
from library.data_processing import NotebookProcessor
from library.feature_extraction import FeatureEngine
from library.modeling import Stage1Ridge, Stage2LGBM


class NotebookOrderingWorkflow:
    """
    Orchestrates the end-to-end training and inference pipelines for the
    Stacked Hybrid Ranking with Multi-View Instance-Based Neighborhoods.
    """

    def __init__(self):
        Config.set_seeds()
        Config.setup()

        self.processor = NotebookProcessor()
        self.feature_engine = FeatureEngine()
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

    def train(self):
        """
        Executes the training pipeline:
        1. Load Data
        2. Fit Vectorizers
        3. Stage 1: Generate OOF preds (CV) and fit final Ridge
        4. Stage 2: Generate features, add Stage 1 preds, fit LightGBM
        5. Validate
        """
        print("=== Starting Training Pipeline ===")

        # ----------------------------------------------------------------
        # 1. Load Data
        # ----------------------------------------------------------------
        df_train = self.processor.load_dataset(split="train", load_cached_data=True)
        df_val = self.processor.load_dataset(split="val", load_cached_data=True)

        # Filter for markdown cells for training targets
        train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(drop=True)
        val_md = df_val[df_val["cell_type"] == "markdown"].reset_index(drop=True)

        y_train = train_md["pct_rank"].values
        y_val = val_md["pct_rank"].values

        # ----------------------------------------------------------------
        # 2. Global Vectorization
        # ----------------------------------------------------------------
        # Fit TF-IDF and SVD on training data
        self.feature_engine.fit_global_vectorizers(df_train, load_cached_models=True)

        # ----------------------------------------------------------------
        # 3. Stage 1: Ridge Regression (Stacking)
        # ----------------------------------------------------------------
        print("--- Stage 1: Ridge Regression Stacking ---")

        # A. Generate Out-Of-Fold (OOF) Predictions for Training Data
        oof_preds_path = os.path.join(Config.WORKING_DIR, "stage1_oof_preds.npy")

        if os.path.exists(oof_preds_path):
            print(f"Loading OOF predictions from {oof_preds_path}...")
            oof_preds = np.load(oof_preds_path)
        else:
            print("Generating OOF predictions via 5-Fold GroupCV...")
            # We need sparse features for the full train markdown set
            X_train_sparse = self.feature_engine.get_stage1_features(train_md)

            groups = train_md["ancestor_id"]
            gkf = GroupKFold(n_splits=5)
            oof_preds = np.zeros(len(train_md))

            # Use a temporary Ridge for CV to avoid overwriting the main model wrapper
            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(X_train_sparse, y_train, groups)
            ):
                X_fold_train = X_train_sparse[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train_sparse[val_idx]

                model = Ridge(alpha=Config.RIDGE_ALPHA, random_state=Config.SEED)
                model.fit(X_fold_train, y_fold_train)
                oof_preds[val_idx] = model.predict(X_fold_val)

            np.save(oof_preds_path, oof_preds)
            print(f"Saved OOF predictions to {oof_preds_path}")

        # B. Fit Final Stage 1 Model on Full Training Data
        # This saves the model to disk for Inference usage
        X_train_full_sparse = self.feature_engine.get_stage1_features(train_md)
        self.stage1_model.fit(X_train_full_sparse, y_train)

        # C. Generate Stage 1 Predictions for Validation Data
        X_val_sparse = self.feature_engine.get_stage1_features(val_md)
        val_stage1_preds = self.stage1_model.predict(X_val_sparse)

        # ----------------------------------------------------------------
        # 4. Stage 2: LightGBM Refinement
        # ----------------------------------------------------------------
        print("--- Stage 2: LightGBM Refinement ---")

        # A. Generate Multi-View Instance Features
        # Note: get_stage2_features processes the whole dataframe (code+md)
        # but returns features only for markdown cells (aligned with train_md)
        X_train_lgbm = self.feature_engine.get_stage2_features(
            df_train, split="train", load_cached_data=True
        )
        X_val_lgbm = self.feature_engine.get_stage2_features(
            df_val, split="val", load_cached_data=True
        )

        # B. Add Stage 1 Predictions as Feature
        X_train_lgbm["ridge_pred"] = oof_preds
        X_val_lgbm["ridge_pred"] = val_stage1_preds

        # Drop non-feature columns for training
        drop_cols = ["id", "cell_id"]
        features_train = X_train_lgbm.drop(columns=drop_cols, errors="ignore")
        features_val = X_val_lgbm.drop(columns=drop_cols, errors="ignore")

        # C. Train LightGBM
        self.stage2_model.fit(features_train, y_train, features_val, y_val)

        # ----------------------------------------------------------------
        # 5. Validation Scoring
        # ----------------------------------------------------------------
        print("--- Final Validation ---")
        # Predict on validation set
        final_val_preds = self.stage2_model.predict(features_val)

        # Reconstruct orders for Kendall Tau calculation
        # We need to sort markdown cells based on predictions and interleave with code cells
        val_ids = val_md["id"].unique()

        # Prepare predictions dataframe
        pred_orders = []

        # Create a lookup for predictions
        # val_md has the same order as final_val_preds
        val_md_preds = val_md.copy()
        val_md_preds["pred_rank"] = final_val_preds

        # Group by notebook to sort
        # We also need the code cells for these notebooks to reconstruct the full order
        df_val_code = df_val[df_val["cell_type"] == "code"]

        for nb_id in val_ids:
            # Get code cells
            nb_code = df_val_code[df_val_code["id"] == nb_id].copy()
            # Get markdown cells with predictions
            nb_md = val_md_preds[val_md_preds["id"] == nb_id].copy()

            # Assign scores for sorting
            # Code cells: equidistant anchors in [0, 1]
            n_code = len(nb_code)
            if n_code > 0:
                nb_code["score"] = np.linspace(0, 1, n_code)
            else:
                nb_code["score"] = []  # Should not happen often

            # Markdown cells: predicted rank
            nb_md["score"] = nb_md["pred_rank"]

            # Concatenate
            nb_full = pd.concat([nb_code, nb_md])

            # Sort
            nb_full = nb_full.sort_values("score")

            # Extract ID list
            cell_order = " ".join(nb_full["cell_id"].tolist())
            pred_orders.append(cell_order)

        pred_df = pd.DataFrame({"id": val_ids, "cell_order": pred_orders})

        # Load Ground Truth
        gt_df = pd.read_csv(Config.VAL_METADATA_PATH)[["id", "cell_order"]]

        score = kendall_tau(gt_df, pred_df)
        print(f"Validation Kendall Tau: {score}")

    def predict(self):
        """
        Executes the inference pipeline:
        1. Load Test Data
        2. Generate Stage 1 Preds
        3. Generate Stage 2 Features + Preds
        4. Sort and Format Submission
        """
        print("=== Starting Inference Pipeline ===")

        # 1. Load Data
        df_test = self.processor.load_dataset(split="test", load_cached_data=True)

        # Filter markdown for feature extraction input
        test_md = df_test[df_test["cell_type"] == "markdown"].reset_index(drop=True)

        # 2. Stage 1 Predictions
        print("Generating Stage 1 (Ridge) predictions...")
        X_test_sparse = self.feature_engine.get_stage1_features(test_md)
        stage1_preds = self.stage1_model.predict(X_test_sparse)

        # 3. Stage 2 Features & Predictions
        print("Generating Stage 2 features...")
        X_test_lgbm = self.feature_engine.get_stage2_features(
            df_test, split="test", load_cached_data=True
        )

        # Add Stage 1 preds
        X_test_lgbm["ridge_pred"] = stage1_preds

        # Prepare features
        drop_cols = ["id", "cell_id"]
        features_test = X_test_lgbm.drop(columns=drop_cols, errors="ignore")

        print("Generating Final (LightGBM) predictions...")
        final_preds = self.stage2_model.predict(features_test)

        # 4. Post-Processing (Anchor-Based Sorting)
        print("Sorting cells and formatting submission...")

        # Attach predictions to markdown metadata
        test_md_preds = test_md.copy()
        test_md_preds["pred_rank"] = final_preds

        # Get code cells
        df_test_code = df_test[df_test["cell_type"] == "code"]

        test_ids = df_test["id"].unique()
        submission_ids = []
        submission_orders = []

        for nb_id in test_ids:
            # Get code cells
            nb_code = df_test_code[df_test_code["id"] == nb_id].copy()
            # Get markdown cells
            nb_md = test_md_preds[test_md_preds["id"] == nb_id].copy()

            # Assign scores
            n_code = len(nb_code)
            if n_code > 0:
                # Code cells are anchors at fixed percentiles
                nb_code["score"] = np.linspace(0, 1, n_code)
            else:
                nb_code["score"] = []

            nb_md["score"] = nb_md["pred_rank"]

            # Merge and Sort
            nb_full = pd.concat([nb_code, nb_md])
            nb_full = nb_full.sort_values("score")

            cell_order = " ".join(nb_full["cell_id"].tolist())

            submission_ids.append(nb_id)
            submission_orders.append(cell_order)

        # 5. Save Submission
        sub_df = pd.DataFrame({"id": submission_ids, "cell_order": submission_orders})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_workflow():
    workflow = NotebookOrderingWorkflow()

    # Train models
    workflow.train()

    # Generate submission
    workflow.predict()


if __name__ == "__main__":
    run_workflow()
