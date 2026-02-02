import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.data_loader import NotebookDataLoader
from library.feature_extraction import FeaturePipeline
from library.model_definitions import Level1Ridge, Level2GBM
from library.utils import format_submission


class PipelineManager:
    """
    Orchestrates the Stacked Hybrid Linear-Tree Ranking pipeline.
    Manages data loading, feature extraction, two-stage model training, and inference.
    """

    def __init__(self):
        self.loader = NotebookDataLoader(debug=Config.DEBUG)
        self.feature_pipeline = FeaturePipeline()
        self.ridge_model = Level1Ridge()
        self.lgbm_model = Level2GBM()

    def train_stacking_ensemble(self, load_cached_data=True):
        """
        Executes the full training pipeline:
        1. Loads Train and Validation data.
        2. Fits/Loads Feature Extraction (TF-IDF, SVD).
        3. Level 1: Generates Out-Of-Fold (OOF) predictions using Ridge Regression (K-Fold).
        4. Level 1: Retrains Ridge on full training data.
        5. Level 2: Constructs stacked features (OOF + LSA + Meta).
        6. Level 2: Trains LightGBM with Early Stopping on Validation set.
        """
        print("=== Starting Stacking Ensemble Training ===")

        # 1. Load Data
        df_train_md, df_train_nb = self.loader.load_data(
            split="train", load_cached_data=load_cached_data
        )
        df_val_md, df_val_nb = self.loader.load_data(
            split="val", load_cached_data=load_cached_data
        )

        # 2. Feature Pipeline
        # Fits on training data (or loads from cache)
        self.feature_pipeline.fit(
            df_train_md, df_train_nb, load_cached_data=load_cached_data
        )

        # 3. Level 1: Ridge Regression (OOF Generation)
        print(f"Generating Level 1 OOF predictions using {Config.N_FOLDS}-Fold CV...")

        # Transform full training text to sparse matrix
        X_train_l1 = self.feature_pipeline.transform_level1(df_train_md)
        y_train = df_train_md["rank"].values
        groups = df_train_md["id"].values  # Group by Notebook ID

        # Initialize OOF array
        oof_preds = np.zeros(len(df_train_md))

        kfold = GroupKFold(n_splits=Config.N_FOLDS)

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(X_train_l1, y_train, groups=groups)
        ):
            # Split data
            X_fold_train = X_train_l1[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_l1[val_idx]

            # Train temporary Ridge model for this fold
            fold_model = Level1Ridge()
            fold_model.fit(X_fold_train, y_fold_train)

            # Predict on hold-out
            preds = fold_model.predict(X_fold_val)
            oof_preds[val_idx] = preds

            # Print fold score (MAE)
            fold_mae = np.mean(np.abs(preds - y_train[val_idx]))
            print(f"  Fold {fold + 1} MAE: {fold_mae}")

        # 4. Level 1: Final Ridge Training
        # Retrain on all data for inference usage
        print("Retraining Level 1 Ridge on full training set...")
        self.ridge_model.fit(X_train_l1, y_train)
        self.ridge_model.save(Config.MODEL_RIDGE_PATH)

        # 5. Level 2: Feature Construction
        print("Constructing Level 2 Training Features...")
        # Train features include OOF predictions
        df_train_l2 = self.feature_pipeline.transform_level2(
            df_train_md, df_train_nb, level1_preds=oof_preds
        )

        print("Constructing Level 2 Validation Features...")
        # Validation features use predictions from the Final Ridge model
        # (Simulating test-time behavior)
        X_val_l1 = self.feature_pipeline.transform_level1(df_val_md)
        val_l1_preds = self.ridge_model.predict(X_val_l1)

        df_val_l2 = self.feature_pipeline.transform_level2(
            df_val_md, df_val_nb, level1_preds=val_l1_preds
        )

        y_val = df_val_md["rank"].values

        # 6. Level 2: LightGBM Training
        print("Training Level 2 LightGBM...")
        self.lgbm_model.fit(
            X_train=df_train_l2, y_train=y_train, X_val=df_val_l2, y_val=y_val
        )
        self.lgbm_model.save(Config.MODEL_LGBM_PATH)

        print("=== Training Complete ===")

    def predict_and_sort(self, load_cached_data=True):
        """
        Runs the inference pipeline on the Test set:
        1. Loads Test data and Models.
        2. Generates Rank predictions (L1 -> L2).
        3. Applies Anchor-Based Sorting (interleaving Code and Markdown).
        4. Generates submission CSV.
        """
        print("=== Starting Inference and Sorting ===")

        # 1. Load Data
        df_test_md, df_test_nb = self.loader.load_data(
            split="test", load_cached_data=load_cached_data
        )

        # 2. Load Models (if not already in memory)
        try:
            self.ridge_model.load(Config.MODEL_RIDGE_PATH)
            self.lgbm_model.load(Config.MODEL_LGBM_PATH)
            # Feature pipeline needs to be fitted/loaded to transform
            self.feature_pipeline.fit(df_test_md, df_test_nb, load_cached_data=True)
        except FileNotFoundError as e:
            print(f"Error loading models: {e}. Please run training first.")
            return

        # 3. Generate Predictions
        print("Predicting Markdown Ranks...")

        # Level 1 Prediction
        X_test_l1 = self.feature_pipeline.transform_level1(df_test_md)
        l1_preds = self.ridge_model.predict(X_test_l1)

        # Level 2 Prediction
        df_test_l2 = self.feature_pipeline.transform_level2(
            df_test_md, df_test_nb, level1_preds=l1_preds
        )
        final_ranks = self.lgbm_model.predict(df_test_l2)

        # Add predictions to dataframe for easy processing
        df_test_md["pred_rank"] = final_ranks

        # 4. Anchor-Based Sorting
        print("Applying Anchor-Based Sorting...")

        submission_ids = []
        submission_orders = []

        # Group markdown predictions by notebook
        md_grouped = df_test_md.groupby("id")

        # Iterate over notebooks in the test set (using df_test_nb for structure)
        for _, row in df_test_nb.iterrows():
            nb_id = row["id"]
            code_cell_ids_str = row["code_cell_ids"]

            # Get Code Cells (Anchors)
            if pd.isna(code_cell_ids_str) or code_cell_ids_str == "":
                code_cells = []
            else:
                code_cells = code_cell_ids_str.split()

            n_code = len(code_cells)

            # Assign equidistant ranks to code cells [0.0, ..., 1.0]
            cells_with_ranks = []

            if n_code > 0:
                if n_code == 1:
                    # Single code cell anchor at 0.0
                    cells_with_ranks.append((code_cells[0], 0.0))
                else:
                    for i, cell_id in enumerate(code_cells):
                        r = i / (n_code - 1)
                        cells_with_ranks.append((cell_id, r))

            # Get Markdown Cells and their predicted ranks
            if nb_id in md_grouped.groups:
                md_group = md_grouped.get_group(nb_id)
                for _, md_row in md_group.iterrows():
                    cells_with_ranks.append((md_row["cell_id"], md_row["pred_rank"]))

            # Sort all cells by rank
            cells_with_ranks.sort(key=lambda x: x[1])

            # Extract ordered cell IDs
            sorted_order = [c[0] for c in cells_with_ranks]

            submission_ids.append(nb_id)
            submission_orders.append(" ".join(sorted_order))

        # 5. Save Submission
        format_submission(submission_ids, submission_orders, Config.SUBMISSION_PATH)
        print("=== Inference Complete ===")
