import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_notebook_cells


class LGBMRanker:
    """
    Explicit Alignment Regressor using LightGBM.
    Predicts the normalized position of markdown cells relative to code cells
    based on semantic alignment features.
    """

    def __init__(self):
        self.model = None

    def train(self, train_df, val_df):
        """
        Trains the LightGBM regressor.

        Args:
            train_df (pd.DataFrame): Training features and targets.
            val_df (pd.DataFrame): Validation features and targets.

        Returns:
            lgb.Booster: The trained model.
        """
        set_seed(Config.SEED)

        # Prepare features and targets
        features = Config.FEATURES
        target_col = "target"

        print(
            f"Training LightGBM with {len(train_df)} samples and {len(features)} features..."
        )

        X_train = train_df[features]
        y_train = train_df[target_col]
        X_val = val_df[features]
        y_val = val_df[target_col]

        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Callbacks for training control
        callbacks = [
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        # Train the model
        self.model = lgb.train(
            Config.LGBM_PARAMS,
            train_data,
            num_boost_round=Config.NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log best validation score
        # best_score structure: {'valid': {'rmse': 0.1234}, ...}
        if self.model.best_score:
            val_rmse = self.model.best_score["valid"]["rmse"]
            print(f"Best Validation RMSE: {val_rmse}")

        return self.model

    def predict(self, test_df):
        """
        Generates predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted normalized ranks.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        features = Config.FEATURES
        X_test = test_df[features]

        # Predict using the iteration with the best validation score
        predictions = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )
        return predictions

    def generate_submission(
        self, test_df, predictions, test_metadata_path=Config.TEST_METADATA_PATH
    ):
        """
        Reconstructs cell order based on predictions and saves the submission file.

        Args:
            test_df (pd.DataFrame): Test features dataframe (used to map predictions to cell_ids).
            predictions (np.ndarray): Predicted normalized ranks.
            test_metadata_path (str): Path to test metadata CSV.
        """
        print("Generating submission file...")

        # 1. Map predictions to (notebook_id, cell_id)
        # We assume test_df and predictions are aligned row-by-row
        test_df_mapped = test_df.copy()
        test_df_mapped["pred_rank"] = predictions

        # Create a lookup dictionary: nb_id -> {cell_id: pred_rank}
        # Grouping by ID first makes this significantly faster than iterating rows
        pred_lookup = {}
        for nb_id, group in test_df_mapped.groupby("id"):
            pred_lookup[nb_id] = dict(zip(group["cell_id"], group["pred_rank"]))

        # 2. Iterate through all test notebooks to reconstruct order
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

        df_test_meta = pd.read_csv(test_metadata_path)
        submission_rows = []

        for _, row in df_test_meta.iterrows():
            nb_id = row["id"]
            file_path = row["file_path"]

            # Retrieve cell structure
            # Code cells are assigned ranks 0, 1, 2...
            # Markdown cells are assigned rank -1
            nb_data = get_notebook_cells(nb_id, file_path)
            code_cells = nb_data["code_cells"]
            markdown_cells = nb_data["markdown_cells"]
            n_code = len(code_cells)

            # Check if we have predictions for this notebook
            if nb_id in pred_lookup:
                nb_preds = pred_lookup[nb_id]

                for md in markdown_cells:
                    cell_id = md["id"]
                    if cell_id in nb_preds:
                        # Convert normalized rank (0.0 - 1.0) to absolute rank
                        # e.g., if n_code=10 and pred=0.5, rank=5.0 (between code cells 4 and 5)
                        pred_pos = nb_preds[cell_id] * n_code
                        md["rank"] = pred_pos
                    else:
                        # Fallback: if a specific cell prediction is missing, place at end
                        md["rank"] = n_code
            else:
                # Fallback: Notebook was skipped in feature extraction (e.g., no code cells)
                # Place markdown cells sequentially after any existing code cells
                start_rank = n_code
                for i, md in enumerate(markdown_cells):
                    md["rank"] = start_rank + i

            # 3. Combine and Sort
            # Code cells have integer ranks, Markdown cells have float ranks
            all_cells = code_cells + markdown_cells
            all_cells.sort(key=lambda x: x["rank"])

            # Extract ordered IDs
            cell_order = " ".join([c["id"] for c in all_cells])
            submission_rows.append({"id": nb_id, "cell_order": cell_order})

        # 4. Save to CSV
        submission_df = pd.DataFrame(submission_rows)
        save_path = Config.SUBMISSION_FILE_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
