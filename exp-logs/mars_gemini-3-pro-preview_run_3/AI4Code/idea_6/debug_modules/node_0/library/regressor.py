import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed, read_notebook


class RankRegressor:
    """
    Wraps a LightGBM Regressor to predict the normalized rank of markdown cells.
    """

    def __init__(self):
        self.params = Config.get_lgbm_params()
        self.model = None
        self.features = [
            "n_code",
            "md_len",
            "tv_sim_max",
            "tv_best_loc",
            "tv_com",
            "cv_sim_max",
            "cv_best_loc",
            "cv_com",
        ]
        self.target = "target"
        set_seed(Config.SEED)

    def train(self, df_train, df_val):
        """
        Trains the LightGBM model using the provided training and validation DataFrames.

        Args:
            df_train (pd.DataFrame): Training features and targets.
            df_val (pd.DataFrame): Validation features and targets.
        """
        print(
            f"Training RankRegressor with {len(df_train)} train and {len(df_val)} val samples..."
        )

        X_train = df_train[self.features]
        y_train = df_train[self.target]
        X_val = df_val[self.features]
        y_val = df_val[self.target]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        # Note: We suppress verbose output in params but use the callback for logging
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate and print full precision metric
        preds_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        mse = np.mean((y_val - preds_val) ** 2)
        print(f"Validation MSE: {mse}")

        # Save the model
        model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

    def predict(self, df_test):
        """
        Generates predictions for the test set.

        Args:
            df_test (pd.DataFrame): Test features.

        Returns:
            np.array: Predicted normalized ranks.
        """
        if self.model is None:
            model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
            if os.path.exists(model_path):
                self.model = lgb.Booster(model_file=model_path)
            else:
                raise ValueError("Model not trained and no saved model found.")

        X_test = df_test[self.features]
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)


def generate_submission(model, df_test_features, test_metadata_path):
    """
    Generates the submission file by reconstructing the cell order.

    Args:
        model (RankRegressor): Trained model instance.
        df_test_features (pd.DataFrame): DataFrame containing test features.
        test_metadata_path (str): Path to the test metadata CSV.
    """
    print("Generating submission...")

    # 1. Generate Predictions
    preds = model.predict(df_test_features)
    df_test_features = df_test_features.copy()
    df_test_features["pred_rank"] = preds

    # 2. Load Test Metadata to get the list of all notebooks
    df_test_meta = pd.read_csv(test_metadata_path)

    submission_rows = []

    # Group predictions by notebook ID for efficient access
    preds_by_id = df_test_features.groupby("id")

    # 3. Reconstruct Order for each notebook
    for _, row in df_test_meta.iterrows():
        notebook_id = row["id"]
        file_path = row["file_path"]

        try:
            data = read_notebook(file_path)
            cell_types = data.get("cell_type", {})
        except Exception:
            cell_types = {}

        # Identify all cells
        # In test JSONs, the order of keys implies the relative order of code cells
        all_cells = list(cell_types.keys())
        code_cells = [c for c in all_cells if cell_types[c] == "code"]

        # Get predictions for markdown cells in this notebook
        if notebook_id in preds_by_id.groups:
            nb_preds = preds_by_id.get_group(notebook_id)
            # Map cell_id -> predicted normalized rank
            md_pred_map = dict(zip(nb_preds["cell_id"], nb_preds["pred_rank"]))
        else:
            md_pred_map = {}

        # Create a list of (position, cell_id) tuples for sorting
        ranking_list = []

        # Assign positions to Code Cells
        # Code cell at index i is effectively at position i + 0.5 (acting as a pivot)
        # This ensures that a markdown predicted at '0' comes before code '0',
        # and '1' comes after code '0' (which is at 0.5) but before code '1' (at 1.5).
        for i, cell_id in enumerate(code_cells):
            ranking_list.append((i + 0.5, cell_id))

        # Assign positions to Markdown Cells
        # Position = predicted_normalized_rank * number_of_code_cells
        n_code = len(code_cells)

        # Handle markdown cells
        for cell_id in all_cells:
            if cell_types[cell_id] == "markdown":
                if cell_id in md_pred_map:
                    pred = md_pred_map[cell_id]
                    # Scale prediction to the number of code cells
                    pos = pred * n_code
                    ranking_list.append((pos, cell_id))
                else:
                    # Fallback: place at end if no prediction (should not happen if features exist)
                    ranking_list.append((n_code + 1.0, cell_id))

        # Sort by position
        ranking_list.sort(key=lambda x: x[0])

        # Extract ordered cell IDs
        final_order = [x[1] for x in ranking_list]

        submission_rows.append({"id": notebook_id, "cell_order": " ".join(final_order)})

    # 4. Save Submission
    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_submission)} rows."
    )
