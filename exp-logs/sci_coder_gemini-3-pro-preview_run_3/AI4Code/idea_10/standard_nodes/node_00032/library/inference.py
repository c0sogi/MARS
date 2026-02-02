import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook, save_submission, set_seed
from library.feature_extractor import FeatureEngineer
from library.regressor import RankRegressor


class Predictor:
    """
    Orchestrates the inference pipeline for the Distribution-Aware Semantic Regressor.
    """

    def __init__(self):
        """
        Initializes the Predictor with the necessary components.
        """
        self.feature_engineer = FeatureEngineer()
        self.regressor = RankRegressor()
        set_seed(Config.SEED)

    def predict_test_set(
        self, test_metadata_path=Config.TEST_METADATA_PATH, load_cached_data=True
    ):
        """
        Generates predictions for the test set.

        Args:
            test_metadata_path (str): Path to the test metadata CSV.
            load_cached_data (bool): Whether to use cached feature files.

        Returns:
            pd.DataFrame: DataFrame containing ['id', 'cell_id', 'pred']
        """
        # 1. Extract Features
        # The FeatureEngineer handles caching internally via the save_path argument
        print("Extracting features for test set...")
        df_test_features = self.feature_engineer.extract_features(
            metadata_path=test_metadata_path,
            save_path=Config.TEST_FEATURES_PATH,
            load_cached_data=load_cached_data,
        )

        if df_test_features.empty:
            raise ValueError("No features extracted for the test set.")

        # 2. Generate Predictions
        print("Generating predictions using RankRegressor...")
        predictions = self.regressor.predict(df_test_features)

        # 3. Format Output
        # We need to map predictions back to specific markdown cells
        output_df = df_test_features[["id", "cell_id"]].copy()
        output_df["pred"] = predictions

        return output_df


def generate_submission(
    predictions_df,
    test_metadata_path=Config.TEST_METADATA_PATH,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Reconstructs the cell order based on predictions and saves the submission file.

    Args:
        predictions_df (pd.DataFrame): DataFrame with ['id', 'cell_id', 'pred'].
        test_metadata_path (str): Path to test metadata to retrieve file paths.
        output_path (str): Path to save the submission CSV.
    """
    print("Reconstructing cell orders and generating submission...")
    set_seed(Config.SEED)

    # Load test metadata to get file paths
    df_meta = pd.read_csv(test_metadata_path)

    # Create a dictionary of predictions for fast lookup: {notebook_id: {cell_id: score}}
    pred_dict = {}
    # Group by notebook id first to avoid iterating the whole dataframe repeatedly
    for nb_id, group in predictions_df.groupby("id"):
        pred_dict[nb_id] = dict(zip(group["cell_id"], group["pred"]))

    submission_rows = []

    # Iterate through each test notebook
    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            code_cells, md_cells = read_notebook(file_path)
        except Exception:
            # If read fails, provide a fallback (empty string) or skip
            # In a competition context, we must provide a prediction for every ID.
            # We will assume empty lists if read fails.
            code_cells = []
            md_cells = []

        n_code = len(code_cells)
        nb_preds = pred_dict.get(nb_id, {})

        cells_with_ranks = []

        # 1. Assign ranks to Code Cells
        # Code cells act as fixed anchors.
        # Code cell at index i (0-based) is effectively at position i + 0.5 relative to the "slots"
        # created by the regression target definition.
        for i, cell in enumerate(code_cells):
            # Rank = index + 0.5 ensures it sits between integer boundaries
            rank = i + 0.5
            cells_with_ranks.append((rank, cell["id"]))

        # 2. Assign ranks to Markdown Cells
        # The regressor predicts a ratio y in [0, 1].
        # The relative rank is y * n_code.
        for cell in md_cells:
            cell_id = cell["id"]
            # Get predicted ratio, default to end of notebook (1.0) if missing
            pred_ratio = nb_preds.get(cell_id, 1.0)

            # Calculate rank
            rank = pred_ratio * n_code
            cells_with_ranks.append((rank, cell_id))

        # 3. Sort by Rank
        cells_with_ranks.sort(key=lambda x: x[0])

        # 4. Extract ordered IDs
        ordered_ids = [c[1] for c in cells_with_ranks]
        cell_order_str = " ".join(ordered_ids)

        submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

    # Create Submission DataFrame
    df_submission = pd.DataFrame(submission_rows)

    # Save
    save_submission(df_submission, filename=output_path)
    print(f"Submission saved to {output_path}")
