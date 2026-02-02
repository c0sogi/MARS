import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.model import DSAPR
from library.dataset import NotebookSequenceDataset
from library.feature_extractor import EmbeddingGenerator


class InferenceEngine:
    """
    Manages the inference process for the DSAPR model, including feature generation,
    model prediction, and submission file creation.
    """

    def __init__(self, debug_limit: int = None):
        """
        Initialize the InferenceEngine.

        Args:
            debug_limit (int, optional): Limit the number of notebooks for debugging.
        """
        self.debug_limit = debug_limit
        self.device = Config.DEVICE
        set_seed(Config.SEED)

    def _ensure_features(self):
        """
        Ensures that the test set features (Parquet file) exist.
        If not, runs the EmbeddingGenerator.
        """
        if not os.path.exists(Config.TEST_CACHE_PATH):
            print(f"Test features not found at {Config.TEST_CACHE_PATH}. Generating...")
            gen = EmbeddingGenerator()
            gen.process_split("test", debug_limit=self.debug_limit)
        else:
            print(f"Test features found at {Config.TEST_CACHE_PATH}.")

    def run_inference(self):
        """
        Runs the model on the test dataset to generate relative rank predictions
        for markdown cells.

        Returns:
            dict: A mapping from (notebook_id, cell_id) to the predicted rank (float).
        """
        # 1. Ensure input features exist
        self._ensure_features()

        # 2. Load Dataset
        # NotebookSequenceDataset handles the caching of processed samples (.pt file)
        print("Loading test dataset...")
        dataset = NotebookSequenceDataset(
            split="test", load_cached_data=True, debug_limit=self.debug_limit
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device == "cuda"),
        )

        # 3. Load Model
        print("Loading model...")
        from library.model import LGBMModel

        model = LGBMModel()

        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {Config.MODEL_SAVE_PATH}. "
                "Please train the model before running inference."
            )

        model.load(Config.MODEL_SAVE_PATH)

        # 4. Prediction Loop
        print(f"Starting inference on {len(dataset)} markdown cells...")
        predictions = {}

        # Extract features for batch prediction
        X_test = []
        ids_list = []
        for sample in dataset.samples:
            X_test.append(sample["features"])
            ids_list.append((sample["nb_id"], sample["cell_id"]))

        X_test = np.array(X_test)
        preds = model.predict(X_test)

        for (nb_id, cell_id), pred in zip(ids_list, preds):
            predictions[(nb_id, cell_id)] = float(pred)

        print("Inference complete.")
        return predictions

    def generate_submission(self):
        """
        Orchestrates the inference process and generates the final submission CSV.
        """
        # 1. Get predictions for all markdown cells
        md_predictions = self.run_inference()

        # 2. Load structural data (cells) to reconstruct the notebook
        # We need the list of code cells and markdown cells for each notebook.
        # The parquet file contains this info.
        print("Loading test features for reconstruction...")
        df = pd.read_parquet(Config.TEST_CACHE_PATH)

        if self.debug_limit:
            # Filter dataframe to match the debug limit used in inference
            target_nb_ids = df["notebook_id"].unique()[: self.debug_limit]
            df = df[df["notebook_id"].isin(target_nb_ids)]

        # 3. Reconstruct Order
        print("Reconstructing cell orders...")

        # We rely on the order in the dataframe to establish the relative order of code cells.
        # Assuming feature extractor processed JSON cells in order.
        df["orig_idx"] = np.arange(len(df))

        grouped = df.groupby("notebook_id")
        submission_rows = []

        for nb_id, group in grouped:
            # Get Code cells (anchors) and Markdown cells (to be sorted)
            code_cells = group[group["cell_type"] == "code"].sort_values("orig_idx")
            md_cells = group[group["cell_type"] == "markdown"]

            ranked_cells = []

            # A. Assign ranks to Code Cells
            # Code cells are anchors. We assign them integer ranks: 0, 1, 2, ...
            code_ids = code_cells["cell_id"].tolist()
            n_code = len(code_ids)

            for i, cid in enumerate(code_ids):
                ranked_cells.append((cid, float(i)))

            # B. Assign ranks to Markdown Cells
            # Model predicts relative position y in [0, 1].
            # Absolute position ~= y * n_code.
            for _, row in md_cells.iterrows():
                cid = row["cell_id"]
                # Default to 0.0 if prediction missing (unlikely)
                pred_ratio = md_predictions.get((nb_id, cid), 0.0)

                # Calculate sorting score
                rank_score = pred_ratio * n_code
                ranked_cells.append((cid, rank_score))

            # C. Sort all cells by rank score
            ranked_cells.sort(key=lambda x: x[1])

            # Extract final ID list
            final_order = [x[0] for x in ranked_cells]

            submission_rows.append({"id": nb_id, "cell_order": " ".join(final_order)})

        # 4. Save Submission
        sub_df = pd.DataFrame(submission_rows)
        # Ensure correct column order
        sub_df = sub_df[["id", "cell_order"]]

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
