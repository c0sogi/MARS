import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import DCCodeBERT
from library.data_loader import get_dataloader


class Predictor:
    """
    Handles inference and submission generation for the DC-CodeBERT model.
    """

    def __init__(self, model_path=None):
        """
        Initialize the Predictor with a trained model.

        Args:
            model_path (str, optional): Path to the model checkpoint.
                                        Defaults to Config.BEST_MODEL_PATH.
        """
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        self.model_path = model_path if model_path else Config.BEST_MODEL_PATH

        print(f"Initializing Predictor on {self.device}...")
        self.model = DCCodeBERT().to(self.device)

        # Load the trained weights
        # We use os.path.basename because load_checkpoint expects just the filename
        # relative to WORKING_DIR, or we can pass the logic to handle paths.
        # The provided utils.load_checkpoint constructs path via Config.WORKING_DIR.
        # We assume the file is in Config.WORKING_DIR.
        filename = os.path.basename(self.model_path)
        score = load_checkpoint(self.model, filename=filename, device=self.device)
        print(f"Loaded model from {filename} (Validation Score: {score})")

        self.model.eval()

    def _load_code_cells(self, features_path):
        """
        Loads the mapping of notebook IDs to their ordered code cell IDs from the features file.

        Args:
            features_path (str): Path to the parquet file containing features.

        Returns:
            dict: {nb_id: [code_cell_id_1, code_cell_id_2, ...]}
        """
        print(f"Loading code cell map from {features_path}...")
        try:
            # Read minimal columns to save memory
            df = pd.read_parquet(
                features_path, columns=["id", "cell_id", "cell_type", "rank"]
            )
        except Exception as e:
            print(f"Error reading parquet: {e}")
            return {}

        # Filter for code cells
        df_code = df[df["cell_type"] == "code"].copy()

        # In the test set, 'rank' might be -1, but the FeatureExtractor processes
        # the JSON sequentially, so the DataFrame index order preserves the file order.
        # If ranks are available (e.g. reusing this on val set), we sort by them.
        if df_code["rank"].max() > -1:
            df_code = df_code.sort_values(["id", "rank"])

        # Group by ID and collect cell_ids into lists
        code_map = df_code.groupby("id")["cell_id"].apply(list).to_dict()
        return code_map

    def _reconstruct_order(self, code_cells, md_cells, md_scores):
        """
        Merges code and markdown cells into a single ordered list based on predicted scores.

        Strategy:
        - Code cell at index i (0-based) is assigned position i + 0.5.
        - Markdown cell is assigned its predicted Expected Index.
        - All cells are sorted by these position values.

        Args:
            code_cells (list): List of code cell IDs (anchors).
            md_cells (list): List of markdown cell IDs (queries).
            md_scores (list/array): Predicted expected positions for markdown cells.

        Returns:
            list: Ordered list of all cell IDs.
        """
        cells = []

        # Add Code Cells with fixed fractional positions
        for i, cid in enumerate(code_cells):
            cells.append((cid, i + 0.5))

        # Add Markdown Cells with predicted positions
        for cid, score in zip(md_cells, md_scores):
            cells.append((cid, score))

        # Sort by position score
        cells.sort(key=lambda x: x[1])

        # Return only the cell IDs
        return [c[0] for c in cells]

    def predict_order(self, dataloader, code_map):
        """
        Runs inference on the dataloader and reconstructs cell orders.

        Args:
            dataloader (DataLoader): PyTorch DataLoader for the test set.
            code_map (dict): Mapping of notebook ID to list of code cell IDs.

        Returns:
            pd.DataFrame: DataFrame with 'id' and 'cell_order' columns.
        """
        results = []

        print("Running inference...")
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                code_emb = batch["code_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_emb = batch["md_embeddings"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)

                nb_ids = batch["id"]
                batch_md_ids = batch["md_ids"]  # List of lists of cell IDs

                # Forward Pass
                # logits: (Batch, Max_Md_Len, Max_Code_Len + 1)
                logits = self.model(code_emb, code_mask, md_emb, md_mask)

                # Compute Probabilities
                probs = torch.softmax(logits, dim=-1)

                # Compute Expected Index (Soft Ranking)
                # We multiply prob[k] by index k.
                # Shape: (Batch, Max_Md_Len, Max_Code_Len + 1)
                max_pos = probs.size(2)
                indices = torch.arange(max_pos, device=self.device).float()

                # expected_pos: (Batch, Max_Md_Len)
                expected_pos = torch.sum(probs * indices, dim=-1)
                expected_pos = expected_pos.cpu().numpy()

                # Reconstruct orders for each notebook in the batch
                for i, nb_id in enumerate(nb_ids):
                    # Get anchors (Code)
                    code_cells = code_map.get(nb_id, [])

                    # Get queries (Markdown)
                    curr_md_ids = batch_md_ids[i]

                    # Get scores
                    # Slice to actual length of markdown cells in this notebook (remove padding)
                    curr_scores = expected_pos[i][: len(curr_md_ids)]

                    # Merge and Sort
                    pred_order = self._reconstruct_order(
                        code_cells, curr_md_ids, curr_scores
                    )

                    # Format as space-delimited string
                    pred_string = " ".join(pred_order)

                    results.append({"id": nb_id, "cell_order": pred_string})

        return pd.DataFrame(results)

    def generate_submission(self, test_features_path=None, output_path=None):
        """
        Main method to generate the submission file.

        Args:
            test_features_path (str, optional): Path to test features parquet.
                                                Defaults to Config.TEST_FEATURES_PATH.
            output_path (str, optional): Path to save submission CSV.
                                         Defaults to Config.SUBMISSION_PATH.
        """
        test_features_path = (
            test_features_path if test_features_path else Config.TEST_FEATURES_PATH
        )
        output_path = output_path if output_path else Config.SUBMISSION_PATH

        # 1. Load Data
        print(f"Loading test data from {test_features_path}...")
        # Use shuffle=False for deterministic inference
        test_loader = get_dataloader(test_features_path, mode="test", shuffle=False)

        # 2. Load Code Map
        code_map = self._load_code_cells(test_features_path)

        # 3. Predict
        df_submission = self.predict_order(test_loader, code_map)

        # 4. Save
        print(f"Saving submission to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        print("Submission generation complete.")
