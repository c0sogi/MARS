import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import get_device, get_ranks
from library.dataset import CachedNotebookDataset
from library.model import DualContextAnchorNetwork
from library.data_preprocessing import Preprocessor


class InferencePipeline:
    """
    Manages the inference process for the DC-AN model on the test set.
    """

    def __init__(self):
        self.device = get_device()
        self.model = DualContextAnchorNetwork().to(self.device)

        # Load trained model weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Initializing with random weights."
            )

        self.model.eval()

    def predict(self, load_cached_data=True):
        """
        Runs inference on the test set and generates the submission file.

        Args:
            load_cached_data (bool): If True, attempts to use existing cached features.
                                     If False or cache missing, regenerates features.
        """
        # 1. Ensure Test Features Exist
        # We check if the parquet file exists. If not, or if reload is forced, we run the preprocessor.
        if not os.path.exists(Config.TEST_FEATURES_PATH) or not load_cached_data:
            print(
                "Test features cache not found or reload requested. Running preprocessing..."
            )
            # Ensure working directory exists
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            preprocessor = Preprocessor()
            # Generate test features using the test metadata
            preprocessor.process_collection(
                Config.TEST_METADATA_PATH,
                Config.TEST_FEATURES_PATH,
                load_cached_data=load_cached_data,
                is_test=True,
            )

        # 2. Prepare Dataset and DataLoader
        # The dataset class handles loading the parquet file
        test_dataset = CachedNotebookDataset(Config.TEST_FEATURES_PATH, split="test")

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,  # Order doesn't matter for inference, but False is safer for debugging
            num_workers=Config.NUM_WORKERS,
            collate_fn=CachedNotebookDataset.collate_fn,
            pin_memory=True,
        )

        print(f"Starting inference on {len(test_dataset)} test notebooks...")

        preds = []

        # 3. Inference Loop
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["ids"]
                md_cell_ids_batch = batch["md_cell_ids"]
                code_cell_ids_batch = batch["code_cell_ids"]

                # Move inputs to device
                code_emb = batch["code_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                code_lens = batch["code_lens"].to(self.device)

                md_emb = batch["md_embeddings"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)
                md_lens = batch["md_lens"].to(self.device)

                # Forward Pass
                # Logits shape: (Batch, MD_Len, Code_Len + 1)
                logits = self.model(
                    code_emb, code_mask, code_lens, md_emb, md_mask, md_lens
                )

                # Compute Probabilities
                probs = torch.softmax(logits, dim=-1)

                # Compute Expected Index (Soft Ranking)
                # We calculate sum(p_i * i) to get a continuous rank
                max_len = probs.size(-1)
                indices = torch.arange(max_len, device=self.device).float()

                # expected_indices: (Batch, MD_Len)
                expected_indices = torch.sum(probs * indices, dim=-1)

                expected_indices_cpu = expected_indices.cpu().numpy()

                # 4. Reconstruct Orders
                for i, nb_id in enumerate(ids):
                    md_ids = md_cell_ids_batch[i]
                    code_ids = code_cell_ids_batch[i]
                    scores = expected_indices_cpu[i]

                    # The tensor is padded, so we slice valid scores for this notebook's markdown cells
                    valid_scores = scores[: len(md_ids)]

                    # Map markdown cell IDs to their predicted scores
                    pred_scores = {
                        mid: score for mid, score in zip(md_ids, valid_scores)
                    }

                    # Combine code and markdown cells into the final order string
                    cell_order = get_ranks(pred_scores, code_ids)
                    preds.append({"id": nb_id, "cell_order": cell_order})

        # 5. Save Submission
        df_submission = pd.DataFrame(preds)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        print(f"Saving submission file to {Config.SUBMISSION_PATH}...")
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Inference complete.")


def run_inference():
    """
    Helper function to execute the inference pipeline.
    """
    set_seed(Config.SEED)
    pipeline = InferencePipeline()
    pipeline.predict()
