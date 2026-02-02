import os
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from library.config import (
    INPUT_DIR,
    FEATURE_DIR,
    BACKBONE_NAME,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SEED,
)
from library.utils import set_seed


class Preprocessor:
    def __init__(self):
        """
        Initializes the Preprocessor.
        """
        set_seed(SEED)
        self.model = None  # Lazy loading

    def _load_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading backbone model: {BACKBONE_NAME} on {device}...")
            self.model = SentenceTransformer(BACKBONE_NAME, device=device)

    def _get_notebook_path(self, filepath):
        return os.path.join(INPUT_DIR, filepath)

    def _clean_text(self, text):
        return text.strip()

    def process_split(
        self, split="train", load_cached_data=True, debug=False, num_debug_samples=100
    ):
        """
        Process a data split (train, val, test), generate embeddings, and save to Parquet.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, try to load from existing parquet file.
            debug (bool): If True, process only a small subset.
            num_debug_samples (int): Number of samples to use in debug mode.

        Returns:
            str: Path to the processed parquet file.
        """
        os.makedirs(FEATURE_DIR, exist_ok=True)
        output_filename = f"{split}_features.parquet"
        output_path = os.path.join(FEATURE_DIR, output_filename)

        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features for {split} from {output_path}...")
            return output_path

        print(f"Processing {split} data (Debug={debug})...")

        # Load Metadata
        if split == "train":
            meta_path = TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = VAL_METADATA_PATH
        elif split == "test":
            meta_path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df_meta = pd.read_csv(meta_path)

        if debug:
            df_meta = df_meta.head(num_debug_samples)

        # Load Model
        self._load_model()

        # Containers for data
        records = []

        # Iterate over notebooks
        for _, row in tqdm(
            df_meta.iterrows(), total=len(df_meta), desc=f"Extracting {split}"
        ):
            nb_id = row["id"]
            json_path = self._get_notebook_path(row["filepath"])

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    nb_data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            cell_types = nb_data.get("cell_type", {})
            sources = nb_data.get("source", {})

            # Determine Code Sequence and Markdown Targets
            # For Train/Val, we rely on 'cell_order' from metadata
            # For Test, we rely on the order in the JSON (Code cells are correct)

            if split in ["train", "val"]:
                ground_truth_order = row["cell_order"].split()

                # Identify code cells in order
                code_sequence = [
                    cid for cid in ground_truth_order if cell_types.get(cid) == "code"
                ]

                # Map code cell ID to its index in the code sequence (0 to N-1)
                code_rank_map = {cid: i for i, cid in enumerate(code_sequence)}

                # Determine labels for markdown cells
                # Label = Index of the NEXT code cell in the sequence.
                # If markdown is at the end (or after all code), Label = len(code_sequence) (EOS token index)

                md_labels = {}
                current_code_idx = 0

                for cid in ground_truth_order:
                    ctype = cell_types.get(cid)
                    if ctype == "code":
                        current_code_idx += 1
                    elif ctype == "markdown":
                        md_labels[cid] = current_code_idx
            else:
                # Test set
                # Code cells are in correct order in the JSON keys?
                # The task description says: "The code cells are in their original (correct) order."
                # However, JSON dictionaries are unordered in older Python, but ordered in 3.7+.
                # We should be careful. Usually, the provided JSONs in this competition have keys in order
                # OR we just extract all code cells and assume their relative order in the file is correct.
                # Given the standard format, we filter cell_types for 'code'.

                # We iterate over the keys in the JSON source/cell_type to get cells.
                # Note: In the provided dataset, 'cell_type' and 'source' are dictionaries.
                # We need a stable iteration. The sample json shows keys.
                # We will assume the order of keys in 'cell_type' reflects the file order,
                # but we filter for code to build the anchor sequence.

                all_cells = list(cell_types.keys())
                code_sequence = [
                    cid for cid in all_cells if cell_types.get(cid) == "code"
                ]
                md_labels = {}  # No labels for test

            # Prepare text for encoding
            # We process all cells relevant to this notebook
            cells_to_process = []

            # Add code cells (Anchors)
            for i, cid in enumerate(code_sequence):
                text = self._clean_text(sources.get(cid, ""))
                cells_to_process.append(
                    {
                        "cell_id": cid,
                        "cell_type": "code",
                        "text": text,
                        "label": -1,  # Code cells don't have a target label
                        "rank_in_code": i,
                    }
                )

            # Add markdown cells (Queries)
            # For Train/Val, we use ground_truth_order to find MD cells
            # For Test, we use all remaining cells
            if split in ["train", "val"]:
                md_cells = [
                    cid
                    for cid in ground_truth_order
                    if cell_types.get(cid) == "markdown"
                ]
            else:
                md_cells = [
                    cid for cid in all_cells if cell_types.get(cid) == "markdown"
                ]

            for cid in md_cells:
                text = self._clean_text(sources.get(cid, ""))
                label = md_labels.get(cid, -1)  # -1 for test
                cells_to_process.append(
                    {
                        "cell_id": cid,
                        "cell_type": "markdown",
                        "text": text,
                        "label": label,
                        "rank_in_code": -1,
                    }
                )

            if not cells_to_process:
                continue

            # Batch Encode
            texts = [c["text"] for c in cells_to_process]
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Aggregate results
            for i, cell_data in enumerate(cells_to_process):
                records.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_data["cell_id"],
                        "cell_type": cell_data["cell_type"],
                        "embedding": embeddings[i].tolist(),  # Serialize for Parquet
                        "label": cell_data["label"],
                        "rank_in_code": cell_data[
                            "rank_in_code"
                        ],  # Helper for reconstruction
                    }
                )

        # Save to Parquet
        df_out = pd.DataFrame(records)

        # Optimize types
        df_out["label"] = df_out["label"].astype(np.int32)
        df_out["rank_in_code"] = df_out["rank_in_code"].astype(np.int32)

        print(f"Saving {len(df_out)} cell features to {output_path}...")
        df_out.to_parquet(output_path, index=False)

        return output_path
