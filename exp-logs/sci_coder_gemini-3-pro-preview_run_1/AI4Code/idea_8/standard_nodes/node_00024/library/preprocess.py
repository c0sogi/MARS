import os
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class FeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        set_seed(self.config.SEED)

        print(f"Initializing FeatureExtractor with model: {self.config.MODEL_NAME}")
        self.model = SentenceTransformer(self.config.MODEL_NAME, device=self.device)
        # MPNet has a max sequence length of 512 tokens
        self.model.max_seq_length = 512

    def _read_notebook(self, filepath):
        """Reads a JSON notebook file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return {}

    def _get_code_order_from_json(self, nb_json):
        """
        Infers the order of code cells from the JSON source dictionary keys.
        Python 3.7+ preserves insertion order, and the task description implies
        code cells appear in their original order in the source files.
        """
        cell_types = nb_json.get("cell_type", {})
        source = nb_json.get("source", {})

        code_ids = []
        md_ids = []

        # Iterate over source keys to respect file order
        for cell_id in source.keys():
            ctype = cell_types.get(cell_id, "unknown")
            if ctype == "code":
                code_ids.append(cell_id)
            elif ctype == "markdown":
                md_ids.append(cell_id)

        return code_ids, md_ids

    def _compute_labels(self, correct_order, code_ids, md_ids):
        """
        Computes the target label for each markdown cell.
        The label is the index of the code cell that immediately follows the markdown cell.
        Label range: [0, len(code_ids)].
        Label i means the markdown cell should be placed before code_ids[i].
        Label len(code_ids) means the markdown cell is after the last code cell (EOS).
        """
        # Filter correct_order to get the ground truth code sequence
        # This ensures we align with the ground truth even if JSON parsing is ambiguous
        gt_code_sequence = [cid for cid in correct_order if cid in code_ids]

        # Map code cell ID to its index in the sequence
        code_rank_map = {cid: i for i, cid in enumerate(gt_code_sequence)}

        md_label_map = {}
        current_label = 0

        for cid in correct_order:
            if cid in code_rank_map:
                # When we pass a code cell, the insertion point for subsequent MD cells increments
                current_label += 1
            elif cid in md_ids:
                # Assign current insertion point to this markdown cell
                # If MD is before C0, label is 0.
                # If MD is after C0 (and before C1), label is 1.
                md_label_map[cid] = current_label

        # Return labels corresponding to the input list of md_ids
        return [md_label_map.get(mid, 0) for mid in md_ids]

    def process_dataset(self, metadata_path, output_path, load_cached_data=True):
        """
        Main processing function. Loads metadata, extracts text, computes embeddings,
        and saves to Parquet.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}")
            return pd.read_parquet(output_path)

        print(f"Processing dataset from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        # Data structures
        # We accumulate notebook data objects and fill their embeddings via batched processing
        notebooks_data = []

        # Buffers for batch encoding
        text_buffer = []
        # Mapping from buffer index to (notebook_index, cell_type, list_index)
        map_buffer = []

        BATCH_SIZE_TEXT = 16384  # Accumulate many texts before sending to GPU

        for idx, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = os.path.join(self.config.INPUT_DIR, row["filepath"])

            nb_json = self._read_notebook(filepath)
            source = nb_json.get("source", {})
            cell_types = nb_json.get("cell_type", {})

            # Determine IDs and Labels
            if "cell_order" in row and pd.notna(row["cell_order"]):
                # Train/Val: Use ground truth
                correct_order = row["cell_order"].split()

                # Identify cells present in both order and JSON
                code_ids = [
                    cid for cid in correct_order if cell_types.get(cid) == "code"
                ]
                md_ids = [
                    cid for cid in correct_order if cell_types.get(cid) == "markdown"
                ]

                labels = self._compute_labels(correct_order, code_ids, md_ids)
            else:
                # Test: Use JSON order
                code_ids, md_ids = self._get_code_order_from_json(nb_json)
                labels = [-1] * len(md_ids)  # Dummy labels

            # Initialize notebook storage
            nb_struct = {
                "id": nb_id,
                "ancestor_id": row.get("ancestor_id", nb_id),
                "cell_order": (
                    row["cell_order"]
                    if "cell_order" in row and pd.notna(row["cell_order"])
                    else ""
                ),
                "code_ids": code_ids,
                "markdown_ids": md_ids,
                "markdown_labels": labels,
                "code_embeddings": [None] * len(code_ids),
                "markdown_embeddings": [None] * len(md_ids),
            }
            notebooks_data.append(nb_struct)

            # Add texts to buffer
            # Code
            for i, cid in enumerate(code_ids):
                text = source.get(cid, "")
                if isinstance(text, list):
                    text = "".join(text)
                # Truncate extremely long text to avoid memory spikes before tokenization
                text_buffer.append(text[:2000])
                map_buffer.append((idx, "code", i))

            # Markdown
            for i, cid in enumerate(md_ids):
                text = source.get(cid, "")
                if isinstance(text, list):
                    text = "".join(text)
                text_buffer.append(text[:2000])
                map_buffer.append((idx, "md", i))

            # Process Buffer if full
            if len(text_buffer) >= BATCH_SIZE_TEXT:
                self._flush_buffer(text_buffer, map_buffer, notebooks_data)
                text_buffer = []
                map_buffer = []

        # Flush remaining texts
        if text_buffer:
            self._flush_buffer(text_buffer, map_buffer, notebooks_data)

        # Convert to DataFrame
        print(f"Constructing DataFrame for {len(notebooks_data)} notebooks...")

        # Filter out any corrupted entries (should not happen)
        valid_data = [
            nb
            for nb in notebooks_data
            if None not in nb["code_embeddings"]
            and None not in nb["markdown_embeddings"]
        ]

        df_output = pd.DataFrame(valid_data)

        # Save to Parquet
        print(f"Saving features to {output_path}...")
        # PyArrow handles list columns efficiently
        df_output.to_parquet(output_path, engine="pyarrow", index=False)

        return df_output

    def _flush_buffer(self, texts, mapping, notebooks_data):
        """Encodes buffered texts and assigns embeddings back to notebook structures."""
        if not texts:
            return

        embeddings = self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for idx, (nb_idx, ctype, list_idx) in enumerate(mapping):
            # Convert numpy array to list for Parquet compatibility
            emb = embeddings[idx].tolist()
            if ctype == "code":
                notebooks_data[nb_idx]["code_embeddings"][list_idx] = emb
            else:
                notebooks_data[nb_idx]["markdown_embeddings"][list_idx] = emb

    def run_preprocessing(self):
        """Executes the full preprocessing pipeline."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.TRAIN_FEATS_PATH), exist_ok=True)

        print("--- Preprocessing Training Set ---")
        self.process_dataset(
            self.config.TRAIN_METADATA_PATH,
            self.config.TRAIN_FEATS_PATH,
            load_cached_data=True,
        )

        print("--- Preprocessing Validation Set ---")
        self.process_dataset(
            self.config.VAL_METADATA_PATH,
            self.config.VAL_FEATS_PATH,
            load_cached_data=True,
        )

        print("--- Preprocessing Test Set ---")
        self.process_dataset(
            self.config.TEST_METADATA_PATH,
            self.config.TEST_FEATS_PATH,
            load_cached_data=True,
        )
        print("Preprocessing complete.")
