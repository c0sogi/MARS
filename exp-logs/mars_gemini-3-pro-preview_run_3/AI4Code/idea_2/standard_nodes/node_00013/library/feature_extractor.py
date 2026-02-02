import os
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class EmbeddingGenerator:
    """
    Handles the conversion of raw notebook text into semantic embeddings using
    a pre-trained Sentence Transformer model. Caches results to Parquet files.
    """

    def __init__(self):
        """
        Initialize the EmbeddingGenerator.
        Loads the Sentence Transformer model onto the configured device.
        """
        Config.set_seed(Config.SEED)
        self.device = Config.DEVICE
        print(
            f"Loading Sentence Transformer model: {Config.MODEL_NAME} on {self.device}..."
        )
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)
        self.model.eval()

    def process_split(
        self, split_name: str, load_cached_data: bool = True, debug_limit: int = None
    ):
        """
        Process a specific data split (train, val, test).

        Args:
            split_name (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from disk cache first.
            debug_limit (int, optional): Limit the number of notebooks to process for debugging.

        Returns:
            pd.DataFrame: DataFrame containing embeddings and metadata.
        """
        # Determine paths based on split
        if split_name == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.TRAIN_CACHE_PATH
        elif split_name == "val":
            metadata_path = Config.VAL_METADATA_PATH
            cache_path = Config.VAL_CACHE_PATH
        elif split_name == "test":
            metadata_path = Config.TEST_METADATA_PATH
            cache_path = Config.TEST_CACHE_PATH
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {split_name} from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                print(f"Successfully loaded {len(df)} cell records.")
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Generating features for {split_name}...")
        df_metadata = pd.read_csv(metadata_path)

        if debug_limit:
            df_metadata = df_metadata.head(debug_limit)
            print(f"Debug mode: Limiting to {debug_limit} notebooks.")

        # Prepare buffers for batch processing
        records = []
        text_buffer = []
        text_meta_buffer = []  # Stores (notebook_id, cell_id, cell_type, rank)

        # Batch size for model inference (sentences)
        # Note: Config.BATCH_SIZE is typically for training the downstream model.
        # We can use a larger batch size for inference here.
        INFERENCE_BATCH_SIZE = 256

        total_notebooks = len(df_metadata)

        for idx, row in df_metadata.iterrows():
            notebook_id = row["id"]
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Determine cell ranks if available
            cell_ranks = {}
            if "cell_order" in row and pd.notna(row["cell_order"]):
                order_list = row["cell_order"].split()
                for rank, cell_id in enumerate(order_list):
                    cell_ranks[cell_id] = rank

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            cell_types = data.get("cell_type", {})
            sources = data.get("source", {})

            for cell_id, c_type in cell_types.items():
                source_text = sources.get(cell_id, "")

                # Pre-processing: Truncate very long text to avoid excessive tokenization overhead
                # The model (all-MiniLM-L6-v2) has a limit of usually 256 or 512 tokens.
                # 1000 chars is a safe heuristic to capture the start.
                processed_text = source_text[:1000] if source_text else ""

                # Determine rank
                # For test set, rank is -1. For train/val, if cell not in order list (rare), -1.
                rank = cell_ranks.get(cell_id, -1)

                text_buffer.append(processed_text)
                text_meta_buffer.append(
                    {
                        "notebook_id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "rank": rank,
                    }
                )

                # If buffer is full, encode and flush
                if len(text_buffer) >= INFERENCE_BATCH_SIZE:
                    self._flush_buffer(text_buffer, text_meta_buffer, records)
                    text_buffer = []
                    text_meta_buffer = []

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{total_notebooks} notebooks...")

        # Process remaining items in buffer
        if text_buffer:
            self._flush_buffer(text_buffer, text_meta_buffer, records)

        # Create DataFrame
        df_features = pd.DataFrame(records)

        # Ensure embedding column is recognized correctly (though parquet handles lists well)
        # We don't need to do anything special for pyarrow/parquet with lists usually.

        # Save to cache
        print(f"Saving {len(df_features)} features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features

    def _flush_buffer(self, text_buffer, meta_buffer, records_list):
        """
        Encodes texts in the buffer and appends combined metadata and embeddings to records_list.
        """
        # Encode batch
        # show_progress_bar=False to reduce clutter
        embeddings = self.model.encode(
            text_buffer,
            batch_size=len(text_buffer),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine metadata with embeddings
        for meta, emb in zip(meta_buffer, embeddings):
            meta["embedding"] = (
                emb.tolist()
            )  # Convert numpy array to list for Parquet compatibility
            records_list.append(meta)
