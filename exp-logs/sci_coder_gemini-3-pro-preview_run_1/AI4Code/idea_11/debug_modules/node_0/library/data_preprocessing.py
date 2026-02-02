import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import read_json


class Preprocessor:
    """
    Handles the extraction of text from notebooks, generation of embeddings
    using a pre-trained Transformer model, and caching of features to disk.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_name = Config.MODEL_BACKBONE
        self.max_length = Config.MAX_LENGTH

        print(f"Initializing Preprocessor with model: {self.model_name}")
        # Initialize the Sentence Transformer model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.max_length

    def process_collection(
        self, metadata_path, output_path, load_cached_data=True, is_test=False
    ):
        """
        Processes a set of notebooks defined in the metadata file.

        Args:
            metadata_path (str): Path to the metadata CSV.
            output_path (str): Path where the output Parquet file should be saved.
            load_cached_data (bool): If True, attempts to load from disk first.
            is_test (bool): Whether processing the test set (no ground truth ranks).

        Returns:
            pd.DataFrame: The processed features.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}")
            return pd.read_parquet(output_path)

        # 2. Load Metadata
        print(f"Processing data from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"DEBUG mode: Sampling {Config.SAMPLE_SIZE} notebooks.")
            df_meta = df_meta.iloc[: Config.SAMPLE_SIZE]

        # 3. Extract Text and Metadata
        # We collect all text first to perform efficient batched encoding
        text_batch = []
        meta_batch = []
        input_dir = Config.INPUT_DIR

        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = os.path.join(input_dir, row["filepath"])

            try:
                nb_json = read_json(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine Ranks
            if is_test:
                # Test set has no ground truth order; rank is irrelevant (-1)
                # We process all cells present in the JSON
                ordered_cell_ids = list(cell_types.keys())
                rank_map = {cid: -1 for cid in ordered_cell_ids}
            else:
                # Train/Val sets have ground truth 'cell_order'
                if "cell_order" in row and not pd.isna(row["cell_order"]):
                    ordered_cell_ids = row["cell_order"].split()
                    rank_map = {cid: i for i, cid in enumerate(ordered_cell_ids)}
                else:
                    # Fallback (should not happen with correct metadata)
                    ordered_cell_ids = list(cell_types.keys())
                    rank_map = {cid: i for i, cid in enumerate(ordered_cell_ids)}

            # Collect cell data
            # We iterate over keys in cell_types to ensure we capture all cells
            # (even if potentially missing from order string, though unlikely)
            for cell_id, ctype in cell_types.items():
                if cell_id not in sources:
                    continue

                text = sources[cell_id]
                rank = rank_map.get(cell_id, -1)

                # Only keep cells that are relevant (in rank_map for train, or all for test)
                # For training, if a cell is in JSON but not in order, we might skip or assign -1.
                # Here we keep everything; the Dataset class can filter based on rank != -1 if needed.

                text_batch.append(text)
                meta_batch.append(
                    {"id": nb_id, "cell_id": cell_id, "cell_type": ctype, "rank": rank}
                )

        # 4. Batched Encoding
        print(f"Encoding {len(text_batch)} cells...")
        # Use a larger batch size for inference than training
        inference_batch_size = Config.BATCH_SIZE * 4

        embeddings = self.model.encode(
            text_batch,
            batch_size=inference_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
        )

        # 5. Assemble and Save
        df_out = pd.DataFrame(meta_batch)
        # Store embeddings as a column of lists (Parquet handles this well)
        df_out["embedding"] = list(embeddings)

        print(f"Saving features to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_parquet(output_path, index=False)

        return df_out

    def run(self, load_cached_data=True):
        """
        Executes the preprocessing pipeline for Train, Validation, and Test sets.
        """
        # Process Train
        self.process_collection(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=False,
        )

        # Process Validation
        self.process_collection(
            Config.VAL_METADATA_PATH,
            Config.VAL_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=False,
        )

        # Process Test
        self.process_collection(
            Config.TEST_METADATA_PATH,
            Config.TEST_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=True,
        )
        print("Data preprocessing complete.")
