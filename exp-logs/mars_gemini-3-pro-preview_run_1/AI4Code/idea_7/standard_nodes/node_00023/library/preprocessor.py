import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import read_notebook, set_seed


class Preprocessor:
    """
    Handles loading raw notebooks, encoding text using a pre-trained Transformer,
    and caching the features to disk.
    """

    def __init__(self):
        self.config = Config
        set_seed(self.config.SEED)

    def process_all(self, load_cached_data=True):
        """
        Processes Train, Validation, and Test splits.
        """
        # Process Train
        self.process_split(
            metadata_path=self.config.TRAIN_METADATA_PATH,
            output_path=self.config.TRAIN_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=False,
        )
        # Process Val
        self.process_split(
            metadata_path=self.config.VAL_METADATA_PATH,
            output_path=self.config.VAL_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=False,
        )
        # Process Test
        self.process_split(
            metadata_path=self.config.TEST_METADATA_PATH,
            output_path=self.config.TEST_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_test=True,
        )

    def process_split(self, metadata_path, output_path, load_cached_data, is_test):
        """
        Loads metadata, reads notebooks, encodes text, and saves to Parquet.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}")
            return pd.read_parquet(output_path)

        print(f"Processing data from {metadata_path}...")

        # 2. Load Metadata
        df_meta = pd.read_csv(metadata_path)

        # Debugging: Subset data if configured
        if self.config.DEBUG:
            print(
                f"DEBUG Mode: Sampling first {self.config.DEBUG_SUBSET_SIZE} notebooks."
            )
            df_meta = df_meta.head(self.config.DEBUG_SUBSET_SIZE)

        # 3. Initialize Model
        print(f"Initializing {self.config.MODEL_BACKBONE} on {self.config.DEVICE}...")
        model = SentenceTransformer(self.config.MODEL_BACKBONE)
        model.to(self.config.DEVICE)
        model.max_seq_length = self.config.MAX_TOKEN_LEN

        # 4. Processing Loop
        # We accumulate text to encode in batches for efficiency
        batch_texts = []
        batch_meta = []  # Tuples of (nb_id, cell_id, cell_type, rank)
        all_data = []

        # Batch size for the Transformer inference (sentences, not notebooks)
        ENCODE_BATCH_SIZE = 4096

        def flush_batch(texts, metas, collector):
            if not texts:
                return

            # Encode
            embeddings = model.encode(
                texts,
                batch_size=512,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Store
            for i, emb in enumerate(embeddings):
                nb_id, c_id, c_type, rank = metas[i]
                collector.append(
                    {
                        "id": nb_id,
                        "cell_id": c_id,
                        "cell_type": c_type,
                        "rank": rank,
                        "embedding": emb.tobytes(),  # Store as bytes to save space/complexity in Parquet
                    }
                )

        total_notebooks = len(df_meta)
        print(f"Extracting and encoding cells for {total_notebooks} notebooks...")

        for idx, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]

            try:
                nb_json = read_notebook(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine processing order and ranks
            if not is_test:
                # Use ground truth order
                if isinstance(row["cell_order"], str):
                    cell_order = row["cell_order"].split()
                else:
                    cell_order = []
                rank_map = {cid: i for i, cid in enumerate(cell_order)}
                process_order = cell_order
            else:
                # Use whatever order is in the JSON keys (Code is sorted, MD is shuffled)
                # We don't have ground truth ranks
                process_order = list(cell_types.keys())
                rank_map = {}

            for cell_id in process_order:
                if cell_id not in sources:
                    continue

                ctype = cell_types.get(cell_id, "unknown")
                text = sources[cell_id]

                # Ensure text is string
                if not isinstance(text, str):
                    text = str(text)

                # Get rank (ground truth or -1)
                rank = rank_map.get(cell_id, -1)

                batch_texts.append(text)
                batch_meta.append((nb_id, cell_id, ctype, rank))

                # Flush if buffer full
                if len(batch_texts) >= ENCODE_BATCH_SIZE:
                    flush_batch(batch_texts, batch_meta, all_data)
                    batch_texts = []
                    batch_meta = []

        # Flush remaining
        flush_batch(batch_texts, batch_meta, all_data)

        # 5. Save to Parquet
        print(f"Constructing DataFrame with {len(all_data)} cells...")
        df_out = pd.DataFrame(all_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Saving to {output_path}...")
        df_out.to_parquet(output_path, index=False)
        print("Done.")

        return df_out
