import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from library.config import Config


class Preprocessor:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"Initializing Preprocessor with model: {Config.MODEL_CHECKPOINT}")
        self.model = SentenceTransformer(Config.MODEL_CHECKPOINT, device=self.device)
        self.model.max_seq_length = Config.MAX_TOKEN_LEN

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def _read_notebook(self, filepath):
        """Reads a notebook JSON file and returns cell data."""
        full_path = os.path.join(Config.INPUT_DIR, filepath)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            return None

    def _extract_features(self, df_meta, split_name, debug=False):
        """
        Iterates over the metadata DataFrame, extracts text from JSONs,
        computes embeddings, and constructs a features DataFrame.
        """
        print(f"Extracting features for {split_name} set...")

        if debug:
            print("Debug mode: processing first 100 notebooks only.")
            df_meta = df_meta.head(100)

        # Containers for data construction
        all_notebook_ids = []
        all_cell_ids = []
        all_cell_types = []
        all_ranks = []
        all_texts = []

        # Iterate over notebooks
        # We use tqdm for progress tracking
        for _, row in tqdm(
            df_meta.iterrows(), total=len(df_meta), desc=f"Parsing {split_name} JSONs"
        ):
            nb_id = row["id"]
            filepath = row["filepath"]

            nb_json = self._read_notebook(filepath)
            if nb_json is None:
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine order and rank
            if "cell_order" in row and pd.notna(row["cell_order"]):
                # Train/Val: Use ground truth order
                cell_order = row["cell_order"].split()
                # Create a map for rank
                # rank is 0 to N-1 based on the ground truth order
                for rank, cell_id in enumerate(cell_order):
                    if cell_id not in sources:
                        continue

                    ctype = cell_types.get(cell_id, "unknown")
                    text = sources.get(cell_id, "")

                    all_notebook_ids.append(nb_id)
                    all_cell_ids.append(cell_id)
                    all_cell_types.append(ctype)
                    all_ranks.append(rank)
                    all_texts.append(text)
            else:
                # Test: No ground truth order. Use JSON keys order (usually arbitrary/shuffled for MD)
                # We assign rank = -1 as it is unknown
                for cell_id, ctype in cell_types.items():
                    text = sources.get(cell_id, "")

                    all_notebook_ids.append(nb_id)
                    all_cell_ids.append(cell_id)
                    all_cell_types.append(ctype)
                    all_ranks.append(-1)
                    all_texts.append(text)

        print(f"Encoding {len(all_texts)} cells for {split_name}...")

        # Encode all texts in batches
        # SentenceTransformer handles batching internally, but we can tune batch_size
        embeddings = self.model.encode(
            all_texts,
            batch_size=1024,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,  # MPNet benefits from normalization for cosine similarity tasks
        )

        # Construct DataFrame
        print(f"Constructing DataFrame for {split_name}...")
        df_features = pd.DataFrame(
            {
                "id": all_notebook_ids,
                "cell_id": all_cell_ids,
                "cell_type": all_cell_types,
                "rank": all_ranks,
            }
        )

        # Add embeddings as a column of arrays (or lists, but arrays are better for parquet)
        # We convert to list of arrays to ensure pandas handles it correctly for parquet storage
        df_features["embedding"] = list(embeddings)

        return df_features

    def process_and_save(
        self, metadata_path, output_path, split_name, load_cached_data=True, debug=False
    ):
        """
        Orchestrates the check-load-compute-save cycle for a specific split.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached {split_name} features from {output_path}...")
            try:
                df = pd.read_parquet(output_path)
                print(f"Successfully loaded {len(df)} rows.")
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # 3. Compute Features
        df_features = self._extract_features(df_meta, split_name, debug=debug)

        # 4. Save to Cache
        print(f"Saving {split_name} features to {output_path}...")
        # Use pyarrow engine for efficiency with array columns
        df_features.to_parquet(output_path, engine="pyarrow", index=False)

        return df_features


def preprocess_data(debug=False, load_cached_data=True):
    """
    Main entry point to preprocess all splits (Train, Val, Test).

    Args:
        debug (bool): If True, processes only a small subset of data.
        load_cached_data (bool): If True, attempts to load from Parquet files.

    Returns:
        tuple: (df_train, df_val, df_test) DataFrames containing features.
    """
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    preprocessor = Preprocessor()

    # Process Train
    df_train = preprocessor.process_and_save(
        metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=Config.TRAIN_FEATURES_PATH,
        split_name="train",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Process Validation
    df_val = preprocessor.process_and_save(
        metadata_path=Config.VAL_METADATA_PATH,
        output_path=Config.VAL_FEATURES_PATH,
        split_name="validation",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Process Test
    df_test = preprocessor.process_and_save(
        metadata_path=Config.TEST_METADATA_PATH,
        output_path=Config.TEST_FEATURES_PATH,
        split_name="test",
        load_cached_data=load_cached_data,
        debug=debug,
    )

    print("Preprocessing complete.")
    return df_train, df_val, df_test
