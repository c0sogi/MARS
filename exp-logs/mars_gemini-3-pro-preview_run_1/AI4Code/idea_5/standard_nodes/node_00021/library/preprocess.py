import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import read_notebook, set_seed


class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"Loading backbone model: {Config.BACKBONE_NAME} on {self.device}...")
        self.model = SentenceTransformer(Config.BACKBONE_NAME, device=self.device)
        self.model.max_seq_length = Config.MAX_LENGTH

    def encode(self, texts):
        if not texts:
            return []
        # Encode and normalize to unit length for dot-product similarity
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


def get_markdown_labels(cell_order, code_ids, markdown_ids):
    """
    Computes the target label for each markdown cell.
    Label = index of the code cell that immediately follows the markdown cell.
    Range: [0, len(code_ids)]
    """
    # Create a set for fast lookup
    md_set = set(markdown_ids)

    # Map markdown ID to its label
    md_label_map = {}

    code_counter = 0

    # cell_order contains the ground truth order of all cells
    for cid in cell_order.split():
        if cid in md_set:
            md_label_map[cid] = code_counter
        elif cid in code_ids:  # It's a code cell (check strictly to be safe)
            code_counter += 1

    # Return labels in the order of markdown_ids (which is the shuffled order from JSON)
    labels = [md_label_map.get(mid, 0) for mid in markdown_ids]
    return labels


def process_split(metadata_path, feature_extractor, is_test=False):
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return None

    df_meta = pd.read_csv(metadata_path)

    if Config.DEBUG:
        print(
            f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks from {metadata_path}"
        )
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"Processing {len(df_meta)} notebooks from {metadata_path}...")

    data_list = []

    # Pre-calculate paths to minimize os.path.join overhead in loop
    input_dir = Config.INPUT_DIR

    for idx, row in df_meta.iterrows():
        nb_id = row["id"]
        filepath = os.path.join(input_dir, row["filepath"])

        # Read notebook content
        # code_ids are in order, markdown_ids are shuffled (for train) or relative order (for test)
        code_ids, markdown_ids, source_dict = read_notebook(filepath)

        if not code_ids and not markdown_ids:
            continue

        # Prepare texts
        code_texts = [source_dict.get(cid, "") for cid in code_ids]
        markdown_texts = [source_dict.get(cid, "") for cid in markdown_ids]

        # Generate Embeddings
        # We encode separately to keep logic simple, though batching together is possible.
        # Given the model speed, this is sufficient.
        code_embeddings = feature_extractor.encode(code_texts)
        markdown_embeddings = feature_extractor.encode(markdown_texts)

        # Compute Labels (only for Train/Val)
        labels = []
        if not is_test:
            cell_order = row["cell_order"]
            # code_ids passed here must be the set of code cells found in the notebook
            labels = get_markdown_labels(cell_order, set(code_ids), markdown_ids)

        # Store data
        # Convert numpy arrays to lists for Parquet serialization
        entry = {
            "id": nb_id,
            "code_ids": code_ids,
            "markdown_ids": markdown_ids,
            "code_embeddings": (
                list(code_embeddings) if len(code_embeddings) > 0 else []
            ),
            "markdown_embeddings": (
                list(markdown_embeddings) if len(markdown_embeddings) > 0 else []
            ),
            "markdown_labels": labels,
        }
        data_list.append(entry)

        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1} notebooks...")

    return pd.DataFrame(data_list)


def precompute_features(load_cached_data=True):
    set_seed(Config.SEED)
    Config.setup()  # Ensure directories exist

    # Define tasks: (Metadata Path, Output Path, Is Test)
    tasks = [
        (Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_PATH, False),
        (Config.VAL_METADATA_PATH, Config.VAL_FEATURES_PATH, False),
        (Config.TEST_METADATA_PATH, Config.TEST_FEATURES_PATH, True),
    ]

    extractor = None

    for meta_path, out_path, is_test in tasks:
        if load_cached_data and os.path.exists(out_path):
            print(f"Cache found for {out_path}. Skipping computation.")
            continue

        print(f"Computing features for {out_path}...")

        # Lazy initialization of the model to save memory/time if all caches exist
        if extractor is None:
            extractor = FeatureExtractor()

        df_features = process_split(meta_path, extractor, is_test=is_test)

        if df_features is not None:
            print(f"Saving {len(df_features)} records to {out_path}...")
            # Use pyarrow engine for efficient storage of list columns
            df_features.to_parquet(out_path, engine="pyarrow", index=False)
            print("Save complete.")
        else:
            print(f"Warning: No data processed for {out_path}")

    print("Feature precomputation finished.")
