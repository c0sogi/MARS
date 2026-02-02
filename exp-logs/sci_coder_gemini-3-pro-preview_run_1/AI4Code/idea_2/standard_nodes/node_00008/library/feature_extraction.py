import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_utils import load_notebook_cells


def process_dataset(metadata_path, output_path, load_cached_data=True):
    """
    Generates or loads embeddings for a dataset defined by metadata_path.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        output_path (str): Path where the resulting Parquet file should be saved.
        load_cached_data (bool): If True, attempts to load from output_path first.

    Returns:
        pd.DataFrame: DataFrame containing cell features and embeddings.
    """
    # 1. Caching Mechanism
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}...")
        try:
            return pd.read_parquet(output_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Generating features for {metadata_path}...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {Config.model_name} on {device}...")
    model = SentenceTransformer(Config.model_name, device=device)
    model.max_seq_length = Config.max_length

    # 4. Process in Chunks
    # Processing in chunks prevents excessive memory usage while accumulating lists
    chunk_size = 2000
    total_notebooks = len(df_meta)
    num_chunks = (total_notebooks + chunk_size - 1) // chunk_size

    all_dfs = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_notebooks)
        batch_meta = df_meta.iloc[start_idx:end_idx]

        batch_cells = []
        batch_texts = []

        # Extract text from notebooks
        for _, row in batch_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]
            # Pass cell_order if available (Train/Val) to get ground truth ranks
            cell_order = (
                row["cell_order"]
                if "cell_order" in row and pd.notna(row["cell_order"])
                else None
            )

            try:
                cells = load_notebook_cells(nb_id, filepath, cell_order)
                for cell in cells:
                    batch_cells.append(cell)
                    batch_texts.append(cell["source"])
            except Exception as e:
                print(f"Warning: Error loading notebook {filepath}: {e}")
                continue

        if not batch_texts:
            continue

        # Encode texts
        # normalize_embeddings=True is beneficial for dot-product/cosine-similarity tasks
        embeddings = model.encode(
            batch_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Create DataFrame
        df_batch = pd.DataFrame(batch_cells)

        # Store embeddings as list of floats for Parquet compatibility
        df_batch["embedding"] = list(embeddings)

        # Drop raw source text to reduce file size (embeddings are the features)
        df_batch = df_batch.drop(columns=["source"])

        all_dfs.append(df_batch)

        if (i + 1) % 5 == 0 or (i + 1) == num_chunks:
            print(f"Processed chunk {i+1}/{num_chunks}")

    # 5. Save to Disk
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Saving {len(full_df)} cell features to {output_path}...")
        full_df.to_parquet(output_path, index=False)
        return full_df
    else:
        print("No data was processed.")
        return pd.DataFrame()


def extract_features(load_cached_data=True):
    """
    Main function to extract features for Train, Validation, and Test sets.
    """
    # Ensure working directory exists (handled by Config, but good practice)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Process Train
    process_dataset(
        Config.train_metadata_path, Config.train_features_path, load_cached_data
    )

    # Process Validation
    process_dataset(
        Config.val_metadata_path, Config.val_features_path, load_cached_data
    )

    # Process Test
    process_dataset(
        Config.test_metadata_path, Config.test_features_path, load_cached_data
    )
