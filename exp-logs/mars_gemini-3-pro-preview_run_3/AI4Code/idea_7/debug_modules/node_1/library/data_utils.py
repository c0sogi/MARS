import os
import json
import random
import pandas as pd
import numpy as np
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_notebook(file_path):
    """
    Reads a notebook JSON file from the input directory.

    Args:
        file_path (str): Relative path to the notebook file (e.g., 'train/00001756c60be8.json').

    Returns:
        dict: A dictionary containing 'cell_type' and 'source' dictionaries.
              Returns None if the file cannot be read.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None


def get_data_splits():
    """
    Loads train, validation, and test metadata splits.
    Applies debug sampling if Config.DEBUG is True.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply Debug Sampling
    if Config.DEBUG:
        set_seed(Config.SEED)

        # Sample train
        if len(df_train) > Config.DEBUG_SAMPLE_SIZE:
            df_train = df_train.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

        # Sample val
        if len(df_val) > Config.DEBUG_SAMPLE_SIZE:
            df_val = df_val.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

        # Sample test (if needed, though usually we want full inference or a subset)
        if len(df_test) > Config.DEBUG_SAMPLE_SIZE:
            df_test = df_test.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

    return df_train, df_val, df_test


def generate_relaxed_pairs(df, load_cached_data=True):
    """
    Generates (Markdown, Code) pairs for fine-tuning using the Relaxed Proximity strategy.
    Pairs every markdown cell with its nearest subsequent code cell in the ground truth order.

    Args:
        df (pd.DataFrame): DataFrame containing notebook metadata (id, cell_order, file_path).
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame with columns ['markdown', 'code'].
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, "train_pairs_relaxed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            pairs_df = pd.read_parquet(cache_path)
            return pairs_df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}, regenerating. Error: {e}")

    # 2. Generate data from scratch
    pairs_data = []

    # Iterate through notebooks
    # Using simple loop to avoid dependency on tqdm/printing
    for _, row in df.iterrows():
        notebook_data = read_notebook(row["file_path"])
        if notebook_data is None:
            continue

        cell_order = row["cell_order"].split()
        cell_types = notebook_data.get("cell_type", {})
        sources = notebook_data.get("source", {})

        # Filter cell_order to only include cells present in the JSON (sanity check)
        valid_order = [cid for cid in cell_order if cid in cell_types]

        n_cells = len(valid_order)

        for i, cell_id in enumerate(valid_order):
            c_type = cell_types[cell_id]

            # We only care about Markdown cells as the "Query"
            if c_type == "markdown":
                markdown_text = sources.get(cell_id, "")

                # Find nearest subsequent code cell
                target_code_text = None

                # Look ahead
                for j in range(i + 1, n_cells):
                    next_id = valid_order[j]
                    if cell_types[next_id] == "code":
                        target_code_text = sources.get(next_id, "")
                        break

                # If a subsequent code cell was found, add the pair
                if target_code_text is not None:
                    # Basic cleaning: ensure they are strings
                    if not isinstance(markdown_text, str):
                        markdown_text = str(markdown_text)
                    if not isinstance(target_code_text, str):
                        target_code_text = str(target_code_text)

                    # Skip empty pairs to reduce noise
                    if markdown_text.strip() and target_code_text.strip():
                        pairs_data.append(
                            {"markdown": markdown_text, "code": target_code_text}
                        )

    pairs_df = pd.DataFrame(pairs_data)

    # 3. Save to cache
    try:
        pairs_df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}. Error: {e}")

    return pairs_df
