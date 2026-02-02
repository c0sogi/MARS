import os
import pandas as pd
from library.config import Config


def get_cpc_texts(load_cached_data=True):
    """
    Parses the CPC description file to create a mapping from CPC codes to their
    textual descriptions. Implements caching using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load the mapping from a
                                 cached Parquet file. If False or if the cache
                                 does not exist, parses the raw file and saves
                                 the cache.

    Returns:
        dict: A dictionary where keys are CPC codes (str) and values are
              descriptions (str).
    """
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    cache_path = os.path.join(Config.output_dir, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert DataFrame back to dictionary
            cpc_texts = dict(zip(df["code"], df["text"]))
            return cpc_texts
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Parse raw data
    cpc_texts = {}
    if not os.path.exists(Config.cpc_path):
        print(
            f"Warning: CPC description file not found at {Config.cpc_path}. Returning empty dict."
        )
        return {}

    with open(Config.cpc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Expecting format: "CODE Description text..."
            # Split on the first whitespace
            parts = line.split(" ", 1)
            if len(parts) == 2:
                code, text = parts
                cpc_texts[code] = text
            else:
                # Handle cases where line might not have a description or malformed
                # Just use the whole line as code if no space, or skip
                if len(parts) == 1:
                    cpc_texts[parts[0]] = ""

    # 3. Save to cache
    try:
        # Convert dictionary to DataFrame for Parquet storage
        df = pd.DataFrame(list(cpc_texts.items()), columns=["code", "text"])
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")

    return cpc_texts
