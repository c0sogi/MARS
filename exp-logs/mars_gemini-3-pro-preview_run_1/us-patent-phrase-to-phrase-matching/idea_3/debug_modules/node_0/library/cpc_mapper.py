import os
import pandas as pd
from library.config import CFG


def get_cpc_texts(cfg, load_cached_data=True):
    """
    Parses the CPC description file to create a mapping from CPC codes (e.g., 'A47')
    to their natural language descriptions. Implements caching using Parquet.

    Args:
        cfg: Configuration object containing file paths.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary mapping CPC codes (str) to descriptions (str).
    """
    # Ensure working directory exists (handled by CFG, but good practice to be safe)
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(cfg.WORKING_DIR, "cpc_texts.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert DataFrame back to dictionary
            cpc_texts = dict(zip(df["code"], df["text"]))
            return cpc_texts
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    cpc_texts = {}

    if not os.path.exists(cfg.CPC_DATA):
        raise FileNotFoundError(f"CPC Data file not found at {cfg.CPC_DATA}")

    with open(cfg.CPC_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # We assume the format: "CODE Description text..."
            # Split on the first whitespace
            parts = line.split(" ", 1)

            if len(parts) == 2:
                code, text = parts
                cpc_texts[code] = text
            else:
                # Handle cases where line might just be a code or malformed
                # In this specific dataset, we expect valid lines.
                # If split fails, we ignore or take whole line as code (unlikely valid)
                pass

    # 3. Save to cache
    # Convert dict to DataFrame for Parquet storage
    df = pd.DataFrame(list(cpc_texts.items()), columns=["code", "text"])
    df.to_parquet(cache_path, index=False)

    return cpc_texts
