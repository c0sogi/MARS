import os
import pandas as pd
from library.config import Config


def get_cpc_texts(cfg: Config, load_cached_data: bool = True) -> dict:
    """
    Parses the CPC description file to map CPC codes to their full text descriptions.
    Implements caching using Parquet to speed up subsequent runs.

    Args:
        cfg (Config): Configuration object containing file paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping CPC codes (str) to descriptions (str).
    """
    # Ensure the working directory exists
    os.makedirs(cfg.working_dir, exist_ok=True)

    cache_path = os.path.join(cfg.working_dir, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading CPC texts from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                # Convert DataFrame back to dictionary
                return dict(zip(df["code"], df["text"]))
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache not found at {cache_path}. Computing from scratch...")
    else:
        print("Ignoring cache. Computing from scratch...")

    # 2. Compute (Parse the raw file)
    print(f"Parsing CPC texts from {cfg.cpc_path}...")
    cpc_texts = {}

    if not os.path.exists(cfg.cpc_path):
        raise FileNotFoundError(f"CPC description file not found at {cfg.cpc_path}")

    with open(cfg.cpc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Robust parsing: split on first whitespace (space or tab)
            # Example format: "A47 FURNITURE; DOMESTIC ARTICLES..."
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                code, text = parts
                cpc_texts[code] = text
            else:
                # Handle cases where line might just be a code or malformed
                # For this specific task, we skip or keep code as text if needed.
                # Here we skip to ensure quality.
                continue

    # 3. Save to cache
    print(f"Saving CPC texts to cache: {cache_path}")
    df = pd.DataFrame(list(cpc_texts.items()), columns=["code", "text"])
    df.to_parquet(cache_path, index=False)

    return cpc_texts
