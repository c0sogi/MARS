import os
import pandas as pd
from library.config import Config


def get_cpc_texts(cfg, load_cached_data=True):
    """
    Parses the CPC description file to map context codes to their full text descriptions.
    Implements caching using parquet to avoid re-parsing on every run.

    Args:
        cfg: Configuration object containing paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping CPC codes (str) to descriptions (str).
    """
    # Ensure working directory exists
    os.makedirs(cfg.working_dir, exist_ok=True)

    cache_path = os.path.join(cfg.working_dir, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert back to dictionary
            return dict(zip(df["code"], df["description"]))
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    if not os.path.exists(cfg.cpc_description_path):
        raise FileNotFoundError(
            f"CPC description file not found at {cfg.cpc_description_path}"
        )

    data = []
    with open(cfg.cpc_description_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Clean markdown list characters if present
            if line.startswith("- ") or line.startswith("* "):
                line = line[2:]

            # Split into Code and Description
            # Assuming format "CODE Description" or "CODE: Description"
            # We split by the first space.
            parts = line.split(" ", 1)

            if len(parts) == 2:
                code = parts[0].strip().rstrip(":")
                desc = parts[1].strip()
                data.append({"code": code, "description": desc})
            elif len(parts) == 1:
                # Fallback for lines that might only contain a code or are malformed
                # Though unlikely given the task description
                code = parts[0].strip().rstrip(":")
                data.append({"code": code, "description": ""})

    df = pd.DataFrame(data)

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return dict(zip(df["code"], df["description"]))
