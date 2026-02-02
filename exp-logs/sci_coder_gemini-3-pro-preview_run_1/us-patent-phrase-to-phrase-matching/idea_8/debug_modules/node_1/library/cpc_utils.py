import os
import pandas as pd
from library.config import CFG


def get_cpc_texts(cfg=CFG, load_cached_data=True):
    """
    Retrieves the CPC context descriptions, either from a cached Parquet file
    or by parsing the raw description.md file.

    Args:
        cfg: Configuration class containing paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping CPC codes (e.g., 'H04') to their
              full hierarchical descriptions.
    """
    cache_path = cfg.cpc_cache_path

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Convert back to dictionary: {code: description}
            cpc_texts = dict(zip(df["code"], df["description"]))
            return cpc_texts
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Parse raw data if cache not used or failed
    if not os.path.exists(cfg.cpc_description_file):
        raise FileNotFoundError(
            f"CPC description file not found at {cfg.cpc_description_file}"
        )

    raw_cpc_data = {}
    with open(cfg.cpc_description_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Assume format: "CODE Description..."
            # We split by the first whitespace.
            parts = line.split(" ", 1)
            if len(parts) == 2:
                code, description = parts
                raw_cpc_data[code] = description
            else:
                # Handle cases where line might just be a code or malformed
                # In this specific dataset, lines are usually well-formed.
                pass

    # 3. Construct Hierarchical Descriptions
    # The context codes in the dataset are typically Classes (e.g., 'A47', 'H04').
    # We want to map 'H04' -> "Description of H; Description of H04"

    cpc_texts = {}

    for code, desc in raw_cpc_data.items():
        # Clean the description (remove leading/trailing punctuation/whitespace)
        desc = desc.strip()

        # Determine hierarchy
        # Section codes are 1 letter (A-H)
        # Class codes are 3 chars (Letter + 2 digits, e.g., H04)

        if len(code) == 3:
            section_code = code[0]
            section_desc = raw_cpc_data.get(section_code, "")

            if section_desc:
                # Combine: Section Desc; Class Desc
                full_desc = f"{section_desc}; {desc}"
            else:
                full_desc = desc

            cpc_texts[code] = full_desc
        else:
            # For sections or other codes, keep as is
            cpc_texts[code] = desc

    # 4. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Convert to DataFrame for Parquet storage
    df_out = pd.DataFrame([{"code": k, "description": v} for k, v in cpc_texts.items()])
    df_out.to_parquet(cache_path, index=False)

    return cpc_texts
