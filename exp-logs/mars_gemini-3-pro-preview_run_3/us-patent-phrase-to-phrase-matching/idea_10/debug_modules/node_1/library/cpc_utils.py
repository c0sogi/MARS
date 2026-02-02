import os
import pandas as pd
from library.config import CFG


def get_cpc_texts(cfg=CFG, load_cached_data=True):
    """
    Retrieves a dictionary mapping CPC context codes (e.g., 'A47') to their
    textual descriptions.

    Implements caching using Parquet to store the mapping.
    Since the external CPC title dataset is not provided in the input,
    this function constructs the context based on the CPC Section definitions
    (the first letter of the code), which provides high-level domain information.

    Args:
        cfg: Configuration class or object containing paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary where keys are context codes and values are description strings.
    """
    # Define cache path
    cache_path = os.path.join(cfg.output_dir, "context_map.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_cache = pd.read_parquet(cache_path)
            # Convert DataFrame back to dictionary
            cpc_texts = dict(zip(df_cache["code"], df_cache["text"]))
            return cpc_texts
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Generate Data

    # Collect all unique contexts from the datasets
    contexts = set()
    files_to_check = [cfg.train_path, cfg.val_path, cfg.test_path]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "context" in df.columns:
                contexts.update(df["context"].dropna().unique().tolist())

    # Define CPC Section mappings (High-level hierarchy)
    # A: Human Necessities
    # B: Performing Operations; Transporting
    # C: Chemistry; Metallurgy
    # D: Textiles; Paper
    # E: Fixed Constructions
    # F: Mechanical Engineering; Lighting; Heating; Weapons; Blasting
    # G: Physics
    # H: Electricity
    # Y: General Tagging

    section_map = {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
    }

    cpc_texts = {}
    for code in contexts:
        # The section is the first character of the context code (e.g. 'A' from 'A47')
        section_char = str(code)[0].upper()
        description = section_map.get(section_char, "")

        # In a full implementation with external data, we would append Class/Subclass descriptions here.
        # For now, we rely on the Section description as the hierarchical context.
        cpc_texts[code] = description

    # 3. Save to cache
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Convert dict to DataFrame for Parquet storage
    df_cache = pd.DataFrame(list(cpc_texts.items()), columns=["code", "text"])
    df_cache.to_parquet(cache_path, index=False)

    return cpc_texts
