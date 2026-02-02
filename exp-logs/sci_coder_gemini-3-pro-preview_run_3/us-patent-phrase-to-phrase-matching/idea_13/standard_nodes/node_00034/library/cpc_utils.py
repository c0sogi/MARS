import os
import pandas as pd
from library.config import Config

# CPC Section Titles (High-level hierarchy)
CPC_SECTIONS = {
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


def get_cpc_texts(cfg: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Generates or loads a mapping between CPC context codes and their textual descriptions.

    Args:
        cfg: Configuration object containing paths.
        load_cached_data: If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: DataFrame with columns ['context', 'context_text'].
    """
    cache_path = cfg.cpc_context_path

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading CPC context map from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Generating CPC context map...")

    # 2. Gather all unique contexts from metadata
    # We need to check train, val, and test to ensure full coverage
    unique_contexts = set()

    paths_to_check = [cfg.train_path, cfg.val_path, cfg.test_path]
    for p in paths_to_check:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "context" in df.columns:
                unique_contexts.update(df["context"].dropna().unique().tolist())
        else:
            print(f"Warning: Metadata file not found at {p}")

    # 3. Build the mapping
    # We look for an external CPC titles file (common in this task),
    # otherwise fallback to Section titles.
    cpc_titles_path = os.path.join(cfg.input_dir, "cpc_titles.csv")
    external_map = {}

    if os.path.exists(cpc_titles_path):
        print(f"Found external CPC titles file at {cpc_titles_path}")
        try:
            # Assuming standard format: code, title
            cpc_df = pd.read_csv(cpc_titles_path)
            # Normalize columns if necessary, assuming 'code' and 'title' exist
            # If not, we might need heuristics, but usually it's standard.
            # For robustness, we'll skip if columns don't match expected patterns
            if "code" in cpc_df.columns and "title" in cpc_df.columns:
                external_map = dict(zip(cpc_df["code"], cpc_df["title"]))
        except Exception as e:
            print(f"Error reading external CPC titles: {e}")

    data = []
    for code in sorted(list(unique_contexts)):
        text = ""
        section_char = code[0]

        # Base: Section Title
        section_title = CPC_SECTIONS.get(section_char, "")

        # Specific: Class/Subclass Title from external file if available
        specific_title = external_map.get(code, "")

        if specific_title:
            # If we have a specific title, combine them
            # e.g. "Human Necessities [SEP] Furniture"
            # We just join them with a space or punctuation; the model handles tokenization.
            if section_title and section_title.lower() not in specific_title.lower():
                text = f"{section_title}; {specific_title}"
            else:
                text = specific_title
        else:
            # Fallback: Just the section title
            text = section_title

        # If text is still empty (invalid code), use the code itself as a fallback
        if not text:
            text = f"CPC Code {code}"

        data.append({"context": code, "context_text": text})

    df_map = pd.DataFrame(data)

    # 4. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_map.to_parquet(cache_path, index=False)
    print(f"Saved CPC context map to {cache_path} with {len(df_map)} entries.")

    return df_map
