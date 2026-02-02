import os
import json
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    ANCHOR_CHAR_LIMIT,
    TOP_K_KEYWORDS,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import seed_everything

seed_everything(SEED)


def _get_python_stopwords():
    """
    Returns a set of common Python keywords and data science library terms
    to be used as stopwords for keyword extraction.
    """
    return {
        "import",
        "from",
        "def",
        "return",
        "class",
        "self",
        "print",
        "if",
        "else",
        "elif",
        "for",
        "in",
        "while",
        "break",
        "continue",
        "pass",
        "try",
        "except",
        "finally",
        "raise",
        "with",
        "as",
        "lambda",
        "yield",
        "global",
        "nonlocal",
        "assert",
        "del",
        "and",
        "or",
        "not",
        "is",
        "None",
        "True",
        "False",
        "np",
        "pd",
        "plt",
        "sns",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "os",
        "sys",
        "sklearn",
        "torch",
        "tensorflow",
        "keras",
        "x",
        "y",
        "data",
        "df",
    }


def _extract_keywords(code_text, top_k=TOP_K_KEYWORDS):
    """
    Extracts top K frequent words from code text, ignoring python stopwords.
    """
    if not code_text or not code_text.strip():
        return ""

    try:
        # Use CountVectorizer to get term frequencies
        # We use a simple regex for tokens: alphanumeric strings of length 2+
        vectorizer = CountVectorizer(
            stop_words=list(_get_python_stopwords()),
            token_pattern=r"(?u)\b\w\w+\b",
            max_features=top_k * 2,  # Get a bit more to filter if needed
        )
        X = vectorizer.fit_transform([code_text])

        # Sum counts (though we only have 1 document)
        counts = X.toarray().flatten()
        vocab = vectorizer.get_feature_names_out()

        # Sort by count descending
        indices = np.argsort(counts)[::-1][:top_k]
        top_words = [vocab[i] for i in indices if counts[i] > 0]

        return " ".join(top_words)
    except ValueError:
        # Handle cases with empty vocabulary or other vectorizer errors
        return ""


def _process_single_notebook(row, partition):
    """
    Processes a single notebook to extract markdown cells, targets, and context anchors.
    """
    nb_id = row["id"]
    filepath = os.path.join(INPUT_DIR, row["filepath"])

    # Load JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

    cell_types = nb_json.get("cell_type", {})
    sources = nb_json.get("source", {})

    # Determine Code Cell Order
    code_cell_ids = []

    if partition in ["train", "val"]:
        # For training/validation, we have the ground truth order
        if isinstance(row["cell_order"], str):
            full_order = row["cell_order"].split()
        else:
            full_order = []

        # Filter for code cells
        code_cell_ids = [cid for cid in full_order if cell_types.get(cid) == "code"]

        # Create a rank map for all cells to calculate targets
        rank_map = {cid: i for i, cid in enumerate(full_order)}
        total_cells = len(full_order)
    else:
        # For test, we infer code order from the JSON file structure.
        # We assume the JSON keys (when iterated) or the implicit structure
        # provides the relative order of code cells.
        # In this dataset, we collect all keys where cell_type is code.
        code_cell_ids = [cid for cid, ctype in cell_types.items() if ctype == "code"]
        # Note: In Python 3.7+, dict insertion order is preserved.
        # We assume the file was read in order.

        rank_map = {}
        total_cells = 0  # Not needed for test inference

    # Extract Anchors
    # 1. Start Anchor: First code cell
    start_anchor = ""
    if code_cell_ids:
        first_code_id = code_cell_ids[0]
        start_anchor = sources.get(first_code_id, "")[:ANCHOR_CHAR_LIMIT]

    # 2. End Anchor: Last code cell
    end_anchor = ""
    if code_cell_ids:
        last_code_id = code_cell_ids[-1]
        end_anchor = sources.get(last_code_id, "")[-ANCHOR_CHAR_LIMIT:]

    # 3. Topic Anchor: Keywords from all code
    # Aggregate all code text
    all_code_text = " ".join([sources.get(cid, "") for cid in code_cell_ids])
    topic_keywords = _extract_keywords(all_code_text)

    # Construct Context String
    # Format: [START] ... [END] ... [KEYWORDS] ...
    # We use simple separators. The tokenizer will handle them.
    context_str = f"START {start_anchor} END {end_anchor} KEYWORDS {topic_keywords}"

    # Clean up newlines in context to keep it as a single line feature if needed,
    # though transformers handle newlines fine. We'll strip extra whitespace.
    context_str = " ".join(context_str.split())

    # Process Markdown Cells
    processed_cells = []

    # Identify markdown cells
    if partition in ["train", "val"]:
        # Iterate in order to get ranks easily, though we computed rank_map
        md_cell_ids = [cid for cid in full_order if cell_types.get(cid) == "markdown"]
    else:
        md_cell_ids = [cid for cid, ctype in cell_types.items() if ctype == "markdown"]

    code_ids_str = " ".join(code_cell_ids)

    for cid in md_cell_ids:
        text = sources.get(cid, "")

        # Calculate Rank
        rank = np.nan
        if partition in ["train", "val"]:
            r = rank_map.get(cid, 0)
            # Normalize rank: 0.0 to 1.0
            if total_cells > 1:
                rank = r / (total_cells - 1)
            else:
                rank = 0.0

        processed_cells.append(
            {
                "id": nb_id,
                "cell_id": cid,
                "text": text,
                "context": context_str,
                "rank": rank,
                "code_cell_ids": code_ids_str,
                "partition": partition,
            }
        )

    return processed_cells


def load_notebook_data(partition="train", load_cached_data=True):
    """
    Loads and processes notebook data.

    Args:
        partition (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from Parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe with columns:
                      ['id', 'cell_id', 'text', 'context', 'rank', 'code_cell_ids', 'partition']
    """
    # Define cache path
    if partition == "train":
        cache_path = TRAIN_CACHE_PATH
        meta_path = TRAIN_METADATA_PATH
    elif partition == "val":
        cache_path = VAL_CACHE_PATH
        meta_path = VAL_METADATA_PATH
    elif partition == "test":
        cache_path = TEST_CACHE_PATH
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid partition: {partition}")

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {partition} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {partition} data from scratch...")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Debugging: Sample data
    if DEBUG:
        print(f"DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} notebooks.")
        df_meta = df_meta.iloc[:DEBUG_SAMPLE_SIZE].copy()

    # Process notebooks
    all_data = []

    # Iterate through notebooks (using standard loop to avoid tqdm dependency)
    total = len(df_meta)
    for idx, row in df_meta.iterrows():
        if idx % 1000 == 0:
            print(f"Processed {idx}/{total} notebooks...", end="\r")

        notebook_data = _process_single_notebook(row, partition)
        all_data.extend(notebook_data)

    print(f"Processed {total}/{total} notebooks.       ")

    # Create DataFrame
    df_processed = pd.DataFrame(all_data)

    # Ensure columns are correct types
    if not df_processed.empty:
        df_processed["rank"] = df_processed["rank"].astype(float)
        df_processed["text"] = df_processed["text"].astype(str)
        df_processed["context"] = df_processed["context"].astype(str)

    # Save to Cache
    # Create directory if it doesn't exist (handled by config, but good to ensure)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(f"Saving processed data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed
