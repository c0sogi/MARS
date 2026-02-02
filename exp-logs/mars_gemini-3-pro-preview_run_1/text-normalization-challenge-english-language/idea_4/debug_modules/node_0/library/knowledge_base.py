import os
import pandas as pd
from library.config import ProjectConfig
from library.data_utils import load_dataset_raw


def build_knowledge_base(load_cached_data=True):
    """
    Builds or loads a deterministic Knowledge Base (KB) mapping
    (raw_token, class) -> normalized_text.

    Logic:
    1. Check if cached parquet exists.
    2. If yes and load_cached_data is True, load and return.
    3. If no, load raw train data.
    4. Compute the most frequent 'after' value for each (before, class) pair.
    5. Save to parquet.
    6. Return dictionary.

    Args:
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        dict: Mapping {(before, class): after}
    """
    kb_path = ProjectConfig.KNOWLEDGE_BASE_PATH

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(kb_path):
        print(f"Loading Knowledge Base from {kb_path}...")
        try:
            df_kb = pd.read_parquet(kb_path)
            # Convert to dictionary for O(1) lookup
            kb_dict = {}
            # Iterating rows is slower than zip for large dfs, using zip
            for before, cls, after in zip(
                df_kb["before"], df_kb["class"], df_kb["after"]
            ):
                kb_dict[(before, cls)] = after
            print(f"Knowledge Base loaded. Size: {len(kb_dict)} entries.")
            return kb_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding from scratch.")

    # 2. Build from scratch
    print("Building Knowledge Base from training data...")

    # Load raw data
    df_train = load_dataset_raw("train")

    # Select relevant columns and ensure strings
    df_subset = df_train[["before", "class", "after"]].astype(str)

    # 3. Resolve Ambiguities by Frequency (Mode)
    # We want the most common normalization for a given (token, class) pair.
    print("Aggregating frequencies...")

    # Group by all columns to count occurrences of each specific mapping
    # This creates a Series with the count as values
    counts = (
        df_subset.groupby(["before", "class", "after"]).size().reset_index(name="count")
    )

    # Sort by count descending so the most frequent appears first
    counts = counts.sort_values(by=["count"], ascending=False)

    # Drop duplicates based on the input key (before, class), keeping the first (most frequent)
    kb_df = counts.drop_duplicates(subset=["before", "class"], keep="first")

    # Clean up
    kb_df = kb_df[["before", "class", "after"]].reset_index(drop=True)

    # 4. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    print(f"Saving Knowledge Base to {kb_path}...")
    kb_df.to_parquet(kb_path, index=False)

    # 5. Convert to Dictionary
    kb_dict = {}
    for before, cls, after in zip(kb_df["before"], kb_df["class"], kb_df["after"]):
        kb_dict[(before, cls)] = after

    print(f"Knowledge Base built. Size: {len(kb_dict)} entries.")
    return kb_dict


def query_kb(kb, token, class_name):
    """
    Look up a token in the Knowledge Base.

    Args:
        kb (dict): The knowledge base dictionary.
        token (str): The raw token text.
        class_name (str): The predicted class of the token.

    Returns:
        str or None: The normalized text if found, else None.
    """
    return kb.get((str(token), str(class_name)), None)
