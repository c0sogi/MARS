import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from library.config import Config
from library.utils import read_notebook, preprocess_text


class NotebookDataset(Dataset):
    """
    Dataset wrapper for SentenceTransformers.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_relaxed_pairs(df_metadata, input_dir):
    """
    Generates (Markdown, Code) pairs where Code is the nearest subsequent code cell.
    """
    pairs = []

    for _, row in df_metadata.iterrows():
        nb_id = row["id"]
        file_path = os.path.join(input_dir, row["file_path"])

        # Ensure cell_order exists (it should for training data)
        if "cell_order" not in row or pd.isna(row["cell_order"]):
            continue

        cell_order = str(row["cell_order"]).split()
        cell_types, sources = read_notebook(file_path)

        if not cell_types:
            continue

        # Create a map of cell_id -> rank
        rank_map = {cid: i for i, cid in enumerate(cell_order)}

        code_cells = []
        markdown_cells = []

        # Separate cells while preserving relative order from cell_order
        for cid in cell_order:
            if cid not in cell_types:
                continue
            ctype = cell_types[cid]
            if ctype == "code":
                code_cells.append(cid)
            elif ctype == "markdown":
                markdown_cells.append(cid)

        # Pair generation: Markdown -> Nearest Subsequent Code
        for md_id in markdown_cells:
            md_rank = rank_map[md_id]
            target_code_id = None

            # Find the first code cell with rank > md_rank
            for code_id in code_cells:
                if rank_map[code_id] > md_rank:
                    target_code_id = code_id
                    break

            if target_code_id:
                md_text = preprocess_text(sources.get(md_id, ""))
                code_text = preprocess_text(sources.get(target_code_id, ""))

                # Filter empty texts to avoid noise
                if md_text and code_text:
                    pairs.append({"text_anchor": md_text, "text_positive": code_text})

    return pd.DataFrame(pairs)


def prepare_training_pairs(load_cached_data=True, debug=False):
    """
    Main function to prepare data for the contrastive fine-tuning stage.
    Handles caching logic using Parquet.
    """
    filename = "train_pairs_debug.parquet" if debug else "train_pairs_relaxed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training pairs from {cache_path}")
        try:
            df_pairs = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")
            load_cached_data = False  # Force regenerate

    # 2. Generate if not loaded
    if not load_cached_data or not os.path.exists(cache_path):
        print("Generating training pairs from scratch...")
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

        if debug:
            print(f"Debug mode: sampling {Config.DEBUG_SAMPLE_SIZE} notebooks.")
            df_train = df_train.sample(
                n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        df_pairs = get_relaxed_pairs(df_train, Config.INPUT_DIR)

        # 3. Save Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        print(f"Saving {len(df_pairs)} pairs to {cache_path}")
        df_pairs.to_parquet(cache_path, index=False)

    # 4. Convert to InputExample objects
    examples = []
    for _, row in df_pairs.iterrows():
        examples.append(InputExample(texts=[row["text_anchor"], row["text_positive"]]))

    return examples


def load_notebook_data(metadata_path, debug=False):
    """
    Loads notebook data into a structured dictionary for the regression stage.
    Returns: {nb_id: {'code': {id: text}, 'markdown': {id: text}, 'order': [id_list]}}
    """
    df = pd.read_csv(metadata_path)
    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        )

    notebooks = {}
    for _, row in df.iterrows():
        nb_id = row["id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        cell_order = []
        if "cell_order" in row and not pd.isna(row["cell_order"]):
            cell_order = str(row["cell_order"]).split()

        cell_types, sources = read_notebook(file_path)
        if not cell_types:
            continue

        nb_data = {"code": {}, "markdown": {}, "order": cell_order}

        for cid, ctype in cell_types.items():
            text = preprocess_text(sources.get(cid, ""))
            if ctype == "code":
                nb_data["code"][cid] = text
            elif ctype == "markdown":
                nb_data["markdown"][cid] = text

        notebooks[nb_id] = nb_data

    return notebooks
