import os
import json
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def _read_notebook(filepath, correct_order=None):
    """
    Reads a notebook JSON file and extracts markdown cells with their ranks
    and a consolidated string of all code content.

    Args:
        filepath (str): Relative path to the JSON file.
        correct_order (str, optional): Space-delimited string of cell IDs in correct order.

    Returns:
        tuple: (list of dicts for markdown cells, string of concatenated code)
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return [], ""

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    # Determine the order to iterate through cells
    if correct_order:
        cell_order = correct_order.split()
    else:
        # For test set, use keys from JSON (order doesn't matter for extraction)
        cell_order = list(cell_types.keys())

    markdown_cells = []
    code_texts = []

    total_cells = len(cell_order)

    for rank, cell_id in enumerate(cell_order):
        ctype = cell_types.get(cell_id, "")
        source = sources.get(cell_id, "")

        if ctype == "markdown":
            # Calculate normalized rank: 0.0 (top) to 1.0 (bottom)
            norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

            markdown_cells.append(
                {"cell_id": cell_id, "source": source, "rank": norm_rank}
            )
        elif ctype == "code":
            code_texts.append(source)

    # Concatenate all code cells to form the notebook's technical context
    full_code_text = " ".join(code_texts)

    return markdown_cells, full_code_text


def _process_raw_data(df_meta, mode):
    """
    Iterates through the metadata dataframe, parses JSON files, and structures the data.
    """
    all_md_rows = []
    notebook_code_map = {}  # Mapping: notebook_id -> full_code_text

    # Use tqdm for progress bar
    iterator = tqdm(
        df_meta.iterrows(), total=len(df_meta), desc=f"Parsing {mode} notebooks"
    )

    for _, row in iterator:
        nb_id = row["id"]
        filepath = row["filepath"]

        # 'cell_order' exists for train/val; None for test
        correct_order = row["cell_order"] if "cell_order" in row else None

        md_cells, code_text = _read_notebook(filepath, correct_order)

        notebook_code_map[nb_id] = code_text

        for cell in md_cells:
            cell["id"] = nb_id  # Add notebook ID to the cell record
            all_md_rows.append(cell)

    df_markdown = pd.DataFrame(all_md_rows)
    return df_markdown, notebook_code_map


def _extract_and_add_context(df_markdown, notebook_code_map, mode):
    """
    Extracts high-signal keywords from code content using TF-IDF and adds them
    as a 'context' column to the dataframe.
    """
    # Create corpus aligned with notebook IDs
    nb_ids = list(notebook_code_map.keys())
    corpus = [notebook_code_map[nid] for nid in nb_ids]

    vectorizer_path = Config.CODE_TFIDF_VECTORIZER_PATH

    # TF-IDF Logic: Fit on Train, Transform on Val/Test
    if mode == "train":
        print("Fitting Code TF-IDF Vectorizer for Context Extraction...")
        # We use a limited vocabulary to capture only common technical terms (e.g., 'pandas', 'train')
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",  # Allow underscores
            min_df=5,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Save vectorizer for inference
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)

    else:
        print("Loading Code TF-IDF Vectorizer...")
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            tfidf_matrix = vectorizer.transform(corpus)
        else:
            # Fallback (should not happen in correct pipeline order)
            print(
                "Warning: Vectorizer not found. Fitting on current data (suboptimal)."
            )
            vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)

    # Extract top K keywords for each notebook
    feature_names = np.array(vectorizer.get_feature_names_out())
    context_map = {}

    print("Extracting top keywords per notebook...")
    # Iterate through rows of the sparse matrix
    for i, nid in enumerate(tqdm(nb_ids, desc="Extracting keywords")):
        row = tfidf_matrix[i]
        _, col_indices = row.nonzero()

        if len(col_indices) == 0:
            context_map[nid] = ""
            continue

        # Get TF-IDF scores and sort descending
        data = row.data
        sorted_indices = np.argsort(data)[::-1]

        # Select top K
        top_k_indices = sorted_indices[: Config.MAX_CODE_TOKENS_CONTEXT]
        top_feat_indices = col_indices[top_k_indices]

        keywords = feature_names[top_feat_indices]
        context_map[nid] = " ".join(keywords)

    # Map the extracted context back to the markdown dataframe
    print("Mapping context to dataframe...")
    df_markdown["context"] = df_markdown["id"].map(context_map)
    df_markdown["context"] = df_markdown["context"].fillna("")

    return df_markdown


def load_data(split="train", debug=False, load_cached_data=True):
    """
    Main entry point to load and preprocess data.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, processes only a small subset for debugging.
        load_cached_data (bool): If True, attempts to load from Parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing ['id', 'cell_id', 'source', 'context', 'rank'].
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if debug:
            df = df.head(1000)
        return df

    # 2. Process from Scratch
    print(f"Processing {split} data from scratch...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    if debug:
        df_meta = df_meta.head(100)

    # Step A: Parse JSONs to get Markdown and raw Code
    df_markdown, notebook_code_map = _process_raw_data(df_meta, split)

    # Step B: Generate Context from Code
    df_processed = _extract_and_add_context(df_markdown, notebook_code_map, split)

    # Step C: Save Cache
    if not debug:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving processed data to {cache_path}...")
        df_processed.to_parquet(cache_path, index=False)

    return df_processed


class NotebookDataset(Dataset):
    """
    PyTorch Dataset for the Context-Aware Transformer.
    Combines Markdown Source and Code Context into a single input sequence.
    """

    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Convert to list for faster access
        self.sources = self.df["source"].astype(str).tolist()
        self.contexts = self.df["context"].astype(str).tolist()

        # Load labels if available
        if "rank" in self.df.columns:
            self.labels = self.df["rank"].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source = self.sources[idx]
        context = self.contexts[idx]

        # Tokenize: [CLS] source [SEP] context [SEP]
        # We use text_pair to handle the context as the second segment
        encoding = self.tokenizer(
            source,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
