import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import CFG
from library.cpc_loader import CPCLoader
from library.utils import get_logger

logger = get_logger("dataset.log")


def prepare_data(metadata_path, load_cached_data=True):
    """
    Loads metadata, merges with CPC context text, and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV file (train, val, or test).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'context_text' column.
    """
    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Create cache filename based on metadata filename hash or name
    filename = os.path.basename(metadata_path).replace(".csv", "_processed.parquet")
    cache_path = os.path.join(CFG.output_dir, filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Process from scratch
    logger.info(f"Processing data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Load Context Map using the library class
    cpc_loader = CPCLoader()
    # We always try to load cached context map to save time, as it is shared across splits
    df_context = cpc_loader.get_cpc_texts(load_cached_data=True)

    # Merge context descriptions
    # df has 'context', df_context has 'context', 'context_text'
    df = df.merge(df_context, on="context", how="left")

    # Fill missing context texts with empty string to avoid tokenizer errors
    df["context_text"] = df["context_text"].fillna("")

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved processed data to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")

    return df


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for Phrase Matching.
    Handles tokenization and input formatting for the Two-Stage Stratified Ensemble.
    """

    def __init__(self, df, tokenizer, max_len=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'anchor', 'target', 'context_text', and optionally 'score'.
            tokenizer: HuggingFace tokenizer.
            max_len (int): Maximum sequence length. Defaults to CFG.max_len.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len is not None else CFG.max_len

        # Pre-extract columns to numpy arrays for faster access during getitem
        self.anchors = df["anchor"].values.astype(str)
        self.targets = df["target"].values.astype(str)
        self.contexts = df["context_text"].values.astype(str)
        self.ids = df["id"].values

        # Handle labels if they exist (Train/Val mode)
        if "score" in df.columns:
            self.scores = df["score"].values.astype(np.float32)
            # Create classification labels: 0, 1, 2, 3, 4
            # Mapping: 0.0->0, 0.25->1, 0.5->2, 0.75->3, 1.0->4
            # We multiply by 4 and round to nearest integer to handle potential float precision issues
            self.labels_cls = np.round(self.scores * 4).astype(int)
        else:
            self.scores = None
            self.labels_cls = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context_text = self.contexts[idx]

        # Construct Input Text
        # Strategy: Context + [SEP] + Anchor [SEP] Target
        # We construct the first segment as "Context [SEP] Anchor" and the second as "Target".
        # When passed to tokenizer(text, text_pair), this typically results in:
        # [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # This structure allows the model to attend to the contextualized anchor vs the target.

        # Note: We add spaces around the separator to ensure clean tokenization
        text_first = f"{context_text} {self.tokenizer.sep_token} {anchor}"
        text_second = target

        inputs = self.tokenizer(
            text_first,
            text_second,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return standard python lists
        )

        # Convert to tensors
        ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        # Handle token_type_ids (Segment IDs)
        # DeBERTa v3 might not strictly require them, but we provide them if generated
        if "token_type_ids" in inputs:
            token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        else:
            token_type_ids = torch.zeros_like(ids)

        output = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids,
        }

        # Add labels if available
        if self.scores is not None:
            output["label"] = torch.tensor(self.scores[idx], dtype=torch.float)
            output["label_cls"] = torch.tensor(self.labels_cls[idx], dtype=torch.long)

        return output
