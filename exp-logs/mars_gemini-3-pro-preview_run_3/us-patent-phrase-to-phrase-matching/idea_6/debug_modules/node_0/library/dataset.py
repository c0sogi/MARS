import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import cfg
from library.utils import get_logger, seed_everything
from library.cpc_utils import CPCMapper

# Initialize logger
logger = get_logger(os.path.join(cfg.working_dir, "dataset.log"))


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Matching task.
    Constructs hierarchical inputs: [CLS] Section [SEP] Class [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, data, tokenizer, max_len=128):
        """
        Args:
            data (pd.DataFrame): DataFrame containing 'anchor', 'target', 'context_text', etc.
            tokenizer (PreTrainedTokenizer): Transformer tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Column names
        self.text_col = "context_text"
        self.anchor_col = "anchor"
        self.target_col = "target"
        self.score_col = "score"
        self.id_col = "id"

        # Check if targets are available
        self.has_score = self.score_col in data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Retrieve Raw Text
        # CPCMapper provides context as "Section Description [SEP] Class Description"
        # We need to handle the [SEP] string carefully to convert it to a special token
        context_text = str(row[self.text_col])
        anchor = str(row[self.anchor_col])
        target = str(row[self.target_col])

        # 2. Construct Segments
        # Split context into its hierarchical components based on the string separator used in CPCMapper
        # Note: CPCMapper uses " [SEP] " as the delimiter.
        context_parts = context_text.split(" [SEP] ")

        # Define the sequence of segments: Section -> Class -> Anchor -> Target
        segments = context_parts + [anchor, target]

        # 3. Build Input IDs manually to ensure correct structure
        # Structure: [CLS] Seg1 [SEP] Seg2 [SEP] ... SegN [SEP]
        input_ids = [self.tokenizer.cls_token_id]

        for seg in segments:
            # Encode segment without special tokens
            seg_ids = self.tokenizer.encode(str(seg), add_special_tokens=False)
            input_ids.extend(seg_ids)
            # Add Separator
            input_ids.append(self.tokenizer.sep_token_id)

        # 4. Truncation
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            # Optionally ensure the last token is SEP, but simple truncation is standard and safe for DeBERTa

        # 5. Attention Mask
        attention_mask = [1] * len(input_ids)

        # 6. Convert to Tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": row[self.id_col],
        }

        # 7. Handle Targets
        if self.has_score:
            score = float(row[self.score_col])

            # Regression Target
            result["labels"] = torch.tensor(score, dtype=torch.float)

            # Classification Target (Auxiliary Head)
            # Map 0.0-1.0 to integers 0-4
            # 0.00 -> 0, 0.25 -> 1, 0.50 -> 2, 0.75 -> 3, 1.00 -> 4
            class_label = int(round(score * 4))
            result["class_labels"] = torch.tensor(class_label, dtype=torch.long)

        return result


class CollateFn:
    """
    Custom collate function to handle dynamic padding.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract fields
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        ids = [item["id"] for item in batch]

        # Pad sequences
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch_output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": ids,
        }

        # Stack targets if they exist
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            class_labels = torch.stack([item["class_labels"] for item in batch])
            batch_output["labels"] = labels
            batch_output["class_labels"] = class_labels

        return batch_output


def prepare_data(split="train", load_cached_data=True, debug=cfg.debug):
    """
    Loads, processes, and caches the dataset for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, returns a small subset of the data.

    Returns:
        pd.DataFrame: The processed dataframe with context text merged.
    """
    seed_everything(cfg.seed)

    # Determine paths based on split
    if split == "train":
        metadata_path = os.path.join(cfg.metadata_dir, "train.csv")
        cache_path = cfg.train_cache_path
    elif split == "val":
        metadata_path = os.path.join(cfg.metadata_dir, "val.csv")
        cache_path = cfg.val_cache_path
    elif split == "test":
        metadata_path = os.path.join(cfg.metadata_dir, "test.csv")
        cache_path = cfg.test_cache_path
    else:
        raise ValueError(f"Invalid split: {split}")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # Apply debug sampling after loading to avoid corrupting cache logic
            if debug:
                logger.info(f"Debug mode: Sampling {min(len(df), 100)} rows.")
                df = df.sample(n=min(len(df), 100), random_state=cfg.seed).reset_index(
                    drop=True
                )
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    logger.info(f"Processing {split} data from scratch...")

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} rows from {metadata_path}")

    # Load Context Map
    cpc_mapper = CPCMapper()
    # We pass load_cached_data to the mapper as well to respect the global flag
    context_map = cpc_mapper.run(load_cached_data=load_cached_data)

    # Merge Context
    # context_map has columns ['context', 'context_text']
    df = df.merge(context_map, on="context", how="left")

    # Fill missing contexts (if any)
    df["context_text"] = df["context_text"].fillna("")

    # 3. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info(f"Saving processed {split} data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    # 4. Debug Sampling
    if debug:
        logger.info(f"Debug mode: Sampling {min(len(df), 100)} rows.")
        df = df.sample(n=min(len(df), 100), random_state=cfg.seed).reset_index(
            drop=True
        )

    return df
