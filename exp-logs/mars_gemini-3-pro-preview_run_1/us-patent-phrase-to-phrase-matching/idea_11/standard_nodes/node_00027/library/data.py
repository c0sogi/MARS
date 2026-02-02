import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, List, Tuple, Optional

from library.config import Config
from library.utils import seed_everything


def get_cpc_texts() -> Dict[str, str]:
    """
    Parses the description.md file to create a mapping of CPC codes to their
    hierarchical textual descriptions.

    Logic:
    - Reads input/description.md.
    - Stores code -> description mapping.
    - For a given context (e.g., 'A47'), constructs 'Section Desc; Class Desc'.
    """
    cpc_codes = {}

    # Check if file exists, otherwise return empty dict or handle error
    if not os.path.exists(Config.cpc_codes_path):
        print(f"Warning: {Config.cpc_codes_path} not found. Using raw codes.")
        return {}

    with open(Config.cpc_codes_path, "r") as f:
        # Assuming format: Code Description (space separated or similar)
        # Based on typical dataset structure, we'll try to parse robustly
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Simple heuristic: first token is code, rest is description
            parts = line.split(" ", 1)
            if len(parts) == 2:
                code, desc = parts
                cpc_codes[code] = desc

    return cpc_codes


def get_hierarchical_context(context_code: str, cpc_map: Dict[str, str]) -> str:
    """
    Constructs the full hierarchical context string.
    Example: context_code='A47' -> Section 'A' desc + '; ' + Class 'A47' desc.
    """
    if not context_code:
        return ""

    section_code = context_code[0]

    section_desc = cpc_map.get(section_code, "")
    class_desc = cpc_map.get(context_code, "")

    # Construct hierarchy
    parts = [p for p in [section_desc, class_desc] if p]

    if not parts:
        # Fallback if map is empty or keys missing
        return context_code

    return "; ".join(parts)


def get_data(
    load_cached_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from metadata CSVs, enriches with CPC context texts,
    and handles caching via Parquet files.
    """
    # Define cache paths
    cache_train_path = os.path.join(Config.cache_dir, "cached_train.parquet")
    cache_val_path = os.path.join(Config.cache_dir, "cached_val.parquet")
    cache_test_path = os.path.join(Config.cache_dir, "cached_test.parquet")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading data from cache...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)
            return train_df, val_df, test_df
        else:
            print("Cache not found. Processing from scratch...")

    # 2. Process from Scratch
    print("Loading metadata...")
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Debug mode: subsample
    if Config.debug:
        print(f"Debug mode enabled. Subsampling {Config.debug_sample_size} rows.")
        train_df = train_df.head(Config.debug_sample_size)
        val_df = val_df.head(Config.debug_sample_size)
        test_df = test_df.head(Config.debug_sample_size)

    # Load CPC texts
    cpc_map = get_cpc_texts()

    # Apply context expansion
    print("Expanding context...")

    def apply_context(code):
        return get_hierarchical_context(code, cpc_map)

    train_df["context_text"] = train_df["context"].apply(apply_context)
    val_df["context_text"] = val_df["context"].apply(apply_context)
    test_df["context_text"] = test_df["context"].apply(apply_context)

    # 3. Save to Cache
    print(f"Saving processed data to {Config.cache_dir}...")
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df


class PatentDataset(Dataset):
    """
    Dataset class for Native Pair Encoding.
    Constructs inputs as:
    Segment A: Context + [SEP] + Anchor
    Segment B: Target
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-extract columns to lists for faster access
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context_text"].values

        # Check if labels exist
        if Config.target_col in df.columns:
            self.labels = df[Config.target_col].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Native Pair Encoding Strategy
        # We want the model to see: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # But specifically using token_type_ids to distinguish segments.
        # DeBERTa treats text_pair as the second segment.
        # We combine Context and Anchor into the first segment.

        # Segment A construction
        # Note: We add the separator manually if the tokenizer doesn't handle
        # complex multi-part segment A automatically.
        # A robust way is: text = context + tokenizer.sep_token + anchor
        text_segment_a = (
            str(context) + " " + self.tokenizer.sep_token + " " + str(anchor)
        )
        text_segment_b = str(target)

        inputs = self.tokenizer(
            text=text_segment_a,
            text_pair=text_segment_b,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=True,  # Important for DeBERTa/BERT
            return_attention_mask=True,
            return_tensors=None,  # Return python lists for Collate to handle
        )

        sample = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if "token_type_ids" in inputs:
            sample["token_type_ids"] = inputs["token_type_ids"]

        if self.labels is not None:
            sample["label"] = float(self.labels[idx])

        return sample


class Collate:
    """
    Collator for Dynamic Padding.
    Pads the batch to the length of the longest sequence in that batch.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]):
        # Find max length in this batch
        max_len = max(len(x["input_ids"]) for x in batch)

        # Prepare output containers
        input_ids = []
        attention_masks = []
        token_type_ids = []
        labels = []

        has_token_types = "token_type_ids" in batch[0]
        has_labels = "label" in batch[0]

        for x in batch:
            # Pad input_ids
            ids = x["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)

            # Pad attention_mask (1 for real, 0 for pad)
            mask = x["attention_mask"]
            attention_masks.append(mask + [0] * pad_len)

            # Pad token_type_ids (usually 0 for pad, but specific to model)
            if has_token_types:
                tt_ids = x["token_type_ids"]
                # Padding usually takes token_type_id 0
                token_type_ids.append(tt_ids + [0] * pad_len)

            if has_labels:
                labels.append(x["label"])

        # Convert to tensors
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        if has_token_types:
            out["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)

        if has_labels:
            out["labels"] = torch.tensor(labels, dtype=torch.float)

        return out


def get_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation.
    """
    train_ds = PatentDataset(train_df, tokenizer, Config.max_length)
    val_ds = PatentDataset(val_df, tokenizer, Config.max_length)

    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(
    test_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase
) -> DataLoader:
    """
    Creates DataLoader for inference.
    """
    test_ds = PatentDataset(test_df, tokenizer, Config.max_length)
    collate_fn = Collate(tokenizer)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
