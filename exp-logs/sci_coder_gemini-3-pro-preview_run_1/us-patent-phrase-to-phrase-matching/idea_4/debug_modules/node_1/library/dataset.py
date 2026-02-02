import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def load_processed_data(csv_path, cpc_texts, load_cached_data=True, debug=False):
    """
    Loads and processes the dataset. Implements caching to Parquet.

    Args:
        csv_path (str): Path to the source CSV file (train/val/test).
        cpc_texts (dict): Mapping from CPC code to description.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, loads a subset of data and uses a debug cache file.

    Returns:
        pd.DataFrame: The processed dataframe with 'context_text' column.
    """
    # 1. Construct Cache Path
    filename = os.path.basename(csv_path).replace(".csv", "")
    if debug:
        filename += "_debug"

    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)
    cache_path = os.path.join(Config.output_dir, f"cached_{filename}.parquet")

    # 2. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {filename} from cache.")
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 3. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file {csv_path} not found.")

    df = pd.read_csv(csv_path)

    if debug:
        df = df.head(100).copy()

    # Map context codes to descriptions
    # Use fillna("") to handle potential missing codes gracefully
    df["context_text"] = df["context"].map(cpc_texts).fillna("")

    # 4. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved processed {filename} to cache.")
    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")

    return df


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for Phrase Similarity.
    Constructs input in the format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, df, tokenizer, max_length=130):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing 'anchor', 'target', 'context_text'.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-extract columns to numpy arrays for faster access in __getitem__
        self.anchors = df["anchor"].astype(str).values
        self.targets = df["target"].astype(str).values
        self.contexts = df["context_text"].astype(str).values
        self.ids = df["id"].astype(str).values

        # Handle labels if they exist (Train/Val) vs Test
        if "score" in df.columns:
            self.labels = df["score"].values.astype(float)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Tokenize components separately without special tokens
        # We manually construct the sequence to ensure the specific structure:
        # [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        tok_context = self.tokenizer.encode(context, add_special_tokens=False)
        tok_anchor = self.tokenizer.encode(anchor, add_special_tokens=False)
        tok_target = self.tokenizer.encode(target, add_special_tokens=False)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Calculate lengths
        # Structure: CLS + Context + SEP + Anchor + SEP + Target + SEP
        # Fixed tokens count = 4 (1 CLS + 3 SEP)
        fixed_overhead = 4

        # Strategy: Prioritize Anchor and Target. Truncate Context if necessary.
        required_len = len(tok_anchor) + len(tok_target) + fixed_overhead
        allowed_context_len = self.max_length - required_len

        if allowed_context_len < 0:
            # Extremely rare case where anchor + target > max_length
            # We truncate context entirely and truncate target/anchor
            tok_context = []
            # Re-calculate remaining space for anchor/target
            remaining = self.max_length - fixed_overhead
            # Simple truncation of the concatenated anchor+target logic
            # But let's just truncate the constructed list at the end for safety
        else:
            # Truncate context to fit
            if len(tok_context) > allowed_context_len:
                tok_context = tok_context[:allowed_context_len]

        # Construct Input IDs
        input_ids = (
            [cls_id]
            + tok_context
            + [sep_id]
            + tok_anchor
            + [sep_id]
            + tok_target
            + [sep_id]
        )

        # Final safety truncation (in case of the rare overflow mentioned above)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            # Ensure the last token is SEP if we truncated
            if input_ids[-1] != sep_id:
                input_ids[-1] = sep_id

        # Create Attention Mask
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to Tensors
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "id": self.ids[idx],
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
