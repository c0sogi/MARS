import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from library.config import Config
from library.tokenizers import HybridTokenizer


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Granularity Hybrid Neuro-Symbolic System.

    It consumes the dataframe prepared by DataManager, which contains:
    - context_left: List[str] (BPE source)
    - before: str (Character source)
    - context_right: List[str] (BPE source)
    - after: str (BPE target, optional)
    """

    def __init__(self, df: pd.DataFrame, tokenizer: HybridTokenizer, config: Config):
        """
        Args:
            df: DataFrame containing the processed sequences.
            tokenizer: Trained HybridTokenizer instance.
            config: Configuration object.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.config = config

        # Check if targets are available
        self.has_target = "after" in df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Use iloc to access by integer position regardless of DataFrame index
        row = self.df.iloc[idx]

        # 1. Extract Inputs
        # context columns are stored as lists/arrays in the Parquet file
        # Ensure they are lists of strings
        context_left = row["context_left"]
        if isinstance(context_left, np.ndarray):
            context_left = context_left.tolist()

        context_right = row["context_right"]
        if isinstance(context_right, np.ndarray):
            context_right = context_right.tolist()

        target_token = str(row["before"])

        # 2. Tokenize Inputs
        # Returns dict with keys: 'context_left_ids', 'target_char_ids', 'context_right_ids'
        encoded_inputs = self.tokenizer.encode(
            context_left, target_token, context_right
        )

        # Convert to tensors
        src_left = torch.tensor(encoded_inputs["context_left_ids"], dtype=torch.long)
        src_target = torch.tensor(encoded_inputs["target_char_ids"], dtype=torch.long)
        src_right = torch.tensor(encoded_inputs["context_right_ids"], dtype=torch.long)

        result = {
            "src_left": src_left,
            "src_target": src_target,
            "src_right": src_right,
            "original_before": target_token,
        }

        # Pass through ID if available (for submission mapping)
        if "id" in row:
            result["id"] = row["id"]

        # 3. Process Target (if training/val)
        if self.has_target:
            target_text = str(row["after"])
            # Encodes with SOS and EOS
            tgt_ids = self.tokenizer.encode_target_text(target_text)
            result["tgt"] = torch.tensor(tgt_ids, dtype=torch.long)
            result["original_after"] = target_text

        return result


class NormalizationCollator:
    """
    Custom collator to handle padding for multiple distinct sequences.
    """

    def __init__(self, bpe_pad_id: int, char_pad_id: int):
        self.bpe_pad_id = bpe_pad_id
        self.char_pad_id = char_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Collect lists of tensors
        src_left_list = [item["src_left"] for item in batch]
        src_target_list = [item["src_target"] for item in batch]
        src_right_list = [item["src_right"] for item in batch]

        # Pad sequences
        # batch_first=True -> [Batch, Seq_Len]
        src_left_padded = pad_sequence(
            src_left_list, batch_first=True, padding_value=self.bpe_pad_id
        )
        src_target_padded = pad_sequence(
            src_target_list, batch_first=True, padding_value=self.char_pad_id
        )
        src_right_padded = pad_sequence(
            src_right_list, batch_first=True, padding_value=self.bpe_pad_id
        )

        batch_out = {
            "src_left": src_left_padded,
            "src_target": src_target_padded,
            "src_right": src_right_padded,
            "original_before": [item["original_before"] for item in batch],
        }

        if "id" in batch[0]:
            batch_out["id"] = [item["id"] for item in batch]

        # Handle Target if present
        if "tgt" in batch[0]:
            tgt_list = [item["tgt"] for item in batch]
            tgt_padded = pad_sequence(
                tgt_list, batch_first=True, padding_value=self.bpe_pad_id
            )
            batch_out["tgt"] = tgt_padded
            batch_out["original_after"] = [item["original_after"] for item in batch]

        return batch_out
