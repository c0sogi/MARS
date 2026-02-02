import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class PearsonDataset(Dataset):
    """
    PyTorch Dataset for the Semantic Similarity Task.
    Handles tokenization of (Context, Anchor, Target) triplets and manages caching of processed tensors.
    """

    def __init__(self, df, tokenizer, cpc_texts, mode, cfg, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'anchor', 'target', 'context', and optionally 'score'.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            cpc_texts (dict): Mapping from CPC code to description.
            mode (str): Dataset mode ('train', 'val', 'test') used for cache naming.
            cfg (Config): Configuration object.
            load_cached_data (bool): Whether to load from cache if available.
        """
        self.cfg = cfg
        self.mode = mode

        # Construct cache path
        # Include debug flag in filename to prevent overwriting full cache with debug data
        debug_suffix = "_debug" if cfg.debug else ""
        self.cache_path = os.path.join(
            cfg.working_dir, f"cached_{mode}{debug_suffix}.npz"
        )

        # Ensure working directory exists
        os.makedirs(cfg.working_dir, exist_ok=True)

        # Logic: Load from cache OR Process from scratch
        if load_cached_data and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._process_and_cache(df, tokenizer, cpc_texts)

    def _load_cache(self):
        """Loads processed tensors from the numpy cache file."""
        # print(f"Loading {self.mode} dataset from cache: {self.cache_path}")
        try:
            data = np.load(self.cache_path)
            self.input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
            self.attention_mask = torch.tensor(data["attention_mask"], dtype=torch.long)

            if "labels" in data:
                self.labels = torch.tensor(data["labels"], dtype=torch.float)
            else:
                self.labels = None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cache from {self.cache_path}. Error: {e}"
            )

    def _process_and_cache(self, df, tokenizer, cpc_texts):
        """
        Processes the dataframe:
        1. Maps CPC codes to descriptions.
        2. Constructs input text: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        3. Tokenizes.
        4. Saves to cache.
        """
        # print(f"Processing {self.mode} dataset from scratch...")

        # 1. Map Context
        # Use .get to handle potential missing keys safely (fallback to code itself)
        contexts = df["context"].apply(lambda x: cpc_texts.get(x, str(x)))
        anchors = df["anchor"].fillna("")
        targets = df["target"].fillna("")

        # 2. Construct Text
        # Format: Context Description [SEP] Anchor [SEP] Target
        # We insert the separator token manually.
        # Ensure spaces around SEP to avoid token merging issues if tokenizer doesn't handle it
        sep = tokenizer.sep_token
        texts = contexts + f" {sep} " + anchors + f" {sep} " + targets
        texts_list = texts.tolist()

        # 3. Tokenize
        # DeBERTa tokenizer handles [CLS] at start and [SEP] at end automatically
        encoded = tokenizer.batch_encode_plus(
            texts_list,
            add_special_tokens=True,
            max_length=self.cfg.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # 4. Handle Labels
        labels = None
        if "score" in df.columns:
            labels = df["score"].values.astype(np.float32)

        # 5. Save to Cache
        save_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            save_dict["labels"] = labels

        np.savez(self.cache_path, **save_dict)
        # print(f"Saved processed data to {self.cache_path}")

        # 6. Assign to self
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = (
            torch.tensor(labels, dtype=torch.float) if labels is not None else None
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
