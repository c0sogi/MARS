import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.cpc_utils import CPCHelper


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (DAPT) using Masked Language Modeling (MLM).
    Aggregates text from Train, Val, Test, and Context descriptions.
    """

    def __init__(self, tokenizer, load_cached_data=True):
        self.tokenizer = tokenizer
        self.cache_path = os.path.join(Config.working_dir, "mlm_corpus.parquet")

        # Load or create the text corpus
        self.texts = self._load_corpus(load_cached_data)

        # Tokenize the corpus
        # We use fixed padding to Config.max_length to ensure consistency
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=Config.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    def _load_corpus(self, load_cached_data):
        """
        Loads the corpus from cache or generates it from raw files.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached MLM corpus from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                return df["text"].tolist()
            except Exception as e:
                print(f"Failed to load MLM cache: {e}. Recomputing...")

        print("Generating MLM corpus from sources...")
        cpc_helper = CPCHelper()

        # Load raw datasets
        train_df = pd.read_csv(Config.train_path)
        val_df = pd.read_csv(Config.val_path)
        test_df = pd.read_csv(Config.test_path)

        # Set of unique phrases to avoid redundancy
        phrases = set()

        # Add Anchors and Targets
        for df in [train_df, val_df, test_df]:
            if "anchor" in df.columns:
                phrases.update(df["anchor"].dropna().astype(str).tolist())
            if "target" in df.columns:
                phrases.update(df["target"].dropna().astype(str).tolist())

        # Add Context Descriptions
        # We use the helper to generate the master map of all contexts
        ctx_map = cpc_helper.generate_context_map(
            train_df,
            val_df,
            test_df,
            cache_path=Config.context_map_cache_path,
            load_cached_data=load_cached_data,
        )
        phrases.update(ctx_map["context_text"].dropna().astype(str).tolist())

        text_list = list(phrases)
        print(f"Compiled MLM corpus with {len(text_list)} unique texts.")

        # Save to cache
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            pd.DataFrame({"text": text_list}).to_parquet(self.cache_path, index=False)
            print(f"Saved MLM corpus to {self.cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save MLM corpus cache: {e}")

        return text_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return dictionary of tensors for the specific index
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "special_tokens_mask": self.encodings["special_tokens_mask"][idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        return item


class PearsonDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    Handles input formatting: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, df, tokenizer, max_length=Config.max_length, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'context_text', 'anchor', 'target', and optionally 'score'.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        self.inputs = None
        self.targets = None
        self.cls_labels = None

        self._process_data()

    def _process_data(self):
        # Extract text columns
        contexts = self.df["context_text"].astype(str).fillna("").tolist()
        anchors = self.df["anchor"].astype(str).fillna("").tolist()
        targets = self.df["target"].astype(str).fillna("").tolist()

        # Prepare Targets for Train/Val
        if self.mode != "test":
            if "score" not in self.df.columns:
                raise ValueError("Column 'score' missing for train/val mode.")

            scores = self.df["score"].values

            # Regression targets
            self.targets = torch.tensor(scores, dtype=torch.float)

            # Classification labels (0.0 -> 0, 0.25 -> 1, ..., 1.0 -> 4)
            # Multiply by 4 and round to nearest int
            cls_indices = (np.round(scores * 4)).astype(int)
            self.cls_labels = torch.tensor(cls_indices, dtype=torch.long)

        # Prepare Inputs
        # We construct the input as:
        # Sequence A: Context
        # Sequence B: Anchor + [SEP] + Target
        # This allows the model to distinguish the context from the phrase pair using segment IDs (if supported)
        # or simply via the SEP token structure.

        sep = self.tokenizer.sep_token
        # Construct second sequence manually
        text_pair_list = [f"{a}{sep}{t}" for a, t in zip(anchors, targets)]

        self.inputs = self.tokenizer(
            contexts,  # First sequence
            text_pair_list,  # Second sequence
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
        }

        if "token_type_ids" in self.inputs:
            item["token_type_ids"] = self.inputs["token_type_ids"][idx]

        if self.mode != "test":
            item["target"] = self.targets[idx]
            item["label"] = self.cls_labels[idx]

        return item


def load_processed_data(load_cached_data=True):
    """
    Utility function to load Train, Val, and Test dataframes.
    Applies CPCHelper to expand context codes into descriptions.
    Leverages caching to avoid re-processing.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cpc_helper = CPCHelper()

    # Load raw dataframes
    # We load them first because CPCHelper.process_dataset requires the DF argument
    # even if it hits the cache (it's the function signature).
    # However, reading the raw CSVs is fast.

    print("Loading and processing datasets...")

    # Train
    train_raw = pd.read_csv(Config.train_path)
    train_df = cpc_helper.process_dataset(
        train_raw, Config.train_cache_path, load_cached_data
    )

    # Val
    val_raw = pd.read_csv(Config.val_path)
    val_df = cpc_helper.process_dataset(
        val_raw, Config.val_cache_path, load_cached_data
    )

    # Test
    test_raw = pd.read_csv(Config.test_path)
    test_df = cpc_helper.process_dataset(
        test_raw, Config.test_cache_path, load_cached_data
    )

    return train_df, val_df, test_df
