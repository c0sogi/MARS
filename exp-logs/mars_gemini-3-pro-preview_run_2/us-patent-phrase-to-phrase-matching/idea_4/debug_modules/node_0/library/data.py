import os
import pandas as pd
import numpy as np
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_cpc_texts

# Ensure nltk resources are available (though edit_distance is usually logic-only)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    # Quietly attempt download if needed, though usually not required for simple edit distance
    pass


def calculate_manual_features(anchor, target):
    """
    Computes manual structural features for a pair of phrases.
    """
    # Ensure strings
    a = str(anchor).lower().strip()
    t = str(target).lower().strip()

    # 1. Jaccard Similarity
    a_words = set(a.split())
    t_words = set(t.split())
    intersection = len(a_words & t_words)
    union = len(a_words | t_words)
    jaccard = intersection / union if union > 0 else 0.0

    # 2. Normalized Levenshtein Distance
    # nltk.edit_distance is standard Levenshtein
    lev_dist = nltk.edit_distance(a, t)
    max_len = max(len(a), len(t))
    norm_lev = lev_dist / max_len if max_len > 0 else 0.0

    # 3. Normalized Length Difference
    len_diff = abs(len(a) - len(t))
    norm_len_diff = len_diff / max_len if max_len > 0 else 0.0

    return jaccard, norm_lev, norm_len_diff


def process_data(df, cpc_texts, cache_path, load_cached_data=True):
    """
    Processes the dataframe: adds context text and manual features.
    Handles caching via parquet.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_processed = pd.read_parquet(cache_path)
            # Verify columns exist
            required = ["context_text", "feat_jaccard", "feat_lev", "feat_len_diff"]
            if all(col in df_processed.columns for col in required):
                return df_processed
        except Exception:
            pass  # Fallback to processing

    # 2. Process Data
    # Map Context
    df["context_text"] = df["context"].map(cpc_texts).fillna("")

    # Compute Features
    # We use a list comprehension for speed over apply
    features = [
        calculate_manual_features(row.anchor, row.target) for row in df.itertuples()
    ]

    df["feat_jaccard"] = [x[0] for x in features]
    df["feat_lev"] = [x[1] for x in features]
    df["feat_len_diff"] = [x[2] for x in features]

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class PhraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-convert columns to lists for faster access
        self.anchors = df["anchor"].astype(str).tolist()
        self.targets = df["target"].astype(str).tolist()
        self.contexts = df["context_text"].astype(str).tolist()
        self.ids = df["id"].tolist()

        # Features: [Jaccard, Levenshtein, LengthDiff]
        self.feats = df[["feat_jaccard", "feat_lev", "feat_len_diff"]].values.astype(
            np.float32
        )

        # Labels
        if not is_test:
            self.scores = df["score"].values
            # Map scores to class indices: 0.0->0, 0.25->1, 0.5->2, 0.75->3, 1.0->4
            self.score_map = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
            self.labels = [self.score_map.get(s, 0) for s in self.scores]
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Manual Tokenization to ensure format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Encode parts without special tokens
        ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
        anc_ids = self.tokenizer.encode(anchor, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(target, add_special_tokens=False)

        # Construct input_ids
        input_ids = (
            [cls_id] + ctx_ids + [sep_id] + anc_ids + [sep_id] + tgt_ids + [sep_id]
        )

        # Truncate if necessary
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

        # Convert to tensors
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "features": torch.tensor(self.feats[idx], dtype=torch.float),
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            # Also return float score for metric calculation if needed
            item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


def get_loaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # 2. Get CPC Texts mapping
    cpc_texts = get_cpc_texts()

    # 3. Process Data (Enrichment + Feature Engineering)
    # Define cache paths
    train_cache = os.path.join(Config.working_dir, "train_processed.parquet")
    val_cache = os.path.join(Config.working_dir, "val_processed.parquet")
    test_cache = os.path.join(Config.working_dir, "test_processed.parquet")

    df_train = process_data(df_train, cpc_texts, train_cache, load_cached_data)
    df_val = process_data(df_val, cpc_texts, val_cache, load_cached_data)
    df_test = process_data(df_test, cpc_texts, test_cache, load_cached_data)

    # 4. Debug Mode Subsampling
    if Config.debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        # Keep test set intact or sample? Usually keep test small in debug to test pipeline
        df_test = df_test.sample(
            n=min(len(df_test), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        print(
            f"Debug Mode: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
        )

    # 5. Create Datasets
    train_dataset = PhraseDataset(df_train, tokenizer, Config.max_length, is_test=False)
    val_dataset = PhraseDataset(df_val, tokenizer, Config.max_length, is_test=False)
    test_dataset = PhraseDataset(df_test, tokenizer, Config.max_length, is_test=True)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    return train_loader, val_loader, test_loader
