import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from nltk.metrics import edit_distance
from library.config import Config
from library.utils import get_cpc_texts

# =========================================================================
# Feature Engineering Helpers
# =========================================================================


def normalized_levenshtein(str1, str2):
    """Computes Levenshtein distance normalized by the max length."""
    len1, len2 = len(str1), len(str2)
    if len1 == 0 and len2 == 0:
        return 0.0
    dist = edit_distance(str1, str2)
    return dist / max(len1, len2)


def jaccard_similarity(str1, str2):
    """Computes Jaccard similarity between sets of words."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def length_ratio(str1, str2):
    """Computes ratio of lengths (min/max)."""
    len1, len2 = len(str1), len(str2)
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0
    return min(len1, len2) / max(len1, len2)


def compute_structural_features(df):
    """
    Computes structural features for the dataframe.
    Returns a numpy array of shape (N, 3).
    """
    # Ensure strings
    anchors = df["anchor"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()

    feats = []
    for a, t in zip(anchors, targets):
        lev = normalized_levenshtein(a, t)
        jac = jaccard_similarity(a, t)
        lr = length_ratio(a, t)
        feats.append([lev, jac, lr])

    return np.array(feats, dtype=np.float32)


def get_soft_targets(score, num_classes=5, sigma=1.0):
    """
    Generates Gaussian Soft Targets for a given scalar score.
    Classes correspond to [0.0, 0.25, 0.5, 0.75, 1.0].
    """
    class_values = np.linspace(0, 1, num_classes)
    # Gaussian distribution unnormalized
    logits = -0.5 * ((class_values - score) ** 2) / (sigma**2)
    probs = np.exp(logits)
    # Normalize
    probs = probs / probs.sum()
    return probs


# =========================================================================
# Data Processing with Caching
# =========================================================================


def preprocess_data(split, load_cached_data=True):
    """
    Loads raw data, merges context, computes features, and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    cache_file = os.path.join(Config.working_dir, f"{split}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # Check if structural feature columns exist, if not, recompute (safety check)
            expected_cols = Config.structural_features
            if all(col in df.columns for col in expected_cols):
                print(f"Loaded {split} data from cache: {cache_file}")
                return df
        except Exception as e:
            print(f"Failed to load cache {cache_file}: {e}. Recomputing...")

    # 2. Load Raw Data
    if split == "train":
        path = Config.train_path
    elif split == "val":
        path = Config.val_path
    elif split == "test":
        path = Config.test_path
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_csv(path)

    # Debug mode: sample small subset
    if Config.debug:
        df = df.head(100).reset_index(drop=True)
        print(f"DEBUG MODE: Sampled {len(df)} rows for {split}.")

    # 3. Context Enrichment
    cpc_texts = get_cpc_texts(load_cached_data=load_cached_data)
    # Merge on 'context' column
    df = df.merge(cpc_texts, on="context", how="left")
    # Fill missing context texts if any
    df["context_text"] = df["context_text"].fillna("")

    # 4. Compute Structural Features
    print(f"Computing structural features for {split}...")
    struct_feats = compute_structural_features(df)

    # Add features to DataFrame
    df["normalized_levenshtein"] = struct_feats[:, 0]
    df["jaccard_similarity"] = struct_feats[:, 1]
    df["length_ratio"] = struct_feats[:, 2]

    # 5. Save to Cache
    try:
        df.to_parquet(cache_file, index=False)
        print(f"Saved processed {split} data to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file}: {e}")

    return df


# =========================================================================
# Dataset Class
# =========================================================================


class PearsonDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=False):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.ids = df["id"].values
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context_text"].values

        # Structural features
        self.structural_features = df[Config.structural_features].values.astype(
            np.float32
        )

        # Targets
        if "score" in df.columns:
            self.scores = df["score"].values.astype(np.float32)
        else:
            self.scores = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = str(self.anchors[idx])
        target = str(self.targets[idx])
        context_text = str(self.contexts[idx])

        # Construct Input: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # DeBERTa tokenizer handles text_pair by adding a SEP.
        # We manually construct the first part to include context and anchor separated by SEP.
        # Note: DeBERTa V3 uses SentencePiece.
        # Standard usage: tokenizer(text, text_pair) -> [CLS] text [SEP] text_pair [SEP]
        # We want: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # We can achieve this by: text = context + [SEP] + anchor, text_pair = target

        sep = self.tokenizer.sep_token
        first_segment = f"{context_text} {sep} {anchor}"
        second_segment = target

        inputs = self.tokenizer(
            first_segment,
            second_segment,
            add_special_tokens=True,
            max_length=Config.max_length,
            padding=False,  # Dynamic padding in collate_fn
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get(
                "token_type_ids", [0] * len(inputs["input_ids"])
            ),
            "structural_features": self.structural_features[idx],
            "id": self.ids[idx],
        }

        if self.scores is not None:
            score = self.scores[idx]
            # Soft Label Generation
            soft_targets = get_soft_targets(
                score, num_classes=Config.num_classes, sigma=Config.label_sigma
            )
            item["labels"] = torch.tensor(soft_targets, dtype=torch.float32)
            item["score"] = torch.tensor(
                score, dtype=torch.float32
            )  # Keep raw score for metric calc

        return item


# =========================================================================
# Collate Function
# =========================================================================


def collate_fn(batch):
    """
    Handles dynamic padding for variable length sequences.
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_masks = []
    token_type_ids = []
    structural_features = []
    labels = []
    scores = []
    ids = []

    for x in batch:
        # Pad input_ids
        ids_len = len(x["input_ids"])
        pad_len = max_len - ids_len

        # 0 is usually the pad token id for DeBERTa/RoBERTa/BERT
        # However, it's safer to check tokenizer.pad_token_id in main, but here we assume 0 or standard.
        # Ideally we should pass tokenizer to collate_fn, but standard transformers use 0 or 1.
        # DeBERTa V3 pad_token_id is 0.
        pad_token_id = 0

        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_masks.append(x["attention_mask"] + [0] * pad_len)
        token_type_ids.append(x["token_type_ids"] + [0] * pad_len)

        structural_features.append(x["structural_features"])
        ids.append(x["id"])

        if "labels" in x:
            labels.append(x["labels"])
            scores.append(x["score"])

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "structural_features": torch.tensor(
            np.array(structural_features), dtype=torch.float32
        ),
        "ids": ids,
    }

    if len(labels) > 0:
        out["labels"] = torch.stack(labels)
        out["scores"] = torch.stack(scores)

    return out


# =========================================================================
# Main DataLoader Getter
# =========================================================================


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Preprocess Data
    train_df = preprocess_data("train", load_cached_data)
    val_df = preprocess_data("val", load_cached_data)
    test_df = preprocess_data("test", load_cached_data)

    # Create Datasets
    train_dataset = PearsonDataset(train_df, tokenizer, is_train=True)
    val_dataset = PearsonDataset(val_df, tokenizer, is_train=False)
    test_dataset = PearsonDataset(test_df, tokenizer, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
