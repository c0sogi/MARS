import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import CFG
from library.utils import seed_everything


def get_levenshtein_distance(s1, s2):
    """
    Computes Levenshtein distance between two strings using DP.
    """
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_cpc_texts():
    """
    Returns a mapping of CPC Section codes to their descriptions.
    """
    return {
        "A": "Human Necessities",
        "B": "Performing Operations; Transporting",
        "C": "Chemistry; Metallurgy",
        "D": "Textiles; Paper",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering; Lighting; Heating; Weapons; Blasting",
        "G": "Physics",
        "H": "Electricity",
        "Y": "General Tagging of New Technological Developments",
    }


def get_structural_features(anchor, target):
    """
    Computes structural features:
    1. Normalized Levenshtein Distance
    2. Jaccard Similarity
    3. Length Ratio
    """
    anchor = str(anchor).lower().strip()
    target = str(target).lower().strip()

    # 1. Normalized Levenshtein
    dist = get_levenshtein_distance(anchor, target)
    max_len = max(len(anchor), len(target))
    norm_lev = dist / max_len if max_len > 0 else 0.0

    # 2. Jaccard Similarity (Word-level)
    a_tokens = set(anchor.split())
    b_tokens = set(target.split())
    intersection = len(a_tokens.intersection(b_tokens))
    union = len(a_tokens.union(b_tokens))
    jaccard = intersection / union if union > 0 else 0.0

    # 3. Length Ratio
    len_a = len(anchor)
    len_b = len(target)
    len_ratio = len_a / (len_b + 1e-8)

    return [norm_lev, jaccard, len_ratio]


def process_data(df, cpc_map):
    """
    Enriches dataframe with context text and structural features.
    """
    # Context Enrichment
    df["context_text"] = (
        df["context"].astype(str).apply(lambda x: cpc_map.get(x[0], ""))
    )

    # Structural Features
    features = []
    for _, row in df.iterrows():
        feats = get_structural_features(row["anchor"], row["target"])
        features.append(feats)

    features = np.array(features)
    df["feat_lev"] = features[:, 0]
    df["feat_jac"] = features[:, 1]
    df["feat_len"] = features[:, 2]

    return df


def load_and_preprocess(path, cache_path, load_cached_data=True):
    """
    Loads data from path, processes it, and caches it to cache_path.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = pd.read_csv(path)

        # Debug mode: subset data
        if CFG.debug:
            df = df.sample(n=min(100, len(df)), random_state=CFG.seed).reset_index(
                drop=True
            )

        cpc_map = get_cpc_texts()
        df = process_data(df, cpc_map)

        df.to_parquet(cache_path, index=False)

    return df


class PhraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context_text"].values
        self.feats = df[["feat_lev", "feat_jac", "feat_len"]].values.astype(np.float32)

        if self.mode != "test":
            self.labels = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = str(self.anchors[idx])
        target = str(self.targets[idx])
        context = str(self.contexts[idx])

        # Input Construction: [CLS] anchor [SEP] target context [SEP]
        # This allows the model to attend to target and context together relative to anchor
        text_first = anchor
        text_second = target + " " + context

        inputs = self.tokenizer(
            text_first,
            text_second,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        structural_features = torch.tensor(self.feats[idx], dtype=torch.float)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_features": structural_features,
        }

        if self.mode != "test":
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_data_loaders(tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    """
    train_df = load_and_preprocess(
        CFG.train_path, CFG.train_processed_path, load_cached_data
    )
    val_df = load_and_preprocess(CFG.val_path, CFG.val_processed_path, load_cached_data)

    train_dataset = PhraseDataset(train_df, tokenizer, CFG.max_len, mode="train")
    val_dataset = PhraseDataset(val_df, tokenizer, CFG.max_len, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer, load_cached_data=True):
    """
    Creates DataLoader for testing. Returns loader and the dataframe (for IDs).
    """
    test_df = load_and_preprocess(
        CFG.test_path, CFG.test_processed_path, load_cached_data
    )
    test_dataset = PhraseDataset(test_df, tokenizer, CFG.max_len, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df
