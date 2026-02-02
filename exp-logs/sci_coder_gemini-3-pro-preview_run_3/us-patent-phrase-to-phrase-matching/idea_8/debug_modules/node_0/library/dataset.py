import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from library.config import Config
from library.cpc_texts import create_context_map


class PhraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=140, inference_only=False):
        """
        PyTorch Dataset for Phrase Matching.

        Args:
            df (pd.DataFrame): Dataframe containing 'anchor', 'target', 'context_text', and optionally 'score'.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            inference_only (bool): If True, does not look for 'score' column.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference_only = inference_only

        # Extract columns to numpy arrays for faster access
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context_text"].values

        if not self.inference_only:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct the input text.
        # We want the structure: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # By passing `context` as the first argument and `anchor + SEP + target` as the second,
        # the tokenizer (e.g., DeBERTa) handles the first separation.
        # We manually insert the separator between anchor and target.
        second_part = str(anchor) + self.tokenizer.sep_token + str(target)

        inputs = self.tokenizer(
            str(context),
            second_part,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.inference_only:
            score = self.scores[idx]
            # Regression target (float)
            sample["label"] = torch.tensor(score, dtype=torch.float)

            # Auxiliary Classification target
            # Scores are 0.0, 0.25, 0.5, 0.75, 1.0 -> Classes 0, 1, 2, 3, 4
            # We round to nearest integer after scaling to handle float precision
            class_label = int(round(score * 4))
            sample["label_class"] = torch.tensor(class_label, dtype=torch.long)

        return sample


def load_dataset(mode="train", load_cached_data=True):
    """
    Loads the dataset, merges it with CPC context descriptions, and handles caching.

    Args:
        mode (str): One of 'train', 'val', 'test', or 'all_train'.
                  'all_train' combines the metadata train and val sets.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The processed dataframe with 'context_text'.
    """
    # Determine paths based on mode
    if mode == "train":
        input_path = Config.train_file
        cache_path = Config.train_cache_path
    elif mode == "val":
        input_path = Config.val_file
        cache_path = Config.val_cache_path
    elif mode == "test":
        input_path = Config.test_file
        cache_path = Config.test_cache_path
    elif mode == "all_train":
        input_path = None  # Composed of train + val
        cache_path = os.path.join(Config.working_dir, "all_train_cache.parquet")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Cache load failed for {mode}: {e}. Regenerating...")

    # 2. Load raw data and process
    # Load context map (handles its own caching)
    context_map = create_context_map(load_cached_data=load_cached_data)

    if mode == "all_train":
        # Combine train and val metadata files
        df_train = pd.read_csv(Config.train_file)
        df_val = pd.read_csv(Config.val_file)
        df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    else:
        df = pd.read_csv(input_path)

    # Merge context text
    # We use a left join to ensure we keep all rows from the dataset
    df = df.merge(context_map, on="context", how="left")

    # Handle any potential missing contexts (though unlikely with valid data)
    df["context_text"] = df["context_text"].fillna("")

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def create_folds(df, n_folds=5, seed=42):
    """
    Performs Stratified K-Fold splitting on the dataframe.

    Args:
        df (pd.DataFrame): The training dataframe.
        n_folds (int): Number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The dataframe with a new 'fold' column.
    """
    df["fold"] = -1

    # Create integer labels for stratification
    # 0.0 -> 0, 0.25 -> 1, ..., 1.0 -> 4
    y = (df["score"] * 4).round().astype(int).values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    return df
