import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import seed_everything


class RegexTokenizer:
    """
    A simple tokenizer using regex to split text and mapping tokens to IDs.
    """

    def __init__(self, vocab_size=20000, max_len=128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pattern = re.compile(r"\b\w+\b")
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>"}

    def tokenize(self, text):
        if pd.isna(text):
            return []
        return self.pattern.findall(str(text).lower())

    def fit(self, texts):
        """
        Builds vocabulary from a list of texts.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Keep top N tokens
        most_common = counter.most_common(self.vocab_size - 2)  # Reserve 2 for PAD, UNK

        for token, _ in most_common:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text):
        """
        Converts text to a list of IDs with padding/truncation.
        Returns: (ids, original_length)
        """
        tokens = self.tokenize(text)
        original_len = len(tokens)

        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        # Truncate
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]

        # Pad
        if len(ids) < self.max_len:
            ids = ids + [self.vocab["<PAD>"]] * (self.max_len - len(ids))

        return ids, original_len

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.vocab, f)

    def load(self, filepath):
        with open(filepath, "r") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}


class ChatbotDataset(Dataset):
    """
    Dataset class for Chatbot Arena.
    """

    def __init__(self, prompt_ids, res_a_ids, res_b_ids, scalars, targets=None):
        self.prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        self.res_a_ids = torch.tensor(res_a_ids, dtype=torch.long)
        self.res_b_ids = torch.tensor(res_b_ids, dtype=torch.long)
        self.scalars = torch.tensor(scalars, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.prompt_ids)

    def __getitem__(self, idx):
        item = {
            "prompt_ids": self.prompt_ids[idx],
            "res_a_ids": self.res_a_ids[idx],
            "res_b_ids": self.res_b_ids[idx],
            "scalars": self.scalars[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        return item


def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding.
    Trims sequences to the maximum length in the batch (ignoring padding).
    """
    prompt_ids = torch.stack([item["prompt_ids"] for item in batch])
    res_a_ids = torch.stack([item["res_a_ids"] for item in batch])
    res_b_ids = torch.stack([item["res_b_ids"] for item in batch])
    scalars = torch.stack([item["scalars"] for item in batch])

    # Determine max length in this batch (find last non-pad index)
    # We check all sequences combined to find a common max length for simplicity in batching
    # or we can trim each type separately. Usually sharing max len across A/B is good for Siamese.
    # Here we check max valid index across all 3 text inputs.

    mask_prompt = prompt_ids != 0
    mask_a = res_a_ids != 0
    mask_b = res_b_ids != 0

    # Find the max length where at least one sequence has content
    # We default to at least 1 to avoid errors if everything is empty
    max_len_prompt = mask_prompt.sum(dim=1).max().item()
    max_len_res = max(mask_a.sum(dim=1).max().item(), mask_b.sum(dim=1).max().item())

    # Clamp to at least 1
    max_len_prompt = max(1, int(max_len_prompt))
    max_len_res = max(1, int(max_len_res))

    # Trim
    prompt_ids = prompt_ids[:, :max_len_prompt]
    res_a_ids = res_a_ids[:, :max_len_res]
    res_b_ids = res_b_ids[:, :max_len_res]

    out = {
        "prompt_ids": prompt_ids,
        "res_a_ids": res_a_ids,
        "res_b_ids": res_b_ids,
        "scalars": scalars,
    }

    if "target" in batch[0]:
        out["target"] = torch.stack([item["target"] for item in batch])

    return out


def process_and_save(df, tokenizer, output_path, is_test=False):
    """
    Helper to process a dataframe and save to npz.
    """
    # Initialize arrays
    n_samples = len(df)
    max_len = tokenizer.max_len

    prompt_arr = np.zeros((n_samples, max_len), dtype=np.int32)
    res_a_arr = np.zeros((n_samples, max_len), dtype=np.int32)
    res_b_arr = np.zeros((n_samples, max_len), dtype=np.int32)
    scalars_arr = np.zeros((n_samples, 3), dtype=np.float32)

    if not is_test:
        targets_arr = np.zeros((n_samples, 3), dtype=np.float32)
        target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
        targets_arr = df[target_cols].values.astype(np.float32)
    else:
        targets_arr = None

    print(f"Processing {n_samples} samples...")

    # Iterate and process
    prompts = df["prompt"].fillna("").astype(str).tolist()
    res_as = df["response_a"].fillna("").astype(str).tolist()
    res_bs = df["response_b"].fillna("").astype(str).tolist()

    for i in range(n_samples):
        # Encode
        p_ids, p_len = tokenizer.encode(prompts[i])
        a_ids, a_len = tokenizer.encode(res_as[i])
        b_ids, b_len = tokenizer.encode(res_bs[i])

        prompt_arr[i] = p_ids
        res_a_arr[i] = a_ids
        res_b_arr[i] = b_ids

        # Scalars: log(len + 1)
        scalars_arr[i] = [np.log1p(p_len), np.log1p(a_len), np.log1p(b_len)]

    # Save
    save_dict = {
        "prompt_ids": prompt_arr,
        "res_a_ids": res_a_arr,
        "res_b_ids": res_b_arr,
        "scalars": scalars_arr,
    }
    if not is_test:
        save_dict["targets"] = targets_arr

    np.savez(output_path, **save_dict)
    print(f"Saved processed data to {output_path}")


def prepare_data(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main function to prepare data.
    Handles caching, tokenization, and dataloader creation.
    """
    seed_everything()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # File paths
    vocab_path = os.path.join(Config.WORKING_DIR, "vocab.json")
    train_cache = os.path.join(Config.WORKING_DIR, "train_data.npz")
    val_cache = os.path.join(Config.WORKING_DIR, "val_data.npz")
    test_cache = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Check if we can load from cache
    cache_exists = (
        os.path.exists(vocab_path)
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    tokenizer = RegexTokenizer(vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN)

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        tokenizer.load(vocab_path)

        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)

    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        if debug:
            print(f"Debug mode: sampling {Config.DEBUG_SUBSET_SIZE} rows.")
            train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
            val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
            test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

        # Fit Tokenizer on Train
        print("Fitting tokenizer...")
        all_text = (
            pd.concat(
                [train_df["prompt"], train_df["response_a"], train_df["response_b"]]
            )
            .fillna("")
            .astype(str)
            .tolist()
        )
        tokenizer.fit(all_text)
        tokenizer.save(vocab_path)

        # Process and Save
        process_and_save(train_df, tokenizer, train_cache, is_test=False)
        process_and_save(val_df, tokenizer, val_cache, is_test=False)
        process_and_save(test_df, tokenizer, test_cache, is_test=True)

        # Reload to ensure consistency
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)

    # Create Datasets
    train_dataset = ChatbotDataset(
        train_data["prompt_ids"],
        train_data["res_a_ids"],
        train_data["res_b_ids"],
        train_data["scalars"],
        train_data["targets"],
    )
    val_dataset = ChatbotDataset(
        val_data["prompt_ids"],
        val_data["res_a_ids"],
        val_data["res_b_ids"],
        val_data["scalars"],
        val_data["targets"],
    )
    test_dataset = ChatbotDataset(
        test_data["prompt_ids"],
        test_data["res_a_ids"],
        test_data["res_b_ids"],
        test_data["scalars"],
        None,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
