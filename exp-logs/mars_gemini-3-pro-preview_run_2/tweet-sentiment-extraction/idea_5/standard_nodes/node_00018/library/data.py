import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    Dataset for Training and Validation.
    Returns inputs and targets for the model.
    """

    def __init__(self, data, indices=None):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.offsets = data["offsets"]
        self.start_targets = data["start_targets"]
        self.end_targets = data["end_targets"]
        self.orig_texts = data["orig_texts"]
        self.sentiments = data["sentiments"]
        # If indices are provided (e.g., for a specific fold), use them. Otherwise use all.
        self.indices = (
            indices if indices is not None else np.arange(len(self.input_ids))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return {
            "input_ids": torch.tensor(self.input_ids[actual_idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_mask[actual_idx], dtype=torch.long
            ),
            "start_targets": torch.tensor(
                self.start_targets[actual_idx], dtype=torch.long
            ),
            "end_targets": torch.tensor(self.end_targets[actual_idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[actual_idx], dtype=torch.long),
            "orig_text": str(self.orig_texts[actual_idx]),
            "sentiment": str(self.sentiments[actual_idx]),
        }


class TweetTestDataset(Dataset):
    """
    Dataset for Inference.
    Does not return targets.
    """

    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.offsets = data["offsets"]
        self.orig_texts = data["orig_texts"]
        self.sentiments = data["sentiments"]
        self.text_ids = data["text_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "orig_text": str(self.orig_texts[idx]),
            "sentiment": str(self.sentiments[idx]),
            "text_id": str(self.text_ids[idx]),
        }


def _preprocess(df, tokenizer, max_len, is_test=False):
    """
    Tokenizes data and computes targets.
    """
    n = len(df)
    input_ids = np.zeros((n, max_len), dtype=np.int32)
    attention_mask = np.zeros((n, max_len), dtype=np.int32)
    offsets = np.zeros((n, max_len, 2), dtype=np.int32)
    start_targets = np.zeros(n, dtype=np.int32)
    end_targets = np.zeros(n, dtype=np.int32)

    orig_texts = []
    sentiments = []
    text_ids = []

    for idx, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # We use the pair encoding capability of the tokenizer
        # This automatically handles the special tokens and segment IDs
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        off = encoded["offset_mapping"]
        seq_ids = encoded.sequence_ids()

        input_ids[idx] = ids
        attention_mask[idx] = mask
        offsets[idx] = off

        orig_texts.append(text)
        sentiments.append(sentiment)

        if is_test:
            text_ids.append(row["textID"])
        else:
            selected_text = str(row["selected_text"])

            # --- Target Extraction ---
            # 1. Find character indices of selected_text within text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: if not found, select entire text
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # 2. Map character indices to token indices
            # Filter tokens that belong to the second sequence (the tweet text)
            # seq_ids is None for special tokens, 0 for sentiment, 1 for text
            tokens_in_text = [i for i, s_id in enumerate(seq_ids) if s_id == 1]

            if not tokens_in_text:
                # Edge case: text truncated completely or empty
                start_targets[idx] = 0
                end_targets[idx] = 0
                continue

            s_token = tokens_in_text[0]
            e_token = tokens_in_text[-1]

            found_start = False
            found_end = False

            for i in tokens_in_text:
                o_start, o_end = off[i]

                # Skip if offset is invalid (e.g. padding)
                if o_start == o_end and o_start == 0:
                    continue

                # Determine Start Token:
                # The token should contain the start_char
                if o_start <= start_char < o_end:
                    s_token = i
                    found_start = True

                # Determine End Token:
                # The token should contain the last character of the selection (end_char - 1)
                if o_start <= (end_char - 1) < o_end:
                    e_token = i
                    found_end = True

            # Fallback logic if exact containment not found (e.g. due to weird tokenization)
            if not found_start:
                # Find first token that starts after start_char
                for i in tokens_in_text:
                    if off[i][1] > start_char:
                        s_token = i
                        break

            if not found_end:
                # Find last token that starts before end_char
                for i in reversed(tokens_in_text):
                    if off[i][0] < end_char:
                        e_token = i
                        break

            start_targets[idx] = s_token
            end_targets[idx] = e_token

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offsets": offsets,
        "start_targets": start_targets,
        "end_targets": end_targets,
        "orig_texts": np.array(orig_texts),
        "sentiments": np.array(sentiments),
        "text_ids": np.array(text_ids) if is_test else None,
    }


def get_data(load_cached_data=True):
    """
    Main function to load and process data.
    Uses caching to avoid re-processing.
    """
    # Define cache paths
    cache_path_train = os.path.join(
        Config.CACHE_DIR, f"cached_train_{Config.MAX_LEN}.npz"
    )
    cache_path_test = os.path.join(
        Config.CACHE_DIR, f"cached_test_{Config.MAX_LEN}.npz"
    )

    train_data = None
    test_data = None

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_train)
        and os.path.exists(cache_path_test)
    ):
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        try:
            train_data = np.load(cache_path_train, allow_pickle=True)
            # Convert npz file object to dict
            train_data = {k: train_data[k] for k in train_data.files}

            test_data = np.load(cache_path_test, allow_pickle=True)
            test_data = {k: test_data[k] for k in test_data.files}

            return train_data, test_data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print("Preprocessing data from scratch...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Metadata CSVs
    # We combine train and val metadata to perform our own 5-Fold CV
    df_train = pd.read_csv(Config.TRAIN_FILE)
    df_val = pd.read_csv(Config.VAL_FILE)
    df_test = pd.read_csv(Config.TEST_FILE)

    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)

    # Basic Cleaning
    df_full_train.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)
    df_test.fillna({"text": ""}, inplace=True)

    # Debugging option
    if Config.DEBUG:
        df_full_train = df_full_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Data
    train_data = _preprocess(df_full_train, tokenizer, Config.MAX_LEN, is_test=False)
    test_data = _preprocess(df_test, tokenizer, Config.MAX_LEN, is_test=True)

    # 3. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    folds = np.zeros(len(df_full_train), dtype=np.int32)

    # Stratify by sentiment
    for fold, (_, val_idx) in enumerate(
        skf.split(df_full_train, df_full_train["sentiment"])
    ):
        folds[val_idx] = fold

    train_data["folds"] = folds

    # 4. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path_train, **train_data)
    np.savez(cache_path_test, **test_data)
    print("Data processed and cached.")

    return train_data, test_data
