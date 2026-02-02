import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        offsets,
        original_texts,
        sentiments,
        start_tokens=None,
        end_tokens=None,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.offsets = offsets
        self.original_texts = original_texts
        self.sentiments = sentiments
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "text": str(self.original_texts[idx]),
            "sentiment": str(self.sentiments[idx]),
            "textID": idx,  # Using index as ID for tracking if needed
        }

        if self.start_tokens is not None and self.end_tokens is not None:
            data["start_tokens"] = torch.tensor(
                self.start_tokens[idx], dtype=torch.long
            )
            data["end_tokens"] = torch.tensor(self.end_tokens[idx], dtype=torch.long)

        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[idx])

        return data


def process_data(
    df, tokenizer, max_len, cache_path, load_cached_data=True, is_test=False
):
    """
    Tokenizes data, computes targets, and manages caching.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary
            result = {f: data[f] for f in data.files}
            # Ensure consistency with dataframe length (simple check)
            if len(result["input_ids"]) == len(df):
                return result
            else:
                print("Cached data length mismatch. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {len(df)} samples...")

    # Pre-allocate arrays
    n_samples = len(df)
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    token_type_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    offsets = np.zeros((n_samples, max_len, 2), dtype=np.int32)
    start_tokens = np.zeros(n_samples, dtype=np.int32)
    end_tokens = np.zeros(n_samples, dtype=np.int32)

    # Extract columns
    texts = df["text"].astype(str).values
    sentiments = df["sentiment"].astype(str).values
    selected_texts = df["selected_text"].astype(str).values if not is_test else None

    for i in range(n_samples):
        text = " " + " ".join(texts[i].split())
        sentiment = sentiments[i]

        # Encode
        # Input format: [CLS] sentiment [SEP] text [SEP]
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids[i] = encoded["input_ids"]
        attention_mask[i] = encoded["attention_mask"]
        token_type_ids[i] = encoded["token_type_ids"]

        # Adjust offsets
        # We only care about offsets for the text part.
        # DeBERTa tokenizer returns offsets relative to the specific string in the pair.
        # We need to identify which tokens belong to 'text'.
        seq_ids = encoded.sequence_ids()
        raw_offsets = encoded["offset_mapping"]

        # Store offsets (needed for inference/decoding)
        # We zero out offsets for special tokens and sentiment to avoid confusion
        cleaned_offsets = []
        for j, seq_id in enumerate(seq_ids):
            if seq_id == 1:  # 1 corresponds to the second sentence (text)
                cleaned_offsets.append(raw_offsets[j])
            else:
                cleaned_offsets.append((0, 0))

        # Pad offsets if necessary (though padding="max_length" usually handles list length,
        # offset_mapping might need manual truncation/padding if not handled by tokenizer exactly same way)
        if len(cleaned_offsets) < max_len:
            cleaned_offsets += [(0, 0)] * (max_len - len(cleaned_offsets))
        else:
            cleaned_offsets = cleaned_offsets[:max_len]

        offsets[i] = cleaned_offsets

        # Compute Targets
        if not is_test and selected_texts is not None:
            sel_text = " " + " ".join(selected_texts[i].split())

            # Find char indices in the normalized text
            start_char = text.find(sel_text)

            # If exact match not found (rare data noise), fallback to full text
            if start_char == -1:
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(sel_text)

            # Map char indices to token indices
            # We use the FastTokenizer's char_to_token method
            # sequence_index=1 points to the 'text' part of the input
            token_start_index = encoded.char_to_token(i=start_char, sequence_index=1)
            token_end_index = encoded.char_to_token(i=end_char - 1, sequence_index=1)

            # Fallback if char_to_token returns None (e.g., char is a space not mapped to a specific token)
            # We search for the nearest valid token
            if token_start_index is None:
                token_start_index = encoded.char_to_token(
                    i=start_char + 1, sequence_index=1
                )

            if token_end_index is None:
                token_end_index = encoded.char_to_token(
                    i=end_char - 2, sequence_index=1
                )

            # Final fallback: if still None, point to CLS or limits of text
            # Find bounds of text tokens
            text_token_indices = [idx for idx, sid in enumerate(seq_ids) if sid == 1]
            if not text_token_indices:
                # Should not happen
                start_tokens[i] = 0
                end_tokens[i] = 0
            else:
                if token_start_index is None:
                    token_start_index = text_token_indices[0]
                if token_end_index is None:
                    token_end_index = text_token_indices[-1]

                start_tokens[i] = token_start_index
                end_tokens[i] = token_end_index

    # Save to cache
    data_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "offsets": offsets,
        "start_tokens": start_tokens,
        "end_tokens": end_tokens,
    }

    np.savez(cache_path, **data_dict)
    return data_dict


def get_loaders(fold, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for a specific fold using StratifiedKFold.
    """
    seed_everything(Config.SEED)

    # Load Data
    df = pd.read_csv(Config.TRAIN_FILE)

    # Debugging
    if debug or Config.DEBUG:
        df = df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"DEBUG Mode: Sampled {len(df)} rows.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Cache Path
    debug_suffix = "_debug" if (debug or Config.DEBUG) else ""
    cache_path = os.path.join(
        Config.CACHE_DIR, f"cached_train_{Config.MAX_LEN}{debug_suffix}.npz"
    )

    # Process Data
    data = process_data(
        df,
        tokenizer,
        Config.MAX_LEN,
        cache_path,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    train_idx, val_idx = list(skf.split(df, df["sentiment"]))[fold]

    # Create Datasets
    train_dataset = TweetDataset(
        input_ids=data["input_ids"][train_idx],
        attention_mask=data["attention_mask"][train_idx],
        token_type_ids=data["token_type_ids"][train_idx],
        offsets=data["offsets"][train_idx],
        original_texts=df["text"].values[train_idx],
        sentiments=df["sentiment"].values[train_idx],
        start_tokens=data["start_tokens"][train_idx],
        end_tokens=data["end_tokens"][train_idx],
        selected_texts=df["selected_text"].values[train_idx],
    )

    val_dataset = TweetDataset(
        input_ids=data["input_ids"][val_idx],
        attention_mask=data["attention_mask"][val_idx],
        token_type_ids=data["token_type_ids"][val_idx],
        offsets=data["offsets"][val_idx],
        original_texts=df["text"].values[val_idx],
        sentiments=df["sentiment"].values[val_idx],
        start_tokens=data["start_tokens"][val_idx],
        end_tokens=data["end_tokens"][val_idx],
        selected_texts=df["selected_text"].values[val_idx],
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True, debug=False):
    """
    Creates DataLoader for the test set.
    """
    seed_everything(Config.SEED)

    df = pd.read_csv(Config.TEST_FILE)

    if debug or Config.DEBUG:
        df = df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    debug_suffix = "_debug" if (debug or Config.DEBUG) else ""
    cache_path = os.path.join(
        Config.CACHE_DIR, f"cached_test_{Config.MAX_LEN}{debug_suffix}.npz"
    )

    data = process_data(
        df,
        tokenizer,
        Config.MAX_LEN,
        cache_path,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    dataset = TweetDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        token_type_ids=data["token_type_ids"],
        offsets=data["offsets"],
        original_texts=df["text"].values,
        sentiments=df["sentiment"].values,
        # No targets for test
        start_tokens=None,
        end_tokens=None,
        selected_texts=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader
