import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Wraps pre-processed numpy arrays for efficient access.
    """

    def __init__(
        self, inputs, targets=None, texts=None, sentiments=None, is_test=False
    ):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.token_type_ids = inputs["token_type_ids"]
        self.offsets = inputs["offsets"]

        self.targets = targets
        self.texts = texts
        self.sentiments = sentiments
        self.is_test = is_test

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        out = {
            "ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "orig_text": self.texts[item],
            "sentiment": self.sentiments[item],
        }

        if not self.is_test and self.targets is not None:
            out["target_start"] = torch.tensor(
                self.targets["start_idx"][item], dtype=torch.long
            )
            out["target_end"] = torch.tensor(
                self.targets["end_idx"][item], dtype=torch.long
            )

        return out


def _find_target_indices(text, selected_text, offsets, sequence_ids):
    """
    Computes the start and end token indices for the selected_text within the text.
    """
    # Handle edge cases where selected_text is not perfectly found
    start_char = text.find(selected_text)
    if start_char == -1:
        # Try stripping
        start_char = text.find(selected_text.strip())

    # Fallback: if still not found, use the whole text
    if start_char == -1:
        end_char = len(text)
        start_char = 0
    else:
        end_char = start_char + len(selected_text)

    idx_start = 0
    idx_end = 0
    found_start = False
    found_end = False

    # Iterate through offsets to find the tokens corresponding to the char range
    # sequence_ids: 0 for sentiment, 1 for text (in DeBERTa tokenizer usually, or None/0/1)
    # We need to identify which part is the text.
    # DeBERTa v3: [CLS] sentiment [SEP] text [SEP]
    # sequence_ids usually returns None for special tokens, 0 for first seq, 1 for second.

    tokens_len = len(offsets)

    # Find the span of tokens that cover the char range
    for i in range(tokens_len):
        # Skip special tokens or sentiment part (sequence_id != 1 usually for pair encoding)
        # Note: We will verify sequence_id usage in the processing loop
        if sequence_ids[i] != 1:
            continue

        o_start, o_end = offsets[i]

        # If the token is within the selected text or overlaps significantly
        if o_start >= start_char and not found_start:
            idx_start = i
            found_start = True

        if o_end <= end_char:
            idx_end = i
            found_end = True

    # If exact match failed (e.g. token boundaries don't align perfectly),
    # we take the best approximation or the whole text part if completely missed
    if not found_start:
        # Find first token of the text part
        for i in range(tokens_len):
            if sequence_ids[i] == 1:
                idx_start = i
                break

    if not found_end:
        # Find last token of the text part
        for i in range(tokens_len - 1, -1, -1):
            if sequence_ids[i] == 1:
                idx_end = i
                break

    # Ensure validity
    if idx_end < idx_start:
        idx_end = idx_start

    return idx_start, idx_end


def process_data(config, load_cached_data=True):
    """
    Loads raw data, tokenizes, and caches the results.
    Returns dictionaries containing numpy arrays for inputs and targets.
    """
    cache_dir = config.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_data.npz")
    test_cache_path = os.path.join(cache_dir, "test_data.npz")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        # print("Loading cached data...")
        train_data = np.load(train_cache_path, allow_pickle=True)
        test_data = np.load(test_cache_path, allow_pickle=True)

        # Reconstruct dictionaries
        train_inputs = {
            k: train_data[k]
            for k in ["input_ids", "attention_mask", "token_type_ids", "offsets"]
        }
        train_targets = {k: train_data[k] for k in ["start_idx", "end_idx"]}
        train_meta = {"text": train_data["text"], "sentiment": train_data["sentiment"]}

        test_inputs = {
            k: test_data[k]
            for k in ["input_ids", "attention_mask", "token_type_ids", "offsets"]
        }
        test_meta = {"text": test_data["text"], "sentiment": test_data["sentiment"]}

        return (train_inputs, train_targets, train_meta), (test_inputs, test_meta)

    # 2. Process from Scratch
    # print("Processing data from scratch...")

    # Load Metadata CSVs
    df_train = pd.read_csv(config.train_path)
    df_val = pd.read_csv(config.val_path)
    df_train = pd.concat([df_train, df_val]).reset_index(
        drop=True
    )  # Combine for CV splitting later

    df_test = pd.read_csv(config.test_path)

    # Debug mode
    if config.debug:
        df_train = df_train.sample(
            n=config.debug_sample_size, random_state=config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=config.debug_sample_size, random_state=config.seed
        ).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Helper to process a dataframe
    def _tokenize_df(df, is_train=True):
        n = len(df)
        input_ids = np.zeros((n, config.max_len), dtype=np.int32)
        attention_mask = np.zeros((n, config.max_len), dtype=np.int8)
        token_type_ids = np.zeros((n, config.max_len), dtype=np.int8)
        offsets = np.zeros((n, config.max_len, 2), dtype=np.int32)
        start_idx = np.zeros(n, dtype=np.int32)
        end_idx = np.zeros(n, dtype=np.int32)

        texts = df["text"].astype(str).values
        sentiments = df["sentiment"].astype(str).values
        selected_texts = df["selected_text"].astype(str).values if is_train else None

        for i in tqdm(range(n), disable=True):  # Silent execution
            text = " " + " ".join(texts[i].split())
            sentiment = sentiments[i]

            # Tokenize: [CLS] sentiment [SEP] text [SEP]
            # DeBERTa tokenizer handles this structure when passed two args
            enc = tokenizer(
                sentiment,
                text,
                add_special_tokens=True,
                max_length=config.max_len,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

            input_ids[i] = enc["input_ids"]
            attention_mask[i] = enc["attention_mask"]
            token_type_ids[i] = enc["token_type_ids"]
            offsets[i] = enc["offset_mapping"]

            if is_train:
                sel_text = " " + " ".join(selected_texts[i].split())
                s_idx, e_idx = _find_target_indices(
                    text, sel_text, enc["offset_mapping"], enc.sequence_ids()
                )
                start_idx[i] = s_idx
                end_idx[i] = e_idx

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "offsets": offsets,
        }
        targets = None
        if is_train:
            targets = {"start_idx": start_idx, "end_idx": end_idx}

        return inputs, targets, texts, sentiments

    # Process Train
    train_inputs, train_targets, train_texts, train_sentiments = _tokenize_df(
        df_train, is_train=True
    )

    # Process Test
    test_inputs, _, test_texts, test_sentiments = _tokenize_df(df_test, is_train=False)

    # Save to Cache
    np.savez(
        train_cache_path,
        input_ids=train_inputs["input_ids"],
        attention_mask=train_inputs["attention_mask"],
        token_type_ids=train_inputs["token_type_ids"],
        offsets=train_inputs["offsets"],
        start_idx=train_targets["start_idx"],
        end_idx=train_targets["end_idx"],
        text=train_texts,
        sentiment=train_sentiments,
    )

    np.savez(
        test_cache_path,
        input_ids=test_inputs["input_ids"],
        attention_mask=test_inputs["attention_mask"],
        token_type_ids=test_inputs["token_type_ids"],
        offsets=test_inputs["offsets"],
        text=test_texts,
        sentiment=test_sentiments,
    )

    train_meta = {"text": train_texts, "sentiment": train_sentiments}
    test_meta = {"text": test_texts, "sentiment": test_sentiments}

    return (train_inputs, train_targets, train_meta), (test_inputs, test_meta)


def get_fold_dls(fold, config):
    """
    Returns train and validation DataLoaders for a specific fold.
    Uses StratifiedKFold to split the processed training data.
    """
    # Load all data
    (train_inputs, train_targets, train_meta), _ = process_data(
        config, load_cached_data=True
    )

    # Stratified Split
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    # We need to split based on sentiment to maintain distribution
    sentiments = train_meta["sentiment"]

    # Get indices for the requested fold
    # skf.split returns a generator, we iterate to find the specific fold
    for i, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(sentiments)), sentiments)
    ):
        if i == fold:
            break

    # Slice arrays for Train
    train_inp_fold = {k: v[train_idx] for k, v in train_inputs.items()}
    train_tgt_fold = {k: v[train_idx] for k, v in train_targets.items()}
    train_texts_fold = train_meta["text"][train_idx]
    train_sents_fold = train_meta["sentiment"][train_idx]

    # Slice arrays for Val
    val_inp_fold = {k: v[val_idx] for k, v in train_inputs.items()}
    val_tgt_fold = {k: v[val_idx] for k, v in train_targets.items()}
    val_texts_fold = train_meta["text"][val_idx]
    val_sents_fold = train_meta["sentiment"][val_idx]

    # Create Datasets
    train_ds = TweetDataset(
        inputs=train_inp_fold,
        targets=train_tgt_fold,
        texts=train_texts_fold,
        sentiments=train_sents_fold,
        is_test=False,
    )

    val_ds = TweetDataset(
        inputs=val_inp_fold,
        targets=val_tgt_fold,
        texts=val_texts_fold,
        sentiments=val_sents_fold,
        is_test=False,
    )

    # Create DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dl, val_dl


def get_test_dl(config):
    """
    Returns the DataLoader for the test set.
    """
    _, (test_inputs, test_meta) = process_data(config, load_cached_data=True)

    test_ds = TweetDataset(
        inputs=test_inputs,
        targets=None,
        texts=test_meta["text"],
        sentiments=test_meta["sentiment"],
        is_test=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return test_dl
