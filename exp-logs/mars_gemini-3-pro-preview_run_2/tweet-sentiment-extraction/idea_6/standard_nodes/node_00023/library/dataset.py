import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class TweetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_tokens,
        end_tokens,
        offsets,
        texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "targets_start": torch.tensor(self.start_tokens[item], dtype=torch.long),
            "targets_end": torch.tensor(self.end_tokens[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "orig_tweet": str(self.texts[item]),
            "sentiment": str(self.sentiments[item]),
        }
        if self.selected_texts is not None:
            data["orig_selected"] = str(self.selected_texts[item])
        return data


def process_data(df, tokenizer, max_len, is_test=False):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_tokens_list = []
    end_tokens_list = []
    offsets_list = []

    texts = df["text"].values
    sentiments = df["sentiment"].values
    selected_texts = df["selected_text"].values if not is_test else None

    n_samples = len(df)

    for i in range(n_samples):
        text = str(texts[i])
        sentiment = str(sentiments[i])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # DeBERTa tokenizer handles this via text and text_pair
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids", [0] * max_len)
        offsets = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        start_idx = 0
        end_idx = 0

        if not is_test:
            selected_text = str(selected_texts[i])
            # Find start and end char indices of selected_text in text
            char_start = text.find(selected_text)
            char_end = char_start + len(selected_text)

            # Fallback if not found (should be rare with clean data)
            if char_start == -1:
                char_start = 0
                char_end = len(text)

            # Map char indices to token indices
            # We only care about tokens belonging to the second sequence (the text)
            valid_token_indices = []
            for idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
                if seq_id == 1:
                    # Check if token overlaps with the selected char span
                    # Token span: [offset[0], offset[1])
                    # Target span: [char_start, char_end)
                    if max(offset[0], char_start) < min(offset[1], char_end):
                        valid_token_indices.append(idx)

            if len(valid_token_indices) > 0:
                start_idx = valid_token_indices[0]
                end_idx = valid_token_indices[-1]
            else:
                # If no tokens overlap (e.g. stripped special chars), point to 0
                start_idx = 0
                end_idx = 0

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        start_tokens_list.append(start_idx)
        end_tokens_list.append(end_idx)
        offsets_list.append(offsets)

    return (
        np.array(input_ids_list),
        np.array(attention_mask_list),
        np.array(token_type_ids_list),
        np.array(start_tokens_list),
        np.array(end_tokens_list),
        np.array(offsets_list),
    )


def get_data(load_cached_data=True):
    seed_everything(Config.seed)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # Debug Sampling
    if Config.debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # Cache Paths
    # Cite debug_lesson_7: Bind Data Cache Filenames to Configuration Parameters
    suffix = f"_{Config.max_len}_debug" if Config.debug else f"_{Config.max_len}_full"
    cache_train = os.path.join(Config.CACHE_DIR, f"train_arrays{suffix}.npz")
    cache_val = os.path.join(Config.CACHE_DIR, f"val_arrays{suffix}.npz")
    cache_test = os.path.join(Config.CACHE_DIR, f"test_arrays{suffix}.npz")

    cache_exists = (
        os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        train_arrays = np.load(cache_train)
        val_arrays = np.load(cache_val)
        test_arrays = np.load(cache_test)

        # Unpack arrays
        train_inputs = (
            train_arrays["input_ids"],
            train_arrays["attention_mask"],
            train_arrays["token_type_ids"],
            train_arrays["start_tokens"],
            train_arrays["end_tokens"],
            train_arrays["offsets"],
        )
        val_inputs = (
            val_arrays["input_ids"],
            val_arrays["attention_mask"],
            val_arrays["token_type_ids"],
            val_arrays["start_tokens"],
            val_arrays["end_tokens"],
            val_arrays["offsets"],
        )
        test_inputs = (
            test_arrays["input_ids"],
            test_arrays["attention_mask"],
            test_arrays["token_type_ids"],
            test_arrays["start_tokens"],
            test_arrays["end_tokens"],
            test_arrays["offsets"],
        )
    else:
        print(f"Processing data from scratch (Model: {Config.model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

        train_inputs = process_data(df_train, tokenizer, Config.max_len, is_test=False)
        val_inputs = process_data(df_val, tokenizer, Config.max_len, is_test=False)
        test_inputs = process_data(df_test, tokenizer, Config.max_len, is_test=True)

        # Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.savez(
            cache_train,
            input_ids=train_inputs[0],
            attention_mask=train_inputs[1],
            token_type_ids=train_inputs[2],
            start_tokens=train_inputs[3],
            end_tokens=train_inputs[4],
            offsets=train_inputs[5],
        )
        np.savez(
            cache_val,
            input_ids=val_inputs[0],
            attention_mask=val_inputs[1],
            token_type_ids=val_inputs[2],
            start_tokens=val_inputs[3],
            end_tokens=val_inputs[4],
            offsets=val_inputs[5],
        )
        np.savez(
            cache_test,
            input_ids=test_inputs[0],
            attention_mask=test_inputs[1],
            token_type_ids=test_inputs[2],
            start_tokens=test_inputs[3],
            end_tokens=test_inputs[4],
            offsets=test_inputs[5],
        )

    # Construct Datasets
    # We combine the numeric arrays (from cache or processing) with the string columns from the dataframe
    train_dataset = TweetDataset(
        *train_inputs,
        texts=df_train["text"].values,
        sentiments=df_train["sentiment"].values,
        selected_texts=df_train["selected_text"].values,
    )

    val_dataset = TweetDataset(
        *val_inputs,
        texts=df_val["text"].values,
        sentiments=df_val["sentiment"].values,
        selected_texts=df_val["selected_text"].values,
    )

    test_dataset = TweetDataset(
        *test_inputs,
        texts=df_test["text"].values,
        sentiments=df_test["sentiment"].values,
        selected_texts=None,
    )

    return train_dataset, val_dataset, test_dataset
