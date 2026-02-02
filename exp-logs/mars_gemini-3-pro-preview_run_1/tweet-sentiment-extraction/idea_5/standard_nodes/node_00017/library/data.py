import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import jaccard


def get_gaussian_target(index, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at the given index.
    Used for Soft Labeling of start and end boundaries.
    """
    if index < 0 or index >= length:
        return np.zeros(length)

    x = np.arange(length)
    val = np.exp(-0.5 * ((x - index) / sigma) ** 2)

    # Normalize to form a valid probability distribution (sum = 1)
    if val.sum() > 0:
        val = val / val.sum()
    return val


class TweetDataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.offsets = data_dict["offsets"]
        self.raw_text = data_dict["raw_text"]
        self.sentiments = data_dict["sentiments"]
        self.is_test = is_test

        if not is_test:
            self.start_targets = data_dict["start_targets"]
            self.end_targets = data_dict["end_targets"]
            self.span_masks = data_dict["span_masks"]
            self.selected_text = data_dict["selected_text"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.raw_text[item]),
            "sentiment": str(self.sentiments[item]),
        }

        if not self.is_test:
            data["start_targets"] = torch.tensor(
                self.start_targets[item], dtype=torch.float
            )
            data["end_targets"] = torch.tensor(
                self.end_targets[item], dtype=torch.float
            )
            data["span_masks"] = torch.tensor(self.span_masks[item], dtype=torch.float)
            data["selected_text"] = str(self.selected_text[item])

        return data


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Tokenizes data and generates targets.
    """
    input_ids_list = []
    attention_mask_list = []
    offsets_list = []

    start_targets_list = []
    end_targets_list = []
    span_masks_list = []

    raw_text_list = df["text"].astype(str).values
    sentiments_list = df["sentiment"].astype(str).values
    selected_text_list = df["selected_text"].astype(str).values if not is_test else []

    def clean_text(t):
        return " ".join(str(t).split())

    for idx in range(len(df)):
        text = clean_text(raw_text_list[idx])
        sentiment = clean_text(sentiments_list[idx])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # We use tokenizer(sentiment, text) to handle this automatically
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        offsets_list.append(offset_mapping)

        if not is_test:
            selected_text = clean_text(selected_text_list[idx])

            # Find character start/end of selected_text in text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: use full text
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            token_start_index = 0
            token_end_index = 0
            found_start = False

            # Identify tokens belonging to the text part (sequence_id == 1)
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not text_token_indices:
                # Edge case: text truncated completely
                token_start_index = 0
                token_end_index = 0
            else:
                # Default to full span
                token_start_index = text_token_indices[0]
                token_end_index = text_token_indices[-1]

                for i in text_token_indices:
                    tok_start, tok_end = offset_mapping[i]

                    # Logic:
                    # Start token is the first token where the end of the token is inside the span (or after start)
                    # End token is the last token where the start of the token is inside the span (or before end)

                    if not found_start and tok_end > start_char:
                        token_start_index = i
                        found_start = True

                    if tok_start < end_char:
                        token_end_index = i

            # Enforce validity
            if token_start_index > token_end_index:
                token_end_index = token_start_index

            # Generate Gaussian Soft Targets
            s_target = get_gaussian_target(
                token_start_index, max_len, Config.gaussian_sigma
            )
            e_target = get_gaussian_target(
                token_end_index, max_len, Config.gaussian_sigma
            )

            # Generate Binary Span Mask (for auxiliary head)
            s_mask = np.zeros(max_len)
            s_mask[token_start_index : token_end_index + 1] = 1.0

            start_targets_list.append(s_target)
            end_targets_list.append(e_target)
            span_masks_list.append(s_mask)

    data_dict = {
        "input_ids": np.array(input_ids_list),
        "attention_mask": np.array(attention_mask_list),
        "offsets": np.array(offsets_list),
        "raw_text": np.array(raw_text_list),
        "sentiments": np.array(sentiments_list),
    }

    if not is_test:
        data_dict["start_targets"] = np.array(start_targets_list)
        data_dict["end_targets"] = np.array(end_targets_list)
        data_dict["span_masks"] = np.array(span_masks_list)
        data_dict["selected_text"] = np.array(selected_text_list)

    return data_dict


def get_loaders(debug=False, load_cached_data=True):
    # Ensure working directory exists
    os.makedirs(Config.base_dir, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Strategy: Filter out 'neutral' tweets from training set only
    if not Config.train_on_neutral:
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    def get_data(df, split_name, is_test=False):
        cache_prefix = os.path.join(Config.base_dir, f"cached_{split_name}")
        if debug:
            cache_prefix += "_debug"

        numeric_keys = ["input_ids", "attention_mask", "offsets"]
        if not is_test:
            numeric_keys += ["start_targets", "end_targets", "span_masks"]

        meta_path = f"{cache_prefix}_meta.parquet"

        # Check if all required files exist
        cache_exists = all(
            [os.path.exists(f"{cache_prefix}_{k}.npy") for k in numeric_keys]
        ) and os.path.exists(meta_path)

        if load_cached_data and cache_exists:
            print(f"Loading cached data for {split_name}...")
            data_dict = {}
            for k in numeric_keys:
                data_dict[k] = np.load(f"{cache_prefix}_{k}.npy")

            # Load text metadata from parquet to avoid pickle issues
            meta_df = pd.read_parquet(meta_path)
            data_dict["raw_text"] = meta_df["text"].values
            data_dict["sentiments"] = meta_df["sentiment"].values
            if not is_test:
                data_dict["selected_text"] = meta_df["selected_text"].values
            return data_dict
        else:
            print(f"Processing data for {split_name}...")
            data_dict = process_data(df, tokenizer, Config.max_len, is_test)

            # Save numeric arrays as .npy
            for k in numeric_keys:
                np.save(f"{cache_prefix}_{k}.npy", data_dict[k])

            # Save text metadata as parquet
            meta_dict = {
                "text": data_dict["raw_text"],
                "sentiment": data_dict["sentiments"],
            }
            if not is_test:
                meta_dict["selected_text"] = data_dict["selected_text"]

            pd.DataFrame(meta_dict).to_parquet(meta_path, index=False)

            return data_dict

    train_data = get_data(train_df, "train", is_test=False)
    val_data = get_data(val_df, "val", is_test=False)
    test_data = get_data(test_df, "test", is_test=True)

    train_dataset = TweetDataset(train_data, is_test=False)
    val_dataset = TweetDataset(val_data, is_test=False)
    test_dataset = TweetDataset(test_data, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
