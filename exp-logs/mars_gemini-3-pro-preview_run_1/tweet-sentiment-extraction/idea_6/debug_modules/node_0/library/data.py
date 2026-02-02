import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from library.config import Config
from library.utils import process_text


class TweetDataset(Dataset):
    """
    Dataset class for Tweet Sentiment Extraction.
    Holds pre-tokenized inputs and targets, along with metadata for evaluation.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_targets=None,
        end_targets=None,
        offsets=None,
        text_ids=None,
        texts=None,
        selected_texts=None,
        sentiments=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.text_ids = text_ids
        self.texts = texts
        self.selected_texts = selected_texts
        self.sentiments = sentiments

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
        }

        if self.start_targets is not None:
            data["start_targets"] = torch.tensor(
                self.start_targets[item], dtype=torch.float
            )
            data["end_targets"] = torch.tensor(
                self.end_targets[item], dtype=torch.float
            )

        if self.offsets is not None:
            data["offsets"] = torch.tensor(self.offsets[item], dtype=torch.long)

        if self.text_ids is not None:
            data["textID"] = self.text_ids[item]

        if self.texts is not None:
            data["text"] = self.texts[item]

        if self.selected_texts is not None:
            data["selected_text"] = self.selected_texts[item]

        if self.sentiments is not None:
            data["sentiment"] = self.sentiments[item]

        return data


def get_gaussian_target(index, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at the given index.
    Used for label smoothing on the start/end positions.
    """
    x = np.arange(length)
    target = np.exp(-0.5 * ((x - index) / sigma) ** 2)
    # Normalize to form a valid probability distribution
    target = target / target.sum()
    return target


def process_data_split(df, tokenizer, max_len, is_train=True, description="Processing"):
    """
    Tokenizes and processes a dataframe split.
    Implements the 'Normalize-First' strategy to ensure alignment.
    """
    n_samples = len(df)
    input_ids_list = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask_list = np.zeros((n_samples, max_len), dtype=np.int32)
    offsets_list = np.zeros((n_samples, max_len, 2), dtype=np.int32)

    start_targets_list = []
    end_targets_list = []

    text_ids = []
    texts = []
    selected_texts = []
    sentiments = []

    for i, row in tqdm(df.iterrows(), total=n_samples, desc=description):
        text_id = row["textID"]
        raw_text = row["text"]
        sentiment = row["sentiment"]

        # Normalize text strictly before tokenization
        text = process_text(raw_text)

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # We treat 'sentiment' as the first sequence and 'text' as the second.
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation="only_second",  # Truncate text, not sentiment
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        input_ids_list[i] = input_ids
        attention_mask_list[i] = attention_mask
        offsets_list[i] = offsets

        text_ids.append(text_id)
        texts.append(text)  # Store normalized text
        sentiments.append(sentiment)

        if is_train:
            raw_selected = row["selected_text"]
            selected_text = process_text(raw_selected)
            selected_texts.append(selected_text)  # Store normalized selected_text

            # Find start/end in normalized text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: use full text if not found (rare with normalization)
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            token_start_index = 0
            token_end_index = 0
            found_start = False

            # Map chars to tokens (only for sequence_id == 1, which is the text)
            for idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
                if seq_id != 1:
                    continue

                # offset is (start, end) relative to the text string
                if (
                    not found_start
                    and offset[0] <= start_char
                    and offset[1] > start_char
                ):
                    token_start_index = idx
                    found_start = True

                if offset[0] < end_char:
                    token_end_index = idx

            # Ensure valid span
            if token_end_index < token_start_index:
                token_end_index = token_start_index

            # Generate Gaussian targets
            start_target = get_gaussian_target(
                token_start_index, max_len, Config.target_smoothing_sigma
            )
            end_target = get_gaussian_target(
                token_end_index, max_len, Config.target_smoothing_sigma
            )

            start_targets_list.append(start_target)
            end_targets_list.append(end_target)
        else:
            selected_texts.append(None)

    if is_train:
        start_targets_list = np.array(start_targets_list, dtype=np.float32)
        end_targets_list = np.array(end_targets_list, dtype=np.float32)
    else:
        start_targets_list = None
        end_targets_list = None

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "offsets": offsets_list,
        "start_targets": start_targets_list,
        "end_targets": end_targets_list,
        "text_ids": np.array(text_ids),
        "texts": np.array(texts),
        "selected_texts": np.array(selected_texts),
        "sentiments": np.array(sentiments),
    }


def get_loaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test.
    Handles caching, filtering of neutral tweets, and data loading.
    """
    os.makedirs(Config.working_dir, exist_ok=True)

    # 1. Load Metadata
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # 2. Filter Neutrals for Training/Validation if Configured
    # We do not filter Test set here; inference logic handles it.
    if Config.filter_neutral:
        df_train = df_train[df_train["sentiment"] != "neutral"].reset_index(drop=True)
        df_val = df_val[df_val["sentiment"] != "neutral"].reset_index(drop=True)

    # Debug mode: subset data
    if Config.debug:
        df_train = df_train.head(Config.debug_sample_size)
        df_val = df_val.head(Config.debug_sample_size)
        df_test = df_test.head(Config.debug_sample_size)

    # 3. Process or Load Cache
    datasets = {}
    splits = [("train", df_train), ("val", df_val), ("test", df_test)]

    for split_name, df in splits:
        cache_prefix = os.path.join(Config.working_dir, f"cached_{split_name}")
        if Config.debug:
            cache_prefix += "_debug"

        files = ["input_ids.npy", "attention_mask.npy", "offsets.npy", "meta.parquet"]
        if split_name != "test":
            files.extend(["start_targets.npy", "end_targets.npy"])

        cache_exists = all(os.path.exists(f"{cache_prefix}_{f}") for f in files)

        if load_cached_data and cache_exists:
            # print(f"Loading {split_name} data from cache...")
            data_dict = {}
            data_dict["input_ids"] = np.load(f"{cache_prefix}_input_ids.npy")
            data_dict["attention_mask"] = np.load(f"{cache_prefix}_attention_mask.npy")
            data_dict["offsets"] = np.load(f"{cache_prefix}_offsets.npy")

            meta_df = pd.read_parquet(f"{cache_prefix}_meta.parquet")
            data_dict["text_ids"] = meta_df["textID"].values
            data_dict["texts"] = meta_df["text"].values
            data_dict["sentiments"] = meta_df["sentiment"].values
            data_dict["selected_texts"] = (
                meta_df["selected_text"].values
                if "selected_text" in meta_df.columns
                else None
            )

            if split_name != "test":
                data_dict["start_targets"] = np.load(
                    f"{cache_prefix}_start_targets.npy"
                )
                data_dict["end_targets"] = np.load(f"{cache_prefix}_end_targets.npy")
            else:
                data_dict["start_targets"] = None
                data_dict["end_targets"] = None
        else:
            # print(f"Processing {split_name} data...")
            data_dict = process_data_split(
                df,
                tokenizer,
                Config.max_len,
                is_train=(split_name != "test"),
                description=f"Processing {split_name}",
            )

            # Save to cache
            np.save(f"{cache_prefix}_input_ids.npy", data_dict["input_ids"])
            np.save(f"{cache_prefix}_attention_mask.npy", data_dict["attention_mask"])
            np.save(f"{cache_prefix}_offsets.npy", data_dict["offsets"])

            if split_name != "test":
                np.save(f"{cache_prefix}_start_targets.npy", data_dict["start_targets"])
                np.save(f"{cache_prefix}_end_targets.npy", data_dict["end_targets"])

            meta_data = {
                "textID": data_dict["text_ids"],
                "text": data_dict["texts"],
                "sentiment": data_dict["sentiments"],
            }
            if data_dict["selected_texts"] is not None and split_name != "test":
                meta_data["selected_text"] = data_dict["selected_texts"]

            pd.DataFrame(meta_data).to_parquet(f"{cache_prefix}_meta.parquet")

        datasets[split_name] = TweetDataset(
            input_ids=data_dict["input_ids"],
            attention_mask=data_dict["attention_mask"],
            start_targets=data_dict["start_targets"],
            end_targets=data_dict["end_targets"],
            offsets=data_dict["offsets"],
            text_ids=data_dict["text_ids"],
            texts=data_dict["texts"],
            selected_texts=data_dict["selected_texts"],
            sentiments=data_dict["sentiments"],
        )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
