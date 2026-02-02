import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments
        self.selected_texts = (
            selected_texts if selected_texts is not None else [""] * len(texts)
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        data = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.texts[item]),
            "sentiment": str(self.sentiments[item]),
            "selected_text": str(self.selected_texts[item]),
        }

        if self.start_targets is not None:
            data["start_targets"] = torch.tensor(
                self.start_targets[item], dtype=torch.float
            )
            data["end_targets"] = torch.tensor(
                self.end_targets[item], dtype=torch.float
            )

        return data


def generate_soft_targets(target_idx, max_len, sigma):
    """
    Generates Gaussian-smoothed targets centered at target_idx.
    """
    if target_idx < 0:
        return np.zeros(max_len)

    x = np.arange(max_len)
    target = np.exp(-((x - target_idx) ** 2) / (2 * sigma**2))
    # Normalize to form a probability distribution
    target = target / (target.sum() + 1e-16)
    return target


def process_data(df, tokenizer, max_len, sigma, is_test=False):
    """
    Processes the dataframe into numpy arrays for the model.
    Applies Normalize-First protocol and generates soft targets.
    """
    n_samples = len(df)
    input_ids = np.zeros((n_samples, max_len), dtype=int)
    attention_mask = np.zeros((n_samples, max_len), dtype=int)
    offsets = np.zeros((n_samples, max_len, 2), dtype=int)

    start_targets = np.zeros((n_samples, max_len), dtype=float) if not is_test else None
    end_targets = np.zeros((n_samples, max_len), dtype=float) if not is_test else None

    texts = df["text"].values
    sentiments = df["sentiment"].values
    selected_texts = (
        df["selected_text"].values
        if "selected_text" in df.columns
        else np.array([""] * n_samples)
    )

    for i in range(n_samples):
        text = str(texts[i])
        sentiment = str(sentiments[i])

        # Normalize text as per protocol
        norm_text = normalize_text(text)

        # Tokenize: [CLS] Sentiment [SEP] Text [SEP]
        # Using tokenizer() directly returns BatchEncoding with sequence_ids
        encoded = tokenizer(
            sentiment,
            norm_text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        input_ids[i] = encoded["input_ids"]
        attention_mask[i] = encoded["attention_mask"]
        offsets[i] = encoded["offset_mapping"]

        if not is_test:
            selected_text = str(selected_texts[i])
            norm_selected_text = normalize_text(selected_text)

            # Find the span in the normalized text
            start_char_idx = norm_text.find(norm_selected_text)

            start_token_idx = -1
            end_token_idx = -1

            if start_char_idx != -1:
                end_char_idx = start_char_idx + len(norm_selected_text)

                # Use sequence_ids to identify the tokens belonging to the text part
                seq_ids = encoded.sequence_ids()
                raw_offsets = encoded["offset_mapping"]

                found_start = False

                for idx, (seq_id, offset) in enumerate(zip(seq_ids, raw_offsets)):
                    # seq_id is None for special tokens, 0 for sentiment, 1 for text
                    if seq_id != 1:
                        continue

                    # Check overlap
                    # offset is (start, end) relative to the start of the text sequence
                    if not found_start:
                        if offset[0] >= start_char_idx:
                            start_token_idx = idx
                            found_start = True

                    if found_start:
                        if offset[1] >= end_char_idx:
                            end_token_idx = idx
                            break
                        end_token_idx = idx

            if start_token_idx != -1 and end_token_idx != -1:
                start_targets[i] = generate_soft_targets(
                    start_token_idx, max_len, sigma
                )
                end_targets[i] = generate_soft_targets(end_token_idx, max_len, sigma)
            else:
                # Fallback for edge cases or empty selections
                start_targets[i] = generate_soft_targets(0, max_len, sigma)
                end_targets[i] = generate_soft_targets(0, max_len, sigma)

    return (
        input_ids,
        attention_mask,
        start_targets,
        end_targets,
        offsets,
        texts,
        sentiments,
        selected_texts,
    )


def get_loaders(
    load_cached_data=True,
    batch_size=Config.TRAIN_BATCH_SIZE,
    val_batch_size=Config.VALID_BATCH_SIZE,
    debug=Config.DEBUG,
    debug_size=Config.DEBUG_SIZE,
):
    """
    Main function to load data, process/cache it, and return DataLoaders.
    Filters 'neutral' tweets from training set but keeps them for validation/test.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Filter Neutrals from Training
    train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

    if debug:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    prefix = "debug_" if debug else ""

    cache_files = {
        "train": {
            "input_ids": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_input_ids.npy"
            ),
            "attention_mask": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_attention_mask.npy"
            ),
            "start_targets": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_start_targets.npy"
            ),
            "end_targets": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_end_targets.npy"
            ),
            "offsets": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_offsets.npy"
            ),
            "meta": os.path.join(
                Config.CACHE_DIR, f"{prefix}train_no_neutral_meta.parquet"
            ),
        },
        "val": {
            "input_ids": os.path.join(Config.CACHE_DIR, f"{prefix}val_input_ids.npy"),
            "attention_mask": os.path.join(
                Config.CACHE_DIR, f"{prefix}val_attention_mask.npy"
            ),
            "start_targets": os.path.join(
                Config.CACHE_DIR, f"{prefix}val_start_targets.npy"
            ),
            "end_targets": os.path.join(
                Config.CACHE_DIR, f"{prefix}val_end_targets.npy"
            ),
            "offsets": os.path.join(Config.CACHE_DIR, f"{prefix}val_offsets.npy"),
            "meta": os.path.join(Config.CACHE_DIR, f"{prefix}val_meta.parquet"),
        },
        "test": {
            "input_ids": os.path.join(Config.CACHE_DIR, f"{prefix}test_input_ids.npy"),
            "attention_mask": os.path.join(
                Config.CACHE_DIR, f"{prefix}test_attention_mask.npy"
            ),
            "offsets": os.path.join(Config.CACHE_DIR, f"{prefix}test_offsets.npy"),
            "meta": os.path.join(Config.CACHE_DIR, f"{prefix}test_meta.parquet"),
        },
    }

    datasets = {}

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        paths = cache_files[split]
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            print(f"Loading cached {split} data...")
            input_ids = np.load(paths["input_ids"])
            attention_mask = np.load(paths["attention_mask"])
            offsets = np.load(paths["offsets"])
            meta_df = pd.read_parquet(paths["meta"])
            texts = meta_df["text"].values
            sentiments = meta_df["sentiment"].values
            selected_texts = (
                meta_df["selected_text"].values
                if "selected_text" in meta_df.columns
                else None
            )

            if split != "test":
                start_targets = np.load(paths["start_targets"])
                end_targets = np.load(paths["end_targets"])
            else:
                start_targets = None
                end_targets = None

        else:
            print(f"Processing {split} data...")
            is_test_split = split == "test"

            out = process_data(
                df, tokenizer, Config.MAX_LEN, Config.SIGMA, is_test=is_test_split
            )
            (
                input_ids,
                attention_mask,
                start_targets,
                end_targets,
                offsets,
                texts,
                sentiments,
                selected_texts,
            ) = out

            np.save(paths["input_ids"], input_ids)
            np.save(paths["attention_mask"], attention_mask)
            np.save(paths["offsets"], offsets)

            meta_dict = {"text": texts, "sentiment": sentiments}
            if selected_texts is not None:
                meta_dict["selected_text"] = selected_texts
            pd.DataFrame(meta_dict).to_parquet(paths["meta"], index=False)

            if not is_test_split:
                np.save(paths["start_targets"], start_targets)
                np.save(paths["end_targets"], end_targets)

        datasets[split] = TweetDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_targets=start_targets,
            end_targets=end_targets,
            offsets=offsets,
            texts=texts,
            sentiments=sentiments,
            selected_texts=selected_texts,
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
