import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Stores pre-tokenized inputs and targets to minimize runtime overhead.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        start_pos,
        end_pos,
        offsets,
        orig_texts,
        sentiments,
        selected_texts=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.offsets = offsets
        self.orig_texts = orig_texts
        self.sentiments = sentiments
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        data = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "start_positions": torch.tensor(self.start_pos[idx], dtype=torch.long),
            "end_positions": torch.tensor(self.end_pos[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "text": str(self.orig_texts[idx]),
            "sentiment": str(self.sentiments[idx]),
        }

        # Include selected_text if available (for validation/debugging)
        if self.selected_texts is not None:
            data["selected_text"] = str(self.selected_texts[idx])

        return data


def _process_data(df, tokenizer, max_len, is_test=False, filter_impossible=False):
    """
    Tokenizes data and computes start/end token indices for the selected text.
    Cite solution_lesson_node_00013: Joint Probability Maximization for Constrained Span Extraction
    (Filtering impossible rows helps training stability).
    """
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []
    offsets_list = []
    valid_indices = []

    for idx, row in enumerate(df.itertuples(index=False)):
        text = str(row.text)
        sentiment = str(row.sentiment)

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
        offsets = encoded["offset_mapping"]
        seq_ids = encoded.sequence_ids()

        start_idx = 0
        end_idx = 0
        keep_row = True

        if not is_test:
            selected_text = str(row.selected_text)
            start_char = text.find(selected_text)
            if start_char == -1:
                start_char = text.find(selected_text.strip())

            if start_char == -1:
                if filter_impossible:
                    keep_row = False
                else:
                    start_char = 0
                    end_char = len(text)

            if keep_row:
                if start_char != -1:
                    end_char = start_char + len(selected_text)
                else:
                    end_char = len(text)

                token_indices = [i for i, s in enumerate(seq_ids) if s == 1]

                if not token_indices:
                    if filter_impossible:
                        keep_row = False
                    else:
                        start_idx = 0
                        end_idx = 0
                else:
                    s_token = token_indices[0]
                    e_token = token_indices[-1]

                    for i_tok in token_indices:
                        off_start, off_end = offsets[i_tok]
                        if off_start <= start_char < off_end:
                            s_token = i_tok
                            break
                        if off_start > start_char:
                            s_token = i_tok
                            break

                    for i_tok in token_indices:
                        off_start, off_end = offsets[i_tok]
                        if off_start < end_char <= off_end:
                            e_token = i_tok
                            break

                    start_idx = s_token
                    end_idx = e_token

                if start_idx >= max_len:
                    start_idx = max_len - 1
                if end_idx >= max_len:
                    end_idx = max_len - 1
                if start_idx > end_idx:
                    end_idx = start_idx

        if keep_row:
            input_ids.append(ids)
            attention_masks.append(mask)
            start_positions.append(start_idx)
            end_positions.append(end_idx)
            offsets_list.append(offsets)
            valid_indices.append(idx)

    return (
        np.array(input_ids),
        np.array(attention_masks),
        np.array(start_positions),
        np.array(end_positions),
        np.array(offsets_list),
        np.array(valid_indices),
    )


def get_data(load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    if Config.FILTER_NEUTRAL_TRAIN:
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)

    def get_split(df, split_name, is_test=False, filter_impossible=False):
        cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        prefix = os.path.join(cache_dir, split_name)
        files = {
            "ids": f"{prefix}_ids.npy",
            "masks": f"{prefix}_masks.npy",
            "start": f"{prefix}_start.npy",
            "end": f"{prefix}_end.npy",
            "offsets": f"{prefix}_offsets.npy",
            "valid_idx": f"{prefix}_valid.npy",
        }

        cache_exists = all(os.path.exists(f) for f in files.values())

        if load_cached_data and cache_exists:
            input_ids = np.load(files["ids"])
            attention_masks = np.load(files["masks"])
            start_pos = np.load(files["start"])
            end_pos = np.load(files["end"])
            offsets = np.load(files["offsets"])
            valid_indices = np.load(files["valid_idx"])
        else:
            input_ids, attention_masks, start_pos, end_pos, offsets, valid_indices = (
                _process_data(df, tokenizer, Config.MAX_LEN, is_test, filter_impossible)
            )
            np.save(files["ids"], input_ids)
            np.save(files["masks"], attention_masks)
            np.save(files["start"], start_pos)
            np.save(files["end"], end_pos)
            np.save(files["offsets"], offsets)
            np.save(files["valid_idx"], valid_indices)

        df_filtered = df.iloc[valid_indices].reset_index(drop=True)

        return TweetDataset(
            input_ids=input_ids,
            attention_masks=attention_masks,
            start_pos=start_pos,
            end_pos=end_pos,
            offsets=offsets,
            orig_texts=df_filtered["text"].values,
            sentiments=df_filtered["sentiment"].values,
            selected_texts=df_filtered["selected_text"].values if not is_test else None,
        )

    train_cache_name = "train"
    if Config.FILTER_NEUTRAL_TRAIN:
        train_cache_name += "_no_neutral"
    if Config.FILTER_IMPOSSIBLE_TRAIN:
        train_cache_name += "_no_impossible"

    train_dataset = get_split(
        train_df,
        train_cache_name,
        is_test=False,
        filter_impossible=Config.FILTER_IMPOSSIBLE_TRAIN,
    )
    val_dataset = get_split(val_df, "val", is_test=False)
    test_dataset = get_split(test_df, "test", is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
