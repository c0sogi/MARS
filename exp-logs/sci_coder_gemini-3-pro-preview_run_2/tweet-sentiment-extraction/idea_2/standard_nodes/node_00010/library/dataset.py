import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from library.config import Config


class TweetDataset(torch.utils.data.Dataset):
    """
    Dataset class for Tweet Sentiment Extraction.
    Handles tokenization, input construction, and span label alignment.
    """

    def __init__(self, mode: str, config: Config, load_cached_data: bool = True):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            config (Config): Configuration object.
            load_cached_data (bool): Whether to load processed data from cache.
        """
        self.mode = mode
        self.config = config
        self.max_len = config.MAX_LEN
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Select data path based on mode
        if self.mode == "train":
            self.path = config.TRAIN_PATH
        elif self.mode == "val":
            self.path = config.VAL_PATH
        else:
            self.path = config.TEST_PATH

        # Load raw dataframe for text/ID access
        self.df = pd.read_csv(self.path)
        # Ensure text columns are strings (handle potential NaNs from raw loading if any)
        self.df["text"] = self.df["text"].astype(str).fillna("")
        if "selected_text" in self.df.columns:
            self.df["selected_text"] = self.df["selected_text"].astype(str).fillna("")

        # Process or load features
        self.features = self.process_data(load_cached_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(self.features["ids"][idx], dtype=torch.long),
            "mask": torch.tensor(self.features["mask"][idx], dtype=torch.long),
            "targets_start": torch.tensor(
                self.features["targets_start"][idx], dtype=torch.long
            ),
            "targets_end": torch.tensor(
                self.features["targets_end"][idx], dtype=torch.long
            ),
            "offsets": torch.tensor(self.features["offsets"][idx], dtype=torch.long),
            "textID": str(self.df.iloc[idx]["textID"]),
            "text": str(self.df.iloc[idx]["text"]),
            "sentiment": str(self.df.iloc[idx]["sentiment"]),
        }

    def process_data(self, load_cached_data: bool):
        """
        Processes the dataframe into model inputs or loads from cache.
        """
        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        # Include model name in cache filename to prevent conflicts when switching models
        model_slug = self.config.MODEL_NAME.replace("/", "-")
        cache_path = os.path.join(
            self.config.CACHE_DIR, f"cached_{self.mode}_{self.max_len}_{model_slug}.npz"
        )

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached data from {cache_path}...")
                data = np.load(cache_path)
                return {
                    "ids": data["ids"],
                    "mask": data["mask"],
                    "targets_start": data["targets_start"],
                    "targets_end": data["targets_end"],
                    "offsets": data["offsets"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process Data
        print(f"Processing {self.mode} data...")

        input_ids_list = []
        attention_mask_list = []
        offsets_list = []
        start_labels_list = []
        end_labels_list = []

        for idx, row in self.df.iterrows():
            text = str(row["text"])
            sentiment = str(row["sentiment"])
            selected_text = (
                str(row["selected_text"]) if "selected_text" in row else None
            )

            # Tokenize Sentiment
            # RoBERTa: <s> sentiment </s>
            # We manually construct: [CLS] sentiment [SEP] text [SEP]
            # IDs: 0, sent_tokens, 2, text_tokens, 2
            sent_tokens = self.tokenizer.encode(sentiment, add_special_tokens=False)

            # Tokenize Text
            # We use encode_plus to get offsets
            # Reserve space for: [CLS] + sent + [SEP] + ... + [SEP]
            # Count: 1 + len(sent) + 1 + ... + 1 = len(sent) + 3
            max_text_len = self.max_len - len(sent_tokens) - 3

            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                max_length=max_text_len,
                truncation=True,
            )

            text_ids = encoded["input_ids"]
            text_offsets = encoded["offset_mapping"]

            # Construct Input Sequence
            # Format: <s> <sentiment> </s> <text> </s>
            input_ids = [0] + sent_tokens + [2] + text_ids + [2]
            attention_mask = [1] * len(input_ids)

            # Construct Offsets
            # (0,0) for special tokens and sentiment
            # text_offsets for text
            # (0,0) for final SEP
            offsets = [(0, 0)] * (len(sent_tokens) + 2) + text_offsets + [(0, 0)]

            # Padding
            pad_len = self.max_len - len(input_ids)
            if pad_len > 0:
                input_ids += [1] * pad_len  # 1 is pad token for RoBERTa
                attention_mask += [0] * pad_len
                offsets += [(0, 0)] * pad_len

            # Target Alignment
            target_start = 0
            target_end = 0

            if selected_text and len(selected_text) > 0:
                # Find character indices of selected_text in text
                start_char = text.find(selected_text)
                if start_char != -1:
                    end_char = start_char + len(selected_text)

                    # Offset for text tokens in the input_ids list
                    # [0, sent..., 2] -> len(sent_tokens) + 2
                    tokens_offset = len(sent_tokens) + 2

                    found_start = False

                    for i, (o_start, o_end) in enumerate(text_offsets):
                        # Check for overlap between token span and selected span
                        # Token span: o_start, o_end
                        # Selected span: start_char, end_char
                        if o_start < end_char and o_end > start_char:
                            if not found_start:
                                target_start = tokens_offset + i
                                found_start = True
                            target_end = tokens_offset + i

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            offsets_list.append(offsets)
            start_labels_list.append(target_start)
            end_labels_list.append(target_end)

        # Convert to NumPy arrays
        data = {
            "ids": np.array(input_ids_list, dtype=np.int64),
            "mask": np.array(attention_mask_list, dtype=np.int64),
            "targets_start": np.array(start_labels_list, dtype=np.int64),
            "targets_end": np.array(end_labels_list, dtype=np.int64),
            "offsets": np.array(offsets_list, dtype=np.int64),
        }

        # 3. Save Cache
        print(f"Saving processed data to {cache_path}...")
        np.savez(cache_path, **data)

        return data
