import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class TextNormalizationDataset(Dataset):
    """
    Dataset for Text Normalization using Token Classification.
    Handles loading, subsampling, and tokenization/alignment.
    """

    def __init__(self, split="train", load_cached_data=True, debug_size=None):
        self.split = split
        self.debug_size = debug_size
        self.max_len = Config.MAX_LEN
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Set seed for reproducibility in any random ops within init
        np.random.seed(Config.SEED)

        # Load and process data (with caching)
        self.data = self._load_data(split, load_cached_data)

        if self.debug_size is not None:
            self.data = self.data.iloc[: self.debug_size].reset_index(drop=True)

    def _load_data(self, split, load_cached_data):
        """
        Loads data from CSV, groups by sentence, subsamples (if train),
        and caches the result as a Parquet file.
        """
        cache_filename = f"{split}_processed.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing {split} data from source...")

        if split == "train":
            input_path = Config.TRAIN_DATA_PATH
        elif split == "val":
            input_path = Config.VAL_DATA_PATH
        elif split == "test":
            input_path = Config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load raw CSV
        # keep_default_na=False prevents 'null'/'nan' strings from becoming NaN
        df_raw = pd.read_csv(
            input_path,
            keep_default_na=False,
            dtype={"sentence_id": int, "token_id": int},
        )

        # Check if labels exist
        has_labels = "class" in df_raw.columns

        # Group by sentence_id
        # We aggregate tokens and other fields into lists
        agg_dict = {"before": list, "id": list}
        if has_labels:
            agg_dict["class"] = list

        # Sort to ensure token order within sentences
        df_raw = df_raw.sort_values(["sentence_id", "token_id"])

        # Grouping
        print(f"Grouping {len(df_raw)} tokens into sentences...")
        grouped = df_raw.groupby("sentence_id", as_index=False).agg(agg_dict)

        # Rename columns
        grouped = grouped.rename(
            columns={"before": "tokens", "id": "token_ids", "class": "labels"}
        )

        # Subsampling Strategy (Only for Train)
        if split == "train" and has_labels:
            print("Applying subsampling to balance PLAIN class...")

            # Identify "purely PLAIN" sentences
            # A sentence is pure plain if ALL its tokens are labeled 'PLAIN'
            def is_pure_plain(labels_list):
                unique_labels = set(labels_list)
                return unique_labels == {"PLAIN"} or len(unique_labels) == 0

            grouped["is_pure_plain"] = grouped["labels"].apply(is_pure_plain)

            plain_df = grouped[grouped["is_pure_plain"]]
            non_plain_df = grouped[~grouped["is_pure_plain"]]

            print(f"  Total Sentences: {len(grouped)}")
            print(f"  Pure PLAIN Sentences: {len(plain_df)}")
            print(f"  Interesting Sentences: {len(non_plain_df)}")

            # Subsample the PLAIN part
            plain_sampled = plain_df.sample(
                frac=Config.PLAIN_SUBSAMPLE_RATIO, random_state=Config.SEED
            )

            print(f"  Kept PLAIN Sentences: {len(plain_sampled)}")

            # Combine and shuffle
            grouped = pd.concat([non_plain_df, plain_sampled])
            grouped = grouped.sample(frac=1, random_state=Config.SEED).reset_index(
                drop=True
            )

            # Cleanup
            grouped = grouped.drop(columns=["is_pure_plain"])
            print(f"  Final Training Set Size: {len(grouped)} sentences")

        # Save to cache
        print(f"Saving processed {split} data to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        grouped.to_parquet(cache_path)

        return grouped

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = list(row["tokens"])  # Ensure list

        # Labels might not exist for test set
        labels = list(row["labels"]) if "labels" in row else None

        # Tokenize
        # We use is_split_into_words=True because we have pre-tokenized text (list of words)
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # We want lists to convert to tensors manually
        )

        input_ids = torch.tensor(tokenized_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(
            tokenized_inputs["attention_mask"], dtype=torch.long
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }

        # Map word_ids to tensor (handling None)
        word_ids = tokenized_inputs.word_ids()
        word_ids_clean = [w if w is not None else -1 for w in word_ids]
        result["word_ids"] = torch.tensor(word_ids_clean, dtype=torch.long)

        # Align Labels
        if labels is not None:
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                # Special tokens (None) -> -100
                if word_idx is None:
                    label_ids.append(-100)
                # New word start
                elif word_idx != previous_word_idx:
                    # Check for truncation boundary
                    if word_idx < len(labels):
                        label_str = labels[word_idx]
                        label_id = Config.LABEL2ID.get(
                            label_str, Config.LABEL2ID["PLAIN"]
                        )
                        label_ids.append(label_id)
                    else:
                        label_ids.append(-100)
                # Same word (sub-token) -> -100
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            result["labels"] = torch.tensor(label_ids, dtype=torch.long)

        return result
