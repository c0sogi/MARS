import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocabulary import CharVocab


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Handles loading, context extraction, balancing, and tokenization.
    """

    def __init__(self, split, vocab, load_cached_data=True, debug=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            vocab (CharVocab): Vocabulary instance for encoding.
            load_cached_data (bool): Whether to load from parquet cache.
            debug (bool): If True, use a small subset of data.
        """
        self.split = split
        self.vocab = vocab
        self.debug = debug or Config.DEBUG

        # Determine file paths based on split
        if split == "train":
            self.raw_path = Config.TRAIN_FILE
            self.cache_path = Config.TRAIN_CACHE
        elif split == "val":
            self.raw_path = Config.VAL_FILE
            self.cache_path = Config.VAL_CACHE
        elif split == "test":
            self.raw_path = Config.TEST_FILE
            self.cache_path = Config.TEST_CACHE
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load data
        self.data = self._load_and_process_data(load_cached_data)

        # Debug slicing
        if self.debug:
            print(
                f"DEBUG Mode: Slicing {split} dataset to {Config.DEBUG_SIZE} samples."
            )
            self.data = (
                self.data.iloc[: Config.DEBUG_SIZE].copy().reset_index(drop=True)
            )

    def _load_and_process_data(self, load_cached_data):
        """
        Loads data from cache if available, otherwise processes raw CSV.
        Implements context window extraction and class balancing.
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"[{self.split}] Loading cached data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                return df
            except Exception as e:
                print(f"[{self.split}] Failed to load cache: {e}. Re-processing...")

        # 2. Process from Scratch
        print(f"[{self.split}] Processing raw data from {self.raw_path}...")

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load raw CSV
        # keep_default_na=False is crucial for tokens like "null" or "nan"
        df = pd.read_csv(self.raw_path, keep_default_na=False)

        # Ensure correct sorting for context extraction
        # (Metadata script groups by sentence, but we sort to be safe)
        if "token_id" in df.columns:
            df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

        # Convert columns to string to avoid type issues
        df["before"] = df["before"].astype(str)
        if "after" in df.columns:
            df["after"] = df["after"].astype(str)

        # --- Context Extraction ---
        # We need prev_token and next_token.
        # We use pandas shift, but must respect sentence boundaries.
        print(f"[{self.split}] Generating context windows...")

        # Shift 'before' column
        df["prev_token"] = df["before"].shift(1).fillna("")
        df["next_token"] = df["before"].shift(-1).fillna("")

        # Shift 'sentence_id' to check boundaries
        df["prev_sid"] = df["sentence_id"].shift(1)
        df["next_sid"] = df["sentence_id"].shift(-1)

        # Mask out context where sentence_id changed
        # If prev_sid != current_sid, then prev_token belongs to previous sentence -> set to empty
        mask_prev = df["prev_sid"] != df["sentence_id"]
        df.loc[mask_prev, "prev_token"] = ""

        mask_next = df["next_sid"] != df["sentence_id"]
        df.loc[mask_next, "next_token"] = ""

        # Construct Input String: "prev|curr|next"
        # We use the separator defined in Config
        sep = Config.SEP_TOKEN
        df["input_text"] = (
            df["prev_token"] + sep + df["before"] + sep + df["next_token"]
        )

        # Drop temporary columns to save memory
        cols_to_drop = ["prev_token", "next_token", "prev_sid", "next_sid"]
        df.drop(columns=cols_to_drop, inplace=True)

        # --- Balancing (Train Only) ---
        if self.split == "train" and "class" in df.columns:
            print(f"[{self.split}] Balancing dataset...")
            initial_len = len(df)

            # Identify tokens that are PLAIN or PUNCT and unchanged
            # We target these for downsampling to force model to learn transformations
            # Note: Config mentions 'PLAIN', prompt mentions 'PLAIN/PUNCT'.
            # We include both for robustness if they are unchanged.
            classes_to_downsample = ["PLAIN", "PUNCT"]
            is_candidate = (df["class"].isin(classes_to_downsample)) & (
                df["before"] == df["after"]
            )

            # Create a keep mask
            # 1. Keep everything that is NOT a candidate (changed tokens, or other classes)
            # 2. For candidates, keep with probability PLAIN_DOWNSAMPLE_RATIO
            rng = np.random.default_rng(Config.SEED)
            keep_probs = rng.random(len(df))

            # Logic: Keep if (NOT candidate) OR (random < ratio)
            keep_mask = (~is_candidate) | (keep_probs < Config.PLAIN_DOWNSAMPLE_RATIO)

            df = df[keep_mask].reset_index(drop=True)
            print(f"[{self.split}] Downsampled from {initial_len} to {len(df)} rows.")

        # --- Save to Cache ---
        print(f"[{self.split}] Saving processed data to {self.cache_path}")
        df.to_parquet(self.cache_path, index=False)

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample.
        Output format:
            src: Tensor [L_src] (Indices)
            tgt: Tensor [L_tgt] (Indices)
            id: str (for submission)
        """
        row = self.data.iloc[idx]

        # 1. Prepare Source
        input_text = row["input_text"]
        # Add EOS to source to mark end of sequence clearly
        src_indices = self.vocab.encode(input_text, add_sos=False, add_eos=True)

        # 2. Prepare Target
        if "after" in row:
            target_text = row["after"]
            # Target needs SOS (start) and EOS (end)
            tgt_indices = self.vocab.encode(target_text, add_sos=True, add_eos=True)
        else:
            # Test set might not have 'after'
            # Return dummy target or empty
            tgt_indices = [Config.SOS_IDX, Config.EOS_IDX]

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "tgt": torch.tensor(tgt_indices, dtype=torch.long),
            "id": row["id"],
            "raw_before": row["before"],
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable length sequences.
        Pads sequences to the max length in the batch.
        """
        # Unpack batch
        src_list = [item["src"] for item in batch]
        tgt_list = [item["tgt"] for item in batch]
        ids = [item["id"] for item in batch]
        raw_before = [item["raw_before"] for item in batch]

        # Get lengths
        src_lens = torch.tensor([len(s) for s in src_list], dtype=torch.long)
        tgt_lens = torch.tensor([len(t) for t in tgt_list], dtype=torch.long)

        # Pad sequences
        # batch_first=True -> [Batch, Seq_Len]
        src_padded = pad_sequence(
            src_list, batch_first=True, padding_value=Config.PAD_IDX
        )
        tgt_padded = pad_sequence(
            tgt_list, batch_first=True, padding_value=Config.PAD_IDX
        )

        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "src_len": src_lens,
            "tgt_len": tgt_lens,
            "id": ids,
            "raw_before": raw_before,
        }
