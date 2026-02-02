import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import load_metadata, is_semiotic
from library.tokenizer import HybridTokenizer


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Cascade Transformer.
    Wraps pre-tokenized encoder (char-level) and decoder (BPE-level) sequences.
    """

    def __init__(self, enc_ids, dec_ids):
        self.enc_ids = enc_ids
        self.dec_ids = dec_ids

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        # Return as tensors
        return {
            "encoder_input": torch.tensor(self.enc_ids[idx], dtype=torch.long),
            "decoder_target": torch.tensor(self.dec_ids[idx], dtype=torch.long),
        }


def collate_fn(batch, pad_enc_id, pad_dec_id):
    """
    Custom collate function to handle dynamic padding for encoder and decoder.
    """
    enc_inputs = [item["encoder_input"] for item in batch]
    dec_targets = [item["decoder_target"] for item in batch]

    # Pad sequences
    # batch_first=True results in (Batch, Seq_Len)
    enc_padded = pad_sequence(enc_inputs, batch_first=True, padding_value=pad_enc_id)
    dec_padded = pad_sequence(dec_targets, batch_first=True, padding_value=pad_dec_id)

    return enc_padded, dec_padded


class DatasetManager:
    """
    Manages data preparation, balancing, tokenization, and caching for the Hybrid Cascade.
    """

    def __init__(self, config: Config, tokenizer: HybridTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_dataloaders(self, load_cached_data=True):
        """
        Main entry point to get training and validation dataloaders.
        """
        # Process or load Training Data
        train_df = self._process_split("train", load_cached_data)

        # Process or load Validation Data
        # Validation data is NOT balanced/upsampled to preserve real distribution evaluation
        val_df = self._process_split("val", load_cached_data)

        # Create Datasets
        train_dataset = NormalizationDataset(
            train_df["enc_ids"].tolist(), train_df["dec_ids"].tolist()
        )
        val_dataset = NormalizationDataset(
            val_df["enc_ids"].tolist(), val_df["dec_ids"].tolist()
        )

        # Create DataLoaders
        # We use a lambda for collate_fn to pass the specific pad IDs
        collate = lambda b: collate_fn(
            b,
            self.tokenizer.char2id[self.tokenizer.PAD_TOKEN],
            self.tokenizer.bpe_pad_id,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _process_split(self, split, load_cached_data):
        """
        Handles the logic for loading, filtering, balancing, and tokenizing a specific split.
        Implements strict caching.
        """
        cache_filename = f"processed_{split}.parquet"
        cache_path = self.config.get_artifact_path(cache_filename)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading processed {split} data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        print(f"Processing {split} data from scratch...")

        # Load raw metadata
        df = load_metadata(split)

        # Debugging: Subsample if configured
        if self.config.debug and self.config.debug_sample_size:
            print(f"Debug mode: subsampling {self.config.debug_sample_size} rows.")
            df = df.iloc[: self.config.debug_sample_size].copy()

        # Ensure order
        if "token_id" in df.columns:
            df["token_id_int"] = df["token_id"].astype(int)
            df.sort_values(["sentence_id", "token_id_int"], inplace=True)

        # Generate Context (Prev/Next)
        self._generate_context(df)

        # Filter Semiotic Tokens
        # We only train the Neural Net on tokens containing digits or latin chars
        print(f"Filtering semiotic tokens for {split}...")
        mask_semiotic = df["before"].apply(is_semiotic)
        df_filtered = df[mask_semiotic].copy()
        print(f"  Original: {len(df)} -> Filtered: {len(df_filtered)}")

        # Balance Classes (Only for Training)
        if split == "train":
            df_filtered = self._balance_classes(df_filtered)

        # Tokenize
        print(f"Tokenizing {split} data...")
        df_processed = self._tokenize_data(df_filtered)

        # Save Cache
        print(f"Saving {split} cache to {cache_path}...")
        # Parquet handles lists of ints efficiently
        df_processed.to_parquet(cache_path, index=False)

        return df_processed

    def _generate_context(self, df):
        """
        Adds 'prev_token' and 'next_token' columns to the dataframe.
        """
        # Shift to get previous and next tokens
        df["prev_token"] = df["before"].shift(1).fillna("<START>")
        df["next_token"] = df["before"].shift(-1).fillna("<END>")

        # Shift sentence_id to detect boundaries
        df["prev_sent"] = df["sentence_id"].shift(1)
        df["next_sent"] = df["sentence_id"].shift(-1)

        # Apply boundaries
        mask_start = df["prev_sent"] != df["sentence_id"]
        df.loc[mask_start, "prev_token"] = "<START>"

        mask_end = df["next_sent"] != df["sentence_id"]
        df.loc[mask_end, "next_token"] = "<END>"

    def _balance_classes(self, df):
        """
        Applies Inverse Frequency Upsampling to rare classes.
        Target count is based on the 'DATE' class frequency.
        """
        print("Balancing classes...")
        class_counts = df["class"].value_counts()

        # Determine target count (Count of DATE class)
        if "DATE" in class_counts:
            target_count = class_counts["DATE"]
        else:
            # Fallback if DATE is missing (unlikely in full data)
            target_count = class_counts.max() if not class_counts.empty else 0

        print(f"  Target count (DATE): {target_count}")

        dfs = []
        for cls, count in class_counts.items():
            cls_df = df[df["class"] == cls]

            if count < target_count:
                # Upsample
                # replace=True allows sampling more than available rows
                cls_df_resampled = cls_df.sample(
                    n=target_count, replace=True, random_state=self.config.seed
                )
                dfs.append(cls_df_resampled)
            else:
                # Keep as is (don't downsample dominant classes like CARDINAL if they are larger)
                dfs.append(cls_df)

        balanced_df = pd.concat(dfs)
        # Shuffle
        balanced_df = balanced_df.sample(
            frac=1, random_state=self.config.seed
        ).reset_index(drop=True)
        print(f"  Balanced size: {len(balanced_df)}")
        return balanced_df

    def _tokenize_data(self, df):
        """
        Converts text columns to lists of IDs.
        Encoder: [Prev] <SEP> <START> [Target] <END> <SEP> [Next]
        Decoder: <BOS> [Target] <EOS>
        """
        enc_ids_list = []
        dec_ids_list = []

        # Pre-fetch special IDs for speed
        SEP_ID = self.tokenizer.SEP_ID
        START_ID = self.tokenizer.START_ID
        END_ID = self.tokenizer.END_ID

        BOS_ID = self.tokenizer.bpe_bos_id
        EOS_ID = self.tokenizer.bpe_eos_id

        # Iterate over rows
        # Using zip for speed
        iterator = zip(
            df["prev_token"].astype(str),
            df["before"].astype(str),
            df["next_token"].astype(str),
            df["after"].astype(str),
        )

        for prev, curr, next_tok, target in iterator:
            # --- Encoder ---
            # Encode parts
            p_ids = self.tokenizer.encode_char(prev)
            c_ids = self.tokenizer.encode_char(curr)
            n_ids = self.tokenizer.encode_char(next_tok)

            # Construct sequence: p + SEP + START + c + END + SEP + n
            # Calculate lengths to handle truncation
            # Fixed overhead: 4 tokens (SEP, START, END, SEP)
            overhead = 4
            available = self.config.max_enc_len - overhead - len(c_ids)

            if available < 0:
                # Current token is too long, truncate it
                c_ids = c_ids[: self.config.max_enc_len - overhead]
                p_ids = []
                n_ids = []
            else:
                # Distribute available space between prev and next
                half = available // 2

                # Truncate prev (keep suffix)
                if len(p_ids) > half:
                    p_ids = p_ids[-half:]

                # Truncate next (keep prefix)
                remaining = available - len(p_ids)
                if len(n_ids) > remaining:
                    n_ids = n_ids[:remaining]

            # Assemble
            full_enc = p_ids + [SEP_ID, START_ID] + c_ids + [END_ID, SEP_ID] + n_ids
            enc_ids_list.append(full_enc)

            # --- Decoder ---
            # BPE encode target
            t_ids = self.tokenizer.encode_bpe(target)
            # Add BOS/EOS
            full_dec = [BOS_ID] + t_ids + [EOS_ID]

            # Truncate decoder if necessary
            if len(full_dec) > self.config.max_dec_len:
                full_dec = full_dec[: self.config.max_dec_len - 1] + [EOS_ID]

            dec_ids_list.append(full_dec)

        # Return new DataFrame
        return pd.DataFrame({"enc_ids": enc_ids_list, "dec_ids": dec_ids_list})
