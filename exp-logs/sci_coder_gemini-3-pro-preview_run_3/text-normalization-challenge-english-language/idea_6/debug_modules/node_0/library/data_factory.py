import os
import pandas as pd
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import CharTokenizer


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Formats input as: [prev] <SEP> [curr] <SEP> [next]
    """

    def __init__(
        self,
        df,
        tokenizer,
        label_encoder=None,
        mode="train",
        max_len=Config.MAX_INPUT_LEN,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.mode = mode
        self.max_len = max_len

        # Pre-convert columns to lists for faster indexing
        self.before = self.df["before"].astype(str).tolist()
        self.prev = self.df["prev"].astype(str).tolist()
        self.next = self.df["next"].astype(str).tolist()
        self.ids = self.df["id"].astype(str).tolist()

        if self.mode in ["train", "val"]:
            self.after = self.df["after"].astype(str).tolist()
            self.classes = self.df["class"].astype(str).tolist()
        else:
            self.after = None
            self.classes = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct Contextual Input
        # Format: prev <SEP> curr <SEP> next
        p_tok = self.prev[idx]
        c_tok = self.before[idx]
        n_tok = self.next[idx]

        # We construct the string with separators
        # Note: The tokenizer handles character splitting.
        # We manually insert SEP tokens in the list of indices below or use a string repr.
        # Using string representation with special token markers might be ambiguous if not handled carefully.
        # Here we encode separately and concatenate to ensure SEP is treated as a single token.

        prev_ids = self.tokenizer.encode(p_tok, add_special_tokens=False)
        curr_ids = self.tokenizer.encode(c_tok, add_special_tokens=False)
        next_ids = self.tokenizer.encode(n_tok, add_special_tokens=False)
        sep_id = [Config.SEP_IDX]

        # Combine: [prev] + [SEP] + [curr] + [SEP] + [next]
        # We also add SOS at start and EOS at end for the whole sequence
        src_ids = (
            [Config.SOS_IDX]
            + prev_ids
            + sep_id
            + curr_ids
            + sep_id
            + next_ids
            + [Config.EOS_IDX]
        )

        # Truncate if necessary (rare for char level but good practice)
        if len(src_ids) > self.max_len:
            src_ids = src_ids[: self.max_len]

        src_tensor = torch.tensor(src_ids, dtype=torch.long)

        if self.mode in ["train", "val"]:
            # Target
            tgt_text = self.after[idx]
            tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

            # Class Label
            cls_name = self.classes[idx]
            # Handle unseen classes in val by assigning a default or erroring
            # Ideally, LabelEncoder covers all.
            try:
                cls_idx = self.label_encoder.transform([cls_name])[0]
            except ValueError:
                cls_idx = 0  # Fallback or specific UNK class index if defined

            return src_tensor, tgt_tensor, torch.tensor(cls_idx, dtype=torch.long)

        else:
            # Test mode: Return ID and raw text for submission/debugging
            return src_tensor, c_tok, self.ids[idx]


class DataFactory:
    """
    Manages data loading, preprocessing, filtering, and DataLoader creation.
    """

    def __init__(self):
        self.tokenizer = CharTokenizer()
        self.label_encoder = LabelEncoder()
        self.tokenizer_fitted = False
        self.encoder_fitted = False

        # Paths for artifacts
        self.tokenizer_path = os.path.join(Config.WORKING_DIR, "tokenizer.json")
        self.encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.joblib")

    def _add_context(self, df):
        """
        Generates 'prev' and 'next' columns based on sentence_id boundaries.
        """
        # Ensure string types
        df["before"] = df["before"].astype(str)
        if "sentence_id" not in df.columns:
            # Fallback if sentence_id is missing (unlikely based on metadata)
            df["prev"] = Config.SOS_TOKEN
            df["next"] = Config.EOS_TOKEN
            return df

        # Shift to get candidates
        df["prev"] = df["before"].shift(1).fillna(Config.SOS_TOKEN)
        df["next"] = df["before"].shift(-1).fillna(Config.EOS_TOKEN)

        # Handle Sentence Boundaries
        sent_id = df["sentence_id"]
        prev_sent_id = sent_id.shift(1).fillna(-1)
        next_sent_id = sent_id.shift(-1).fillna(-1)

        # If current sentence != prev sentence, prev token is SOS
        df.loc[sent_id != prev_sent_id, "prev"] = Config.SOS_TOKEN
        # If current sentence != next sentence, next token is EOS
        df.loc[sent_id != next_sent_id, "next"] = Config.EOS_TOKEN

        return df

    def _filter_hard_samples(self, df):
        """
        Filters the dataframe to retain only 'Hard' samples.
        Criteria: NOT (PLAIN OR PUNCT OR Alpha)
        """
        if "class" not in df.columns:
            return df

        # 1. Exclude PLAIN and PUNCT
        mask_class = ~df["class"].isin(["PLAIN", "PUNCT"])

        # 2. Exclude purely alphabetic tokens
        # Note: We check the 'before' token.
        mask_alpha = ~df["before"].str.isalpha()

        # Combine masks
        df_filtered = df[mask_class & mask_alpha].copy()
        return df_filtered

    def process_data(
        self, input_path, cache_name, load_cached_data=True, is_train_split=False
    ):
        """
        Loads raw data, adds context, filters (if training split), and caches.
        """
        cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print(f"Processing data from {input_path}...")
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            df = pd.read_parquet(input_path)

            # 1. Add Context (Must be done on full dataset before filtering)
            df = self._add_context(df)

            # 2. Filter if it is the training/val split (Test set has no classes)
            if is_train_split:
                original_len = len(df)
                df = self._filter_hard_samples(df)
                print(
                    f"Filtered {cache_name}: {original_len} -> {len(df)} samples ({len(df)/original_len:.2%})"
                )

            # 3. Save to cache
            print(f"Saving processed data to {cache_path}")
            df.to_parquet(cache_path, index=False)

        return df

    def fit_artifacts(self, df_train):
        """
        Fits the tokenizer and label encoder on the training data.
        """
        print("Fitting tokenizer and label encoder...")
        # Fit Tokenizer on Input (before/prev/next) and Target (after)
        # We concatenate all text to ensure vocab coverage
        texts = (
            df_train["before"].astype(str).tolist()
            + df_train["after"].astype(str).tolist()
            + df_train["prev"].astype(str).tolist()
            + df_train["next"].astype(str).tolist()
        )
        self.tokenizer.fit_on_texts(texts)
        self.tokenizer.save(self.tokenizer_path)
        self.tokenizer_fitted = True

        # Fit Label Encoder
        self.label_encoder.fit(df_train["class"].astype(str))
        joblib.dump(self.label_encoder, self.encoder_path)
        self.encoder_fitted = True
        print(
            f"Vocab Size: {len(self.tokenizer)}, Classes: {len(self.label_encoder.classes_)}"
        )

    def load_artifacts(self):
        """
        Loads tokenizer and label encoder from disk.
        """
        if os.path.exists(self.tokenizer_path):
            self.tokenizer.load(self.tokenizer_path)
            self.tokenizer_fitted = True

        if os.path.exists(self.encoder_path):
            self.label_encoder = joblib.load(self.encoder_path)
            self.encoder_fitted = True

    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences.
        """
        # batch is a list of tuples
        # Train/Val: (src, tgt, cls)
        # Test: (src, raw_txt, id)

        src_tensors = [item[0] for item in batch]
        src_padded = pad_sequence(
            src_tensors, batch_first=True, padding_value=Config.PAD_IDX
        )

        if len(batch[0]) == 3 and isinstance(batch[0][1], torch.Tensor):
            # Train/Val mode
            tgt_tensors = [item[1] for item in batch]
            tgt_padded = pad_sequence(
                tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX
            )
            cls_tensors = torch.stack([item[2] for item in batch])
            return src_padded, tgt_padded, cls_tensors
        else:
            # Test mode
            raw_txts = [item[1] for item in batch]
            ids = [item[2] for item in batch]
            return src_padded, raw_txts, ids

    def get_train_loader(self, load_cached_data=True, debug=Config.DEBUG):
        df = self.process_data(
            Config.TRAIN_DATA_PATH,
            "train_processed",
            load_cached_data=load_cached_data,
            is_train_split=True,
        )

        if debug:
            df = df.head(Config.DEBUG_SIZE)

        if not self.tokenizer_fitted:
            self.fit_artifacts(df)

        dataset = NormalizationDataset(
            df, self.tokenizer, self.label_encoder, mode="train"
        )
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def get_val_loader(self, load_cached_data=True, debug=Config.DEBUG):
        df = self.process_data(
            Config.VAL_DATA_PATH,
            "val_processed",
            load_cached_data=load_cached_data,
            is_train_split=True,
        )

        if debug:
            df = df.head(Config.DEBUG_SIZE)

        # Ensure artifacts are loaded if not fitted in this session
        if not self.tokenizer_fitted or not self.encoder_fitted:
            self.load_artifacts()
            if not self.tokenizer_fitted:
                raise RuntimeError(
                    "Tokenizer not found. Run get_train_loader first to fit."
                )

        dataset = NormalizationDataset(
            df, self.tokenizer, self.label_encoder, mode="val"
        )
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def get_test_loader(self, load_cached_data=True):
        # Test data is NOT filtered by class (as we don't have classes)
        # We return the full test set with context
        df = self.process_data(
            Config.TEST_DATA_PATH,
            "test_processed",
            load_cached_data=load_cached_data,
            is_train_split=False,
        )

        # Ensure tokenizer is loaded
        if not self.tokenizer_fitted:
            self.load_artifacts()

        dataset = NormalizationDataset(
            df, self.tokenizer, self.label_encoder, mode="test"
        )
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
