import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any

from library.config import Config
from library.utils import get_logger
from library.vocab_manager import Vocab
from library.feature_engineering import FeatureEngineer

logger = get_logger("datasets")


class TaggerDataset(Dataset):
    """
    Dataset for the Gated Multi-Granularity Bi-LSTM Tagger.
    Prepares Word, BPE, Char-CNN, Regex, and Prior features.
    """

    def __init__(
        self,
        data_path: str,
        word_vocab: Vocab,
        char_vocab: Vocab,
        class_vocab: Vocab,
        bpe_tokenizer,
        feature_engineer: FeatureEngineer,
        priors_df: pd.DataFrame,
        split: str = "train",
        load_cached_data: bool = True,
        debug: bool = False,
        max_bpe_len: int = 16,
    ):
        """
        Args:
            data_path: Path to the CSV file (train, val, or test).
            word_vocab: Vocabulary for words.
            char_vocab: Vocabulary for characters.
            class_vocab: Vocabulary for classes.
            bpe_tokenizer: SentencePiece processor.
            feature_engineer: Instance of FeatureEngineer.
            priors_df: DataFrame containing global priors.
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to load from cache if available.
            debug: If True, uses a subset of data.
            max_bpe_len: Max length for BPE token sequence.
        """
        self.split = split
        self.max_bpe_len = max_bpe_len
        self.max_char_len = Config.MAX_CHAR_LEN

        # Define cache file path
        debug_suffix = "_debug" if debug else ""
        cache_filename = f"tagger_dataset_{split}{debug_suffix}.pt"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(self.cache_path):
            logger.info(
                f"Loading cached TaggerDataset ({split}) from {self.cache_path}..."
            )
            try:
                self.data = torch.load(self.cache_path)
                logger.info(
                    f"Successfully loaded {len(self.data['word_ids'])} samples."
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Process from Scratch
        logger.info(f"Processing TaggerDataset ({split}) from {data_path}...")
        self.data = self._process_data(
            data_path,
            word_vocab,
            char_vocab,
            class_vocab,
            bpe_tokenizer,
            feature_engineer,
            priors_df,
            debug,
        )

        # 3. Save to Cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        torch.save(self.data, self.cache_path)
        logger.info(f"Saved processed dataset to {self.cache_path}")

    def _process_data(
        self,
        data_path: str,
        word_vocab: Vocab,
        char_vocab: Vocab,
        class_vocab: Vocab,
        bpe,
        fe: FeatureEngineer,
        priors_df: pd.DataFrame,
        debug: bool,
    ) -> Dict[str, torch.Tensor]:

        # Load Data
        df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
        if debug:
            df = df.head(Config.DEBUG_SIZE)

        tokens = df["before"].astype(str).tolist()

        # --- Feature 1: Word IDs ---
        # Vectorized map is faster than loop
        # Handle OOV by checking if token is in vocab, else <unk>
        # Vocab class handles this in __getitem__ but for speed we might use apply
        # We'll use a list comprehension which is reasonably fast for 7M items with simple dict lookup
        word_ids = [word_vocab[t] for t in tokens]

        # --- Feature 2: BPE IDs ---
        # SentencePiece is implemented in C++, fast.
        # We need to pad/truncate to max_bpe_len
        bpe_ids_list = []
        pad_id = bpe.pad_id()
        for t in tokens:
            ids = bpe.encode_as_ids(t)
            if len(ids) > self.max_bpe_len:
                ids = ids[: self.max_bpe_len]
            else:
                ids = ids + [pad_id] * (self.max_bpe_len - len(ids))
            bpe_ids_list.append(ids)

        # --- Feature 3: Char IDs (for CNN) ---
        # Pad/truncate to max_char_len
        char_ids_list = []
        char_pad_id = char_vocab["<pad>"]
        for t in tokens:
            # Get indices
            ids = [char_vocab[c] for c in t]
            if len(ids) > self.max_char_len:
                ids = ids[: self.max_char_len]
            else:
                ids = ids + [char_pad_id] * (self.max_char_len - len(ids))
            char_ids_list.append(ids)

        # --- Feature 4: Regex Features ---
        regex_feats = fe.extract_regex_features(tokens)

        # --- Feature 5: Prior Vectors ---
        prior_feats = fe.get_priors_vector(tokens, priors_df)

        # --- Labels ---
        if "class" in df.columns:
            labels = [class_vocab[c] for c in df["class"].astype(str)]
        else:
            # For test set, use dummy labels (-1)
            labels = [-1] * len(tokens)

        # --- IDs (for submission) ---
        # Stored as list of strings, not tensor
        row_ids = df["id"].tolist() if "id" in df.columns else []

        # Convert to Tensors
        data = {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "bpe_ids": torch.tensor(bpe_ids_list, dtype=torch.long),
            "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
            "regex_feats": torch.tensor(regex_feats, dtype=torch.float32),
            "prior_feats": torch.tensor(prior_feats, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "row_ids": row_ids,  # Not a tensor
        }

        return data

    def __len__(self) -> int:
        return len(self.data["word_ids"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "word_ids": self.data["word_ids"][idx],
            "bpe_ids": self.data["bpe_ids"][idx],
            "char_ids": self.data["char_ids"][idx],
            "regex_feats": self.data["regex_feats"][idx],
            "prior_feats": self.data["prior_feats"][idx],
            "labels": self.data["labels"][idx],
            "row_id": self.data["row_ids"][idx] if self.data["row_ids"] else "",
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Character-Level LSTM Seq2Seq Fallback Model.
    Filters data for 'changed' tokens (before != after) and prepares
    source and target character sequences.
    """

    def __init__(
        self,
        data_path: str,
        char_vocab: Vocab,
        class_vocab: Vocab,
        split: str = "train",
        load_cached_data: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            data_path: Path to CSV.
            char_vocab: Character vocabulary (must include <sos>, <eos>).
            class_vocab: Class vocabulary.
            split: 'train' or 'val'.
            load_cached_data: Use cache if available.
            debug: Use subset.
        """
        self.split = split
        self.max_seq_len = Config.MAX_SEQ_LEN

        debug_suffix = "_debug" if debug else ""
        cache_filename = f"seq2seq_dataset_{split}{debug_suffix}.pt"
        self.cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        if load_cached_data and os.path.exists(self.cache_path):
            logger.info(
                f"Loading cached Seq2SeqDataset ({split}) from {self.cache_path}..."
            )
            try:
                self.data = torch.load(self.cache_path)
                logger.info(f"Successfully loaded {len(self.data['src_ids'])} samples.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        logger.info(f"Processing Seq2SeqDataset ({split}) from {data_path}...")
        self.data = self._process_data(data_path, char_vocab, class_vocab, debug)

        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        torch.save(self.data, self.cache_path)
        logger.info(f"Saved processed dataset to {self.cache_path}")

    def _process_data(
        self, data_path: str, char_vocab: Vocab, class_vocab: Vocab, debug: bool
    ) -> Dict[str, torch.Tensor]:

        df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
        if debug:
            df = df.head(Config.DEBUG_SIZE)

        # Filter for changed tokens only
        # We only train the fallback model on things that actually require normalization
        df = df[df["before"] != df["after"]].copy()

        src_tokens = df["before"].astype(str).tolist()
        tgt_tokens = df["after"].astype(str).tolist()
        classes = df["class"].astype(str).tolist()

        pad_id = char_vocab["<pad>"]
        sos_id = char_vocab["<sos>"]
        eos_id = char_vocab["<eos>"]

        src_ids_list = []
        tgt_ids_list = []
        class_ids_list = []

        for src, tgt, cls in zip(src_tokens, tgt_tokens, classes):
            # Encode Source (just chars + pad)
            s_ids = [char_vocab[c] for c in src]
            if len(s_ids) > self.max_seq_len:
                s_ids = s_ids[: self.max_seq_len]
            else:
                s_ids = s_ids + [pad_id] * (self.max_seq_len - len(s_ids))

            # Encode Target (<sos> + chars + <eos> + pad)
            # We truncate to max_seq_len - 1 to fit sos/eos
            t_ids = [char_vocab[c] for c in tgt]
            if len(t_ids) > (self.max_seq_len - 2):
                t_ids = t_ids[: (self.max_seq_len - 2)]

            t_ids = [sos_id] + t_ids + [eos_id]
            # Pad remainder
            if len(t_ids) < self.max_seq_len:
                t_ids = t_ids + [pad_id] * (self.max_seq_len - len(t_ids))

            src_ids_list.append(s_ids)
            tgt_ids_list.append(t_ids)
            class_ids_list.append(class_vocab[cls])

        return {
            "src_ids": torch.tensor(src_ids_list, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids_list, dtype=torch.long),
            "class_ids": torch.tensor(class_ids_list, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.data["src_ids"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.data["src_ids"][idx],
            self.data["class_ids"][idx],
            self.data["tgt_ids"][idx],
        )
