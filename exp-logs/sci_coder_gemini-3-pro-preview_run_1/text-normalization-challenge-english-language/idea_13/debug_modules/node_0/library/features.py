import os
import re
import json
import pandas as pd
import numpy as np
import sentencepiece as spm
from library.config import Config


class RegexFeaturizer:
    """
    Generates explicit morphological features using regex patterns defined in Config.
    """

    def __init__(self):
        self.patterns = [re.compile(p) for p in Config.REGEX_PATTERNS]
        self.dim = len(self.patterns)

    def get_features(self, token):
        """
        Returns a binary list for a single token.
        """
        token_str = str(token)
        return [1.0 if p.search(token_str) else 0.0 for p in self.patterns]

    def transform(self, tokens):
        """
        Batch processing for tokens.
        Args:
            tokens: List of strings.
        Returns:
            np.ndarray: Shape (len(tokens), REGEX_DIM)
        """
        features = [self.get_features(t) for t in tokens]
        return np.array(features, dtype=np.float32)


class BPETokenizerWrapper:
    """
    Wrapper for SentencePiece BPE tokenizer.
    """

    def __init__(self):
        self.model_prefix = Config.BPE_MODEL_PREFIX
        self.model_path = f"{self.model_prefix}.model"
        self.vocab_size = Config.VOCAB_SIZE_BPE
        self.sp = spm.SentencePieceProcessor()

    def train(self, corpus_path):
        """
        Trains the BPE model on the provided corpus file.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.model_prefix), exist_ok=True)

        if os.path.exists(self.model_path):
            print(f"BPE model already exists at {self.model_path}. Loading...")
            self.load()
            return

        print(f"Training BPE model on {corpus_path}...")
        # Train sentencepiece
        # We use a high character coverage to include most characters
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=[],
        )
        # Load the trained model
        self.load()

    def load(self):
        """
        Loads the trained BPE model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BPE model not found at {self.model_path}")
        self.sp.load(self.model_path)

    def encode(self, tokens):
        """
        Encodes a list of tokens into subword IDs.
        """
        # sp.encode_as_ids accepts a string
        return [self.sp.encode_as_ids(str(t)) for t in tokens]

    def encode_as_padded_tensor(self, tokens, max_len=10):
        """
        Encodes and pads to fixed length.
        Returns np.ndarray of shape (batch_size, max_len)
        """
        ids_list = self.encode(tokens)
        batch_size = len(tokens)
        tensor = np.zeros((batch_size, max_len), dtype=np.int64)  # pad_id is 0

        for i, ids in enumerate(ids_list):
            length = min(len(ids), max_len)
            tensor[i, :length] = ids[:length]

        return tensor


class GlobalPriorMap:
    """
    Computes and stores the empirical probability distribution of classes for each token.
    """

    def __init__(self):
        self.prior_map = {}  # dict: token -> np.array (probs)
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.num_classes = 0
        self.zero_vector = None

    def build(self, train_csv_path, load_cached_data=True):
        """
        Builds the prior map from the training data or loads from cache.
        """
        # Check cache
        if (
            load_cached_data
            and os.path.exists(Config.PRIORS_PATH)
            and os.path.exists(Config.VOCAB_CLASSES_PATH)
        ):
            print("Loading Global Priors from cache...")
            self._load_cache()
            return

        print("Computing Global Priors from scratch...")
        # Load data
        df = pd.read_csv(
            train_csv_path,
            usecols=["before", "class"],
            dtype=str,
            keep_default_na=False,
        )

        # Build Class Vocabulary
        unique_classes = sorted(df["class"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        self.idx_to_class = {i: c for i, c in enumerate(unique_classes)}
        self.num_classes = len(unique_classes)

        # Save class vocab
        with open(Config.VOCAB_CLASSES_PATH, "w") as f:
            json.dump(self.class_to_idx, f)

        # Aggregate counts
        counts = df.groupby(["before", "class"]).size().reset_index(name="count")

        # Pivot: index=before, columns=class, values=count
        pivot = counts.pivot(index="before", columns="class", values="count").fillna(0)

        # Normalize to probabilities
        probs = pivot.div(pivot.sum(axis=1), axis=0)

        # Add missing classes as columns with 0 if any (though pivot handles present classes)
        # Ensure all vocab classes are present as columns
        for c in unique_classes:
            if c not in probs.columns:
                probs[c] = 0.0

        # Reorder columns to match vocab index
        probs = probs[unique_classes]

        # Save to parquet
        # Reset index to save 'before' as column
        probs_df = probs.reset_index()
        probs_df.to_parquet(Config.PRIORS_PATH, index=False)

        # Load into memory
        self._load_from_df(probs_df)

    def _load_cache(self):
        # Load class vocab
        with open(Config.VOCAB_CLASSES_PATH, "r") as f:
            self.class_to_idx = json.load(f)
        # Ensure keys are strings in json, values are ints
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Load priors
        df = pd.read_parquet(Config.PRIORS_PATH)
        self._load_from_df(df)

    def _load_from_df(self, df):
        # Ensure 'before' is index
        if "before" in df.columns:
            df.set_index("before", inplace=True)

        # Ensure columns match vocab order
        vocab_classes = [self.idx_to_class[i] for i in range(self.num_classes)]

        # Reindex to ensure order and fill missing
        df = df.reindex(columns=vocab_classes, fill_value=0.0)

        # Convert to dictionary of numpy arrays
        self.prior_map = {}
        for token, row in df.iterrows():
            self.prior_map[token] = row.values.astype(np.float32)

        # Initialize zero vector
        self.zero_vector = np.zeros(self.num_classes, dtype=np.float32)

    def get_priors(self, tokens):
        """
        Returns prior vectors for a list of tokens.
        Args:
            tokens: List of strings.
        Returns:
            np.ndarray: Shape (len(tokens), PRIOR_DIM)
        """
        # Retrieve vectors
        vectors = [self.prior_map.get(str(t), self.zero_vector) for t in tokens]
        arr = np.array(vectors, dtype=np.float32)

        # Handle Dimension Mismatch with Config.PRIOR_DIM
        target_dim = Config.PRIOR_DIM
        current_dim = arr.shape[1]

        if current_dim < target_dim:
            # Pad with zeros
            pad_width = target_dim - current_dim
            arr = np.pad(arr, ((0, 0), (0, pad_width)), "constant")
        elif current_dim > target_dim:
            # Truncate (Warning: losing info)
            arr = arr[:, :target_dim]

        return arr


def prepare_bpe_training_data(train_csv_path, output_path):
    """
    Extracts unique tokens from training data to a text file for BPE training.
    """
    if os.path.exists(output_path):
        return

    print(f"Extracting tokens for BPE training from {train_csv_path}...")
    df = pd.read_csv(
        train_csv_path, usecols=["before"], dtype=str, keep_default_na=False
    )
    unique_tokens = df["before"].unique()

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for t in unique_tokens:
            f.write(str(t) + "\n")
    print(f"Saved {len(unique_tokens)} unique tokens to {output_path}")
