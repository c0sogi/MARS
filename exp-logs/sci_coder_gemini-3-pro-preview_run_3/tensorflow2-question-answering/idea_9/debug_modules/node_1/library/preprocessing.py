import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from library.config import Config


class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = Config.VOCAB_SIZE
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN
        self.pad_idx = 0
        self.unk_idx = 1

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def tokenize(self, text):
        """
        Splits text on whitespace.
        """
        if not text:
            return []
        return text.split()

    def extract_candidates_from_json(self, data_json):
        """
        Parses the raw document text into discrete candidate paragraphs based on
        the long_answer_candidates field provided in the NQ dataset.

        Args:
            data_json (dict): A dictionary representing a single NQ sample.

        Returns:
            list of str: A list of candidate paragraph strings.
        """
        doc_text = data_json.get("document_text", "")
        candidates_info = data_json.get("long_answer_candidates", [])

        tokens = self.tokenize(doc_text)
        candidates = []

        for cand in candidates_info:
            # We typically only care about top-level candidates for ranking
            if cand.get("top_level", False):
                start = cand["start_token"]
                end = cand["end_token"]

                # Bounds check
                if start < 0 or end > len(tokens) or start >= end:
                    continue

                span_tokens = tokens[start:end]
                candidates.append(" ".join(span_tokens))

        return candidates

    def build_vocabulary(self, load_cached_data=True):
        """
        Builds or loads the vocabulary.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            dict: The vocabulary mapping (token -> index).
        """
        cache_path = Config.VOCAB_CACHE

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading vocabulary from {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                self.vocab = dict(zip(df["token"], df["index"]))
                return self.vocab
            except Exception as e:
                print(f"Failed to load vocabulary cache: {e}. Rebuilding...")

        # 2. Compute from scratch
        print("Building vocabulary from training data...")

        # Load metadata to access training samples efficiently
        if not os.path.exists(Config.TRAIN_METADATA):
            raise FileNotFoundError(
                f"Train metadata not found at {Config.TRAIN_METADATA}"
            )

        metadata_df = pd.read_csv(Config.TRAIN_METADATA)

        # Sample if configured (for debugging/speed)
        if Config.SAMPLE_SIZE is not None and Config.SAMPLE_SIZE < len(metadata_df):
            metadata_df = metadata_df.sample(
                n=Config.SAMPLE_SIZE, random_state=Config.SEED
            )

        token_counter = Counter()

        with open(Config.TRAIN_FILE, "rb") as f:
            for _, row in metadata_df.iterrows():
                offset = row["byte_offset"]
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))

                    # Add question tokens
                    q_text = data.get("question_text", "")
                    token_counter.update(self.tokenize(q_text))

                    # Add document tokens (sampling to save time/memory if doc is huge)
                    # We use the document text directly.
                    # In a full run, we might want to limit this, but here we process it.
                    doc_text = data.get("document_text", "")
                    doc_tokens = self.tokenize(doc_text)
                    token_counter.update(doc_tokens)

                except json.JSONDecodeError:
                    continue

        # Create vocabulary
        # Start with special tokens
        vocab_dict = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}

        # Add most common tokens up to VOCAB_SIZE - 2
        most_common = token_counter.most_common(self.vocab_size - 2)
        for i, (token, _) in enumerate(most_common):
            vocab_dict[token] = i + 2

        self.vocab = vocab_dict

        # 3. Save to cache
        print(f"Saving vocabulary to {cache_path}")
        df_vocab = pd.DataFrame(list(self.vocab.items()), columns=["token", "index"])
        df_vocab.to_parquet(cache_path, index=False)

        return self.vocab

    def load_embeddings(self, load_cached_data=True, pretrained_path=None):
        """
        Loads or initializes the embedding matrix.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
            pretrained_path (str, optional): Path to a GloVe-format text file.
                                             If None or file not found, initializes randomly.

        Returns:
            numpy.ndarray: Embedding matrix of shape (VOCAB_SIZE, EMBED_DIM).
        """
        cache_path = Config.EMBEDDING_MATRIX_CACHE

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading embedding matrix from {cache_path}")
            try:
                matrix = np.load(cache_path)
                if matrix.shape == (self.vocab_size, Config.EMBED_DIM):
                    return matrix
                else:
                    print(f"Cached matrix shape {matrix.shape} mismatch. Rebuilding...")
            except Exception as e:
                print(f"Failed to load embedding cache: {e}. Rebuilding...")

        # 2. Compute/Initialize from scratch
        print("Initializing embedding matrix...")

        # Initialize with random normal distribution
        embedding_matrix = np.random.normal(
            scale=0.6, size=(self.vocab_size, Config.EMBED_DIM)
        )

        # Zero out padding index
        if self.pad_token in self.vocab:
            embedding_matrix[self.vocab[self.pad_token]] = np.zeros(Config.EMBED_DIM)

        # If a pretrained file is provided and exists, load it
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained embeddings from {pretrained_path}...")
            hits = 0
            with open(pretrained_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip().split(" ")
                    word = parts[0]
                    if word in self.vocab:
                        vector = np.array(parts[1:], dtype=np.float32)
                        if vector.shape[0] == Config.EMBED_DIM:
                            embedding_matrix[self.vocab[word]] = vector
                            hits += 1
            print(f"Loaded {hits} vectors from pretrained file.")
        else:
            print(
                "No pretrained embedding path provided or file not found. Using random initialization."
            )

        # 3. Save to cache
        print(f"Saving embedding matrix to {cache_path}")
        np.save(cache_path, embedding_matrix)

        return embedding_matrix.astype(np.float32)

    def text_to_indices(self, text, max_len=None):
        """
        Converts a text string to a padded tensor of vocabulary indices.

        Args:
            text (str): The input text string.
            max_len (int, optional): The maximum length for truncation/padding.

        Returns:
            torch.LongTensor: Tensor of shape (max_len,) containing indices.
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")

        tokens = self.tokenize(text)
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]

        if max_len is None:
            # If no max_len provided, just return the list as tensor
            return torch.tensor(indices, dtype=torch.long)

        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long)
