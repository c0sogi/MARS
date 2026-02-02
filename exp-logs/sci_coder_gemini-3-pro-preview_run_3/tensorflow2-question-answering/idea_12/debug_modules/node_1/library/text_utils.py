import os
import json
import pandas as pd
import numpy as np
import collections
import torch
from library.config import Config


class Tokenizer:
    """
    Simple whitespace tokenizer for the Simplified NQ dataset.
    The dataset description states: "The text can be tokenized by splitting on whitespace."
    """

    @staticmethod
    def tokenize(text):
        """Splits text on whitespace."""
        if not text:
            return []
        return text.split()


def parse_candidates(document_text, candidates_json, max_candidates=None):
    """
    Extracts candidate paragraphs from the document text using the provided
    candidate spans (start_token, end_token).

    Args:
        document_text (str): The raw document text (space separated tokens).
        candidates_json (list): List of dicts with 'start_token' and 'end_token'.
        max_candidates (int): Maximum number of candidates to return.

    Returns:
        list: A list of dictionaries containing 'text', 'start_token', 'end_token', 'tokens'.
    """
    tokens = Tokenizer.tokenize(document_text)
    parsed_candidates = []

    if not candidates_json:
        return []

    count = 0
    for cand in candidates_json:
        # In NQ, 'top_level' indicates a candidate that is not contained within another candidate.
        # We typically prioritize these for ranking.
        if not cand.get("top_level", False):
            continue

        start = cand["start_token"]
        end = cand["end_token"]

        # Boundary checks
        if start < 0 or end > len(tokens) or start >= end:
            continue

        cand_tokens = tokens[start:end]
        cand_text = " ".join(cand_tokens)

        parsed_candidates.append(
            {
                "text": cand_text,
                "start_token": start,
                "end_token": end,
                "tokens": cand_tokens,
            }
        )

        count += 1
        if max_candidates is not None and count >= max_candidates:
            break

    return parsed_candidates


class Vocab:
    """
    Vocabulary management class. Handles token-to-id mapping and embedding matrix.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.embedding_matrix = None
        self.vocab_size = 0

    def build(self, texts, max_vocab_size, embedding_dim=100, glove_path=None):
        """
        Builds vocabulary from a list of text strings.

        Args:
            texts (list): List of strings to build vocab from.
            max_vocab_size (int): Maximum size of vocabulary.
            embedding_dim (int): Dimension of embedding vectors.
            glove_path (str): Path to pre-trained GloVe file (optional).
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        counter = collections.Counter()
        for text in texts:
            tokens = Tokenizer.tokenize(text)
            counter.update(tokens)

        # Start with special tokens
        self.token_to_id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.id_to_token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

        # Add most common tokens
        # max_vocab_size includes special tokens
        num_tokens_to_add = max_vocab_size - 2
        most_common = counter.most_common(num_tokens_to_add)

        for token, _ in most_common:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.vocab_size = len(self.token_to_id)
        print(f"Vocabulary built with {self.vocab_size} tokens.")

        # Initialize embeddings
        self._init_embeddings(embedding_dim, glove_path)

    def _init_embeddings(self, embedding_dim, glove_path):
        """
        Initializes embedding matrix. Loads GloVe if available, else random.
        """
        # Random initialization (normal distribution)
        np.random.seed(Config.SEED)
        self.embedding_matrix = np.random.normal(
            scale=0.6, size=(self.vocab_size, embedding_dim)
        )
        # PAD token should be zero vector
        self.embedding_matrix[0] = np.zeros(embedding_dim)

        if glove_path and os.path.exists(glove_path):
            print(f"Loading GloVe embeddings from {glove_path}...")
            hits = 0
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    if word in self.token_to_id:
                        try:
                            vector = np.array(parts[1:], dtype=np.float32)
                            if vector.shape[0] == embedding_dim:
                                idx = self.token_to_id[word]
                                self.embedding_matrix[idx] = vector
                                hits += 1
                        except ValueError:
                            continue
            print(f"Loaded {hits} vectors from GloVe.")
        else:
            print(
                "No pre-trained embeddings found or provided. Using random initialization."
            )

    def save(self, vocab_path, emb_path):
        """Saves vocabulary map (parquet) and embedding matrix (npy)."""
        # Save token_to_id as dataframe
        data = [{"token": k, "id": v} for k, v in self.token_to_id.items()]
        df = pd.DataFrame(data)
        df.to_parquet(vocab_path, index=False)

        # Save matrix
        np.save(emb_path, self.embedding_matrix)
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Embeddings saved to {emb_path}")

    def load(self, vocab_path, emb_path):
        """Loads vocabulary map and embedding matrix."""
        if not os.path.exists(vocab_path) or not os.path.exists(emb_path):
            raise FileNotFoundError("Vocab or Embedding file not found.")

        df = pd.read_parquet(vocab_path)
        self.token_to_id = dict(zip(df["token"], df["id"]))
        self.id_to_token = dict(zip(df["id"], df["token"]))

        self.embedding_matrix = np.load(emb_path)
        self.vocab_size = len(self.token_to_id)
        print(f"Loaded vocabulary of size {self.vocab_size}")

    def text_to_ids(self, text, max_len=None):
        """
        Converts text string to list of IDs. Truncates or pads if max_len provided.
        """
        tokens = Tokenizer.tokenize(text)
        ids = [
            self.token_to_id.get(t, self.token_to_id[self.UNK_TOKEN]) for t in tokens
        ]

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                # Pad with 0 (PAD_TOKEN)
                ids = ids + [0] * (max_len - len(ids))

        return ids


def build_or_load_vocab(metadata_df, load_cached_data=True):
    """
    Builds or loads the vocabulary based on caching logic.

    Args:
        metadata_df (pd.DataFrame): Train metadata containing offsets to read text.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Vocab: Initialized Vocab object.
    """
    Config.setup_directories()
    vocab = Vocab()

    vocab_path = Config.VOCAB_PATH
    emb_path = Config.EMBEDDING_MATRIX_PATH

    # 1. Try loading
    if load_cached_data:
        try:
            vocab.load(vocab_path, emb_path)
            return vocab
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Cached vocab not found or corrupt ({e}). Rebuilding...")

    # 2. Rebuild
    # Read a sample of text to build vocab (to avoid reading full dataset if huge)
    sample_size = Config.TRAIN_SAMPLE_SIZE if Config.TRAIN_SAMPLE_SIZE else 50000

    print(f"Reading up to {sample_size} samples to build vocabulary...")

    # Sample metadata if necessary
    if len(metadata_df) > sample_size:
        sample_meta = metadata_df.sample(n=sample_size, random_state=Config.SEED)
    else:
        sample_meta = metadata_df

    texts = []

    with open(Config.TRAIN_RAW_FILE, "rb") as f:
        for _, row in sample_meta.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                # Add question
                q_text = data.get("question_text", "")
                if q_text:
                    texts.append(q_text)

                # Add document text (truncated to capture common words without OOM)
                doc_text = data.get("document_text", "")
                if doc_text:
                    # Take first 1000 tokens of document
                    tokens = doc_text.split()
                    texts.append(" ".join(tokens[:1000]))
            except json.JSONDecodeError:
                continue

    vocab.build(
        texts=texts,
        max_vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        glove_path=None,  # No external file provided in input directory
    )

    vocab.save(vocab_path, emb_path)
    return vocab
