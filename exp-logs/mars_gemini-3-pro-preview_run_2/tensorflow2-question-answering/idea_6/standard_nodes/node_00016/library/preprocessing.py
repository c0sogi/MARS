import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config
from library.utils import load_jsonl


class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN
        self.pad_id = 0
        self.unk_id = 1
        # Use parquet for vocab storage to ensure no pickle usage
        self.vocab_cache_path = Config.VOCAB_CACHE_FILE.replace(".npy", ".parquet")

    def fit(self, file_path, sample_size=0, load_cached_data=True):
        """
        Fits the tokenizer on the provided JSONL file.
        Strict caching logic:
        1. If load_cached_data is True, try to load vocab from cache.
        2. If load fails or load_cached_data is False, compute from scratch and save.
        """
        if load_cached_data and os.path.exists(self.vocab_cache_path):
            try:
                print(f"Loading vocabulary from {self.vocab_cache_path}...")
                df = pd.read_parquet(self.vocab_cache_path)
                vocab_list = df["token"].tolist()
                self.id_to_token = {i: token for i, token in enumerate(vocab_list)}
                self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
                self.vocab_size = len(vocab_list)
                # Verify special tokens
                self.pad_id = self.token_to_id.get(self.pad_token, 0)
                self.unk_id = self.token_to_id.get(self.unk_token, 1)
                return
            except Exception as e:
                print(f"Failed to load vocabulary: {e}. Recomputing...")

        print("Computing vocabulary from scratch...")
        counter = Counter()

        # Determine limit: explicit sample_size takes precedence, else debug limit from config
        limit = sample_size if sample_size > 0 else Config.DEBUG_SAMPLE_SIZE

        # Iterate through the file
        for entry in load_jsonl(file_path, limit):
            # Tokenize document text
            doc_text = entry.get("document_text", "")
            if doc_text:
                counter.update(doc_text.split())

            # Tokenize question text
            q_text = entry.get("question_text", "")
            if q_text:
                counter.update(q_text.split())

        # Filter by min frequency and size
        # Start with special tokens
        vocab_list = [self.pad_token, self.unk_token]

        # Most common tokens
        # We subtract 2 for PAD and UNK
        max_tokens = Config.VOCAB_SIZE - 2

        for token, count in counter.most_common(max_tokens):
            if count >= Config.MIN_FREQ:
                vocab_list.append(token)
            else:
                break

        # Build maps
        self.id_to_token = {i: token for i, token in enumerate(vocab_list)}
        self.token_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

        # Save to cache
        os.makedirs(os.path.dirname(self.vocab_cache_path), exist_ok=True)
        pd.DataFrame({"token": vocab_list}).to_parquet(
            self.vocab_cache_path, index=False
        )
        print(f"Vocabulary saved to {self.vocab_cache_path}. Size: {self.vocab_size}")

    def text_to_sequence(self, text):
        """Converts a string to a list of integer IDs."""
        if not text:
            return []
        tokens = text.split()
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def pad_sequence(self, sequence, max_len):
        """Pads or truncates a sequence to max_len."""
        if len(sequence) >= max_len:
            return sequence[:max_len]
        else:
            return sequence + [self.pad_id] * (max_len - len(sequence))

    def encode(self, text, max_len):
        """Combined conversion and padding."""
        seq = self.text_to_sequence(text)
        return self.pad_sequence(seq, max_len)


def build_embedding_matrix(tokenizer, load_cached_data=True):
    """
    Creates or loads the embedding matrix.
    Strict caching logic applied.
    """
    cache_path = Config.EMBED_MATRIX_CACHE_FILE

    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading embedding matrix from {cache_path}...")
            embedding_matrix = np.load(cache_path)
            if embedding_matrix.shape == (tokenizer.vocab_size, Config.EMBED_DIM):
                return embedding_matrix
            else:
                print(
                    f"Cached embedding shape {embedding_matrix.shape} mismatch with vocab {tokenizer.vocab_size}. Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load embedding matrix: {e}. Recomputing...")

    print("Creating embedding matrix from scratch...")

    # Initialize random embeddings
    # Using uniform distribution for initialization
    embedding_matrix = np.random.uniform(
        -0.1, 0.1, (tokenizer.vocab_size, Config.EMBED_DIM)
    ).astype(np.float32)

    # Zero out the PAD token
    if tokenizer.pad_id < tokenizer.vocab_size:
        embedding_matrix[tokenizer.pad_id] = np.zeros(Config.EMBED_DIM)

    # Logic for pretrained embeddings (Placeholder)
    if Config.USE_PRETRAINED:
        # In a full implementation, this would load GloVe/Word2Vec
        pass

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embedding_matrix)
    print(f"Embedding matrix saved to {cache_path}. Shape: {embedding_matrix.shape}")

    return embedding_matrix
