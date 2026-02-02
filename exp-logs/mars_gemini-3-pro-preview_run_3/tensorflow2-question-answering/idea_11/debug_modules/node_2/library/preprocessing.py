import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config


class HTMLParser:
    """
    Segments raw document text into candidate paragraphs based on HTML tags.
    """

    def __init__(self, tags=Config.CANDIDATE_TAGS):
        self.tags = set(tags)

    def segment(self, document_text):
        """
        Splits document_text into candidate paragraphs based on top-level HTML tags.

        Args:
            document_text (str): The raw document text with HTML tags.

        Returns:
            list of dict: Each dict contains 'text', 'start_token', 'end_token'.
        """
        tokens = document_text.split()
        candidates = []
        current_start = -1

        # Heuristic: A candidate starts at a tag in self.tags and ends at the next tag in self.tags
        # or the end of the document.
        for i, token in enumerate(tokens):
            if token in self.tags:
                if current_start != -1:
                    # Close previous candidate
                    candidates.append(
                        {
                            "text": " ".join(tokens[current_start:i]),
                            "start_token": current_start,
                            "end_token": i,
                        }
                    )
                current_start = i

        # Append the last candidate
        if current_start != -1:
            candidates.append(
                {
                    "text": " ".join(tokens[current_start:]),
                    "start_token": current_start,
                    "end_token": len(tokens),
                }
            )

        # Fallback: If no tags found, treat whole doc as one candidate
        if not candidates and tokens:
            candidates.append(
                {"text": document_text, "start_token": 0, "end_token": len(tokens)}
            )

        return candidates


class Tokenizer:
    """
    Tokenizes text and maps tokens to vocabulary indices.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        unk_token=Config.UNK_TOKEN,
        pad_token=Config.PAD_TOKEN,
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.token_to_id = {pad_token: 0, unk_token: 1}
        self.id_to_token = {0: pad_token, 1: unk_token}
        self.is_fitted = False

    def fit_on_texts(self, texts_generator):
        """
        Builds vocabulary from a generator of texts.
        """
        counter = Counter()
        for text in texts_generator:
            tokens = text.split()
            counter.update(tokens)

        # Select top words. 0 and 1 are reserved.
        most_common = counter.most_common(self.vocab_size - 2)

        for i, (token, _) in enumerate(most_common):
            idx = i + 2
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.is_fitted = True

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts to a list of lists of indices.
        """
        sequences = []
        unk_id = self.token_to_id[self.unk_token]
        for text in texts:
            tokens = text.split()
            seq = [self.token_to_id.get(t, unk_id) for t in tokens]
            sequences.append(seq)
        return sequences

    def save(self, path):
        """
        Saves vocabulary to a parquet file.
        """
        data = []
        for token, idx in self.token_to_id.items():
            data.append({"token": token, "id": idx})
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        """
        Loads vocabulary from a parquet file.
        """
        df = pd.read_parquet(path)
        self.token_to_id = dict(zip(df["token"], df["id"]))
        self.id_to_token = dict(zip(df["id"], df["token"]))
        self.is_fitted = True


class EmbeddingLoader:
    """
    Loads pre-trained word embeddings or initializes random ones.
    """

    def __init__(self, embedding_dim=Config.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim

    def create_matrix(self, tokenizer, source_file=None):
        """
        Creates an embedding matrix for the tokenizer's vocabulary.
        If source_file is provided and exists, loads GloVe vectors.
        Otherwise, initializes randomly.
        """
        vocab_size = len(tokenizer.token_to_id)
        # Initialize with random normal distribution
        embedding_matrix = np.random.normal(
            scale=0.6, size=(vocab_size, self.embedding_dim)
        )

        # Zero out PAD token
        if tokenizer.pad_token in tokenizer.token_to_id:
            pad_idx = tokenizer.token_to_id[tokenizer.pad_token]
            embedding_matrix[pad_idx] = np.zeros(self.embedding_dim)

        if source_file and os.path.exists(source_file):
            print(f"Loading embeddings from {source_file}...")
            hits = 0
            with open(source_file, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in tokenizer.token_to_id:
                        idx = tokenizer.token_to_id[word]
                        try:
                            vector = np.asarray(values[1:], dtype="float32")
                            if vector.shape[0] == self.embedding_dim:
                                embedding_matrix[idx] = vector
                                hits += 1
                        except ValueError:
                            continue
            print(f"Loaded {hits} vectors out of {vocab_size} vocab size.")

        return embedding_matrix.astype(np.float32)


def get_tokenizer(load_cached_data=True):
    """
    Returns a fitted Tokenizer.
    Loads from cache if available and requested.
    Otherwise builds from training data and caches it.
    """
    tokenizer = Tokenizer()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.VOCAB_PATH):
        print(f"Loading vocab from {Config.VOCAB_PATH}")
        tokenizer.load(Config.VOCAB_PATH)
    else:
        print("Building vocab from training data...")
        # Load metadata to locate training data
        if not os.path.exists(Config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata not found at {Config.TRAIN_METADATA_PATH}"
            )

        meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

        # If debugging, use subset
        if Config.TRAIN_SUBSET_SIZE:
            meta_df = meta_df.head(Config.TRAIN_SUBSET_SIZE)

        def text_generator():
            with open(Config.TRAIN_DATA_FILE, "rb") as f:
                for _, row in meta_df.iterrows():
                    offset = row["byte_offset"]
                    f.seek(offset)
                    line = f.readline()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        # Add question
                        yield data.get("question_text", "")
                        # Add document text
                        yield data.get("document_text", "")
                    except json.JSONDecodeError:
                        continue

        tokenizer.fit_on_texts(text_generator())
        tokenizer.save(Config.VOCAB_PATH)
        print(f"Vocab saved to {Config.VOCAB_PATH}")

    return tokenizer


def get_embedding_matrix(tokenizer, load_cached_data=True):
    """
    Returns the embedding matrix.
    Loads from cache if available and requested.
    Otherwise creates it and caches it.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.EMBEDDING_MATRIX_PATH):
        print(f"Loading embedding matrix from {Config.EMBEDDING_MATRIX_PATH}")
        matrix = np.load(Config.EMBEDDING_MATRIX_PATH)
        # Check consistency
        if matrix.shape[0] != len(tokenizer.token_to_id):
            print("Cached matrix shape mismatch with tokenizer. Recomputing...")
        else:
            return matrix

    print("Creating embedding matrix...")
    loader = EmbeddingLoader()
    # No source file provided in this environment, so it will initialize randomly
    matrix = loader.create_matrix(tokenizer, source_file=None)

    np.save(Config.EMBEDDING_MATRIX_PATH, matrix)
    print(f"Embedding matrix saved to {Config.EMBEDDING_MATRIX_PATH}")

    return matrix
