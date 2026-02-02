import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config


class DataProcessor:
    """
    Handles text preprocessing, vocabulary construction, and embedding matrix generation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.vocab = {}
        self.idx2word = []

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def tokenize(self, text: str):
        """
        Splits text into tokens based on whitespace.

        Args:
            text (str): Input text.

        Returns:
            list: List of tokens.
        """
        if not text:
            return []
        return text.split()

    def build_vocab(self, load_cached_data: bool = True):
        """
        Builds or loads the vocabulary from the training data.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            dict: Mapping of token to integer index.
        """
        vocab_path = self.config.VOCAB_PATH

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}...")
            try:
                # Load the array of words
                self.idx2word = np.load(vocab_path)
                # Reconstruct the dictionary
                self.vocab = {word: idx for idx, word in enumerate(self.idx2word)}
                print(f"Vocabulary loaded. Size: {len(self.vocab)}")
                return self.vocab
            except Exception as e:
                print(f"Failed to load vocabulary: {e}. Rebuilding...")

        # 2. Compute from scratch
        print("Building vocabulary from training data...")

        # We use the raw training file.
        # Metadata is useful for splitting, but for vocab we can use the whole train file
        # to ensure good coverage, or strictly use the train subset defined in metadata.
        # To be safe and robust, we'll use the IDs from train_metadata to filter.

        train_meta = pd.read_csv(self.config.TRAIN_META_PATH)
        train_ids = set(train_meta["example_id"].astype(str))

        token_counter = Counter()

        # Read JSONL in chunks
        chunksize = 10000
        reader = pd.read_json(
            self.config.TRAIN_DATA_PATH, lines=True, chunksize=chunksize
        )

        processed_count = 0
        for chunk in reader:
            # Filter for training examples only
            chunk["example_id"] = chunk["example_id"].astype(str)
            train_chunk = chunk[chunk["example_id"].isin(train_ids)]

            if train_chunk.empty:
                continue

            # Aggregate text from document and questions
            # Note: Document text is long, we might want to limit vocab building to questions
            # and a subset of doc text, or just process everything.
            # Given constraints, we process everything but efficiently.

            for _, row in train_chunk.iterrows():
                # Tokenize question
                q_tokens = self.tokenize(row["question_text"])
                token_counter.update(q_tokens)

                # Tokenize document (this can be heavy, but necessary for coverage)
                d_tokens = self.tokenize(row["document_text"])
                token_counter.update(d_tokens)

            processed_count += len(train_chunk)
            if (
                self.config.DEBUG_SAMPLE_SIZE
                and processed_count >= self.config.DEBUG_SAMPLE_SIZE
            ):
                print(f"Debug limit reached: {processed_count} samples.")
                break

        # 3. Create Vocab Mappings
        # Start with special tokens
        self.idx2word = [self.config.PAD_TOKEN, self.config.UNK_TOKEN]

        # Select most common words up to VOCAB_SIZE - 2
        most_common = token_counter.most_common(self.config.VOCAB_SIZE - 2)
        self.idx2word.extend([word for word, count in most_common])

        # Create dictionary
        self.vocab = {word: idx for idx, word in enumerate(self.idx2word)}

        # 4. Save to cache
        # We save as a numpy array of strings to avoid pickle
        print(f"Saving vocabulary to {vocab_path}...")
        np.save(vocab_path, np.array(self.idx2word))

        print(f"Vocabulary built. Size: {len(self.vocab)}")
        return self.vocab

    def create_embedding_matrix(self, load_cached_data: bool = True):
        """
        Creates or loads the embedding matrix.
        Since no external GloVe file is provided in the input, we initialize random embeddings.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embed_dim).
        """
        if self.vocab is None:
            raise ValueError(
                "Vocabulary must be built before creating embedding matrix."
            )

        embed_path = self.config.EMBEDDING_MATRIX_PATH
        vocab_size = len(self.vocab)
        embed_dim = self.config.EMBED_DIM

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(embed_path):
            print(f"Loading embedding matrix from {embed_path}...")
            try:
                matrix = np.load(embed_path)
                if matrix.shape == (vocab_size, embed_dim):
                    self.embedding_matrix = matrix
                    return matrix
                else:
                    print(
                        f"Cached matrix shape {matrix.shape} mismatch with config {(vocab_size, embed_dim)}. Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load embedding matrix: {e}. Recomputing...")

        # 2. Compute (Initialize) from scratch
        print("Initializing random embedding matrix...")

        # Set seed for reproducibility
        np.random.seed(self.config.SEED)

        # Random initialization (uniform distribution)
        # Typically embeddings are initialized with small values, e.g., [-0.1, 0.1] or Xavier
        scale = 1.0 / np.sqrt(embed_dim)
        matrix = np.random.uniform(-scale, scale, (vocab_size, embed_dim)).astype(
            np.float32
        )

        # Set PAD token to zero vector
        if self.config.PAD_TOKEN in self.vocab:
            pad_idx = self.vocab[self.config.PAD_TOKEN]
            matrix[pad_idx] = np.zeros(embed_dim)

        self.embedding_matrix = matrix

        # 3. Save to cache
        print(f"Saving embedding matrix to {embed_path}...")
        np.save(embed_path, matrix)

        return matrix

    def text_to_indices(self, text: str, max_len: int):
        """
        Converts a text string to a list of integer indices based on the vocab.
        Truncates or pads to max_len.

        Args:
            text (str): Input text.
            max_len (int): Maximum sequence length.

        Returns:
            list: List of integers.
        """
        tokens = self.tokenize(text)
        unk_idx = self.vocab.get(self.config.UNK_TOKEN, 1)
        pad_idx = self.vocab.get(self.config.PAD_TOKEN, 0)

        # Map to indices
        indices = [self.vocab.get(t, unk_idx) for t in tokens]

        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]

        # Pad
        if len(indices) < max_len:
            indices += [pad_idx] * (max_len - len(indices))

        return indices
