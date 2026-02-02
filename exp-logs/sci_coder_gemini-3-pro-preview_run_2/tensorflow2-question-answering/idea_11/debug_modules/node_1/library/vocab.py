import os
import json
import numpy as np
import collections
from library.config import Config


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # Initialize with special tokens
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)

    def add_token(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def lookup_index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def lookup_word(self, idx):
        if 0 <= idx < len(self.idx2word):
            return self.idx2word[idx]
        return self.unk_token

    @staticmethod
    def tokenize(text):
        """
        Simple whitespace tokenizer.
        """
        if not text:
            return []
        # Lowercase and split
        return text.lower().split()

    def text_to_indices(self, text, max_length):
        """
        Converts text to a list of indices with padding/truncation.
        """
        tokens = self.tokenize(text)
        indices = [self.lookup_index(token) for token in tokens]

        # Truncate
        if len(indices) > max_length:
            indices = indices[:max_length]

        # Pad
        if len(indices) < max_length:
            pad_idx = self.word2idx[self.pad_token]
            indices += [pad_idx] * (max_length - len(indices))

        return indices

    def build_from_corpus(self, file_path, sample_size=None):
        """
        Builds vocabulary from the provided JSONL file.
        """
        print(f"Building vocabulary from {file_path}...")
        counter = collections.Counter()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break

                    data = json.loads(line)
                    # Process document text
                    doc_text = data.get("document_text", "")
                    counter.update(self.tokenize(doc_text))

                    # Process question text
                    q_text = data.get("question_text", "")
                    counter.update(self.tokenize(q_text))

                    if i % 10000 == 0 and i > 0:
                        print(f"  Processed {i} lines...")

        except FileNotFoundError:
            print(f"Error: File {file_path} not found. Vocabulary might be empty.")
            return

        # Filter by frequency
        print(f"Total unique tokens found: {len(counter)}")

        # Sort by frequency (descending)
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # Add words to vocab respecting constraints
        added_count = 0
        for word, freq in sorted_words:
            if len(self) >= Config.MAX_VOCAB_SIZE:
                break
            if freq >= Config.MIN_FREQ:
                self.add_token(word)
                added_count += 1

        print(
            f"Vocabulary built. Size: {len(self)} (Added {added_count} words from corpus)"
        )

    def create_embedding_matrix(self):
        """
        Creates an embedding matrix.
        In a real scenario, this would load GloVe.
        Here, we initialize random embeddings.
        """
        print("Creating embedding matrix...")
        vocab_size = len(self)
        embed_dim = Config.EMBEDDING_DIM

        # Set seed for reproducibility
        np.random.seed(Config.SEED)

        # Initialize random embeddings (uniform distribution)
        # Scale by 1/sqrt(dim) to keep variance reasonable
        scale = 1.0 / np.sqrt(embed_dim)
        embedding_matrix = np.random.uniform(
            low=-scale, high=scale, size=(vocab_size, embed_dim)
        ).astype(np.float32)

        # Set PAD token embedding to zeros
        pad_idx = self.word2idx.get(self.pad_token, 0)
        embedding_matrix[pad_idx] = np.zeros((embed_dim,), dtype=np.float32)

        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        return embedding_matrix

    def save(self, vocab_path, embedding_path, embedding_matrix):
        """
        Saves vocabulary and embedding matrix to disk.
        Vocab is saved as a numpy array of strings to avoid pickle.
        """
        print(f"Saving vocabulary to {vocab_path}...")
        # Save idx2word list as numpy array of strings
        np.save(vocab_path, np.array(self.idx2word))

        print(f"Saving embedding matrix to {embedding_path}...")
        np.save(embedding_path, embedding_matrix)

    @classmethod
    def load(cls, vocab_path):
        """
        Loads vocabulary from disk.
        """
        print(f"Loading vocabulary from {vocab_path}...")
        vocab = cls()
        # Load numpy array of strings
        try:
            idx2word_arr = np.load(vocab_path, allow_pickle=False)
            # Reconstruct object state
            vocab.idx2word = idx2word_arr.tolist()
            vocab.word2idx = {word: idx for idx, word in enumerate(vocab.idx2word)}
            return vocab
        except Exception as e:
            print(f"Failed to load vocabulary: {e}")
            return None

    @classmethod
    def load_embeddings(cls, embedding_path):
        """
        Loads embedding matrix from disk.
        """
        print(f"Loading embedding matrix from {embedding_path}...")
        try:
            return np.load(embedding_path, allow_pickle=False)
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            return None

    @classmethod
    def load_or_build(cls, load_cached_data=True, sample_size=None):
        """
        Main entry point. Checks cache, loads if available, otherwise builds.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.
            sample_size (int, optional): Number of lines to process for debugging.

        Returns:
            tuple: (Vocabulary object, embedding_matrix numpy array)
        """
        # Ensure directories exist
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        vocab_path = Config.VOCAB_CACHE_PATH
        emb_path = Config.EMBEDDING_MATRIX_CACHE_PATH

        # 1. Try to load
        if load_cached_data:
            if os.path.exists(vocab_path) and os.path.exists(emb_path):
                print("Cache found. Loading vocabulary and embeddings...")
                vocab = cls.load(vocab_path)
                embedding_matrix = cls.load_embeddings(emb_path)

                if vocab is not None and embedding_matrix is not None:
                    # Quick consistency check
                    if len(vocab) == embedding_matrix.shape[0]:
                        return vocab, embedding_matrix
                    else:
                        print(
                            "Cache inconsistency (vocab size != embedding rows). Rebuilding..."
                        )
                else:
                    print("Cache loading failed. Rebuilding...")
            else:
                print("Cache missing. Rebuilding...")
        else:
            print("Force rebuild requested.")

        # 2. Build from scratch
        vocab = cls()
        # Use training data to build vocab
        vocab.build_from_corpus(Config.TRAIN_DATA_PATH, sample_size=sample_size)

        # Create embeddings
        embedding_matrix = vocab.create_embedding_matrix()

        # 3. Save to cache
        vocab.save(vocab_path, emb_path, embedding_matrix)

        return vocab, embedding_matrix
