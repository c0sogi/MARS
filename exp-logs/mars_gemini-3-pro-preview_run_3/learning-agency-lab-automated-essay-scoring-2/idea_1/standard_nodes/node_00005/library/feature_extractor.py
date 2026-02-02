import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeatureEngineer:
    """
    Extracts structural meta-features from text to augment dense embeddings.
    Cite solution_lesson_node_00004
    """

    def extract_features(self, texts):
        """
        Extracts a set of structural features for each text.

        Features:
        1. Word Count
        2. Character Count
        3. Sentence Count
        4. Paragraph Count
        5. Average Word Length
        6. Average Sentence Length
        7. Unique Word Count
        8. Lexical Diversity (Unique / Total)
        """
        features = []
        for text in texts:
            # Clean and tokenize
            # Simple whitespace tokenization is sufficient for meta-features
            words = text.split()
            word_count = len(words)
            char_count = len(text)

            # Sentence count (split by punctuation)
            sentences = re.split(r"[.!?]+", text)
            sentence_count = len([s for s in sentences if s.strip()])

            # Paragraph count
            paragraph_count = text.count("\n") + 1

            # Lexical stats
            unique_words = set(w.lower() for w in words)
            unique_count = len(unique_words)

            # Derived ratios
            lexical_diversity = unique_count / word_count if word_count > 0 else 0
            avg_word_len = char_count / word_count if word_count > 0 else 0
            avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

            features.append(
                [
                    word_count,
                    char_count,
                    sentence_count,
                    paragraph_count,
                    avg_word_len,
                    avg_sentence_len,
                    unique_count,
                    lexical_diversity,
                ]
            )

        return np.array(features)


class EmbeddingEngine:
    """
    Handles the conversion of text data into dense vector embeddings using
    pre-trained Sentence Transformer models.
    """

    def __init__(self):
        """
        Initializes the SentenceTransformer model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing EmbeddingEngine on device: {self.device}")

        # Load the pre-trained model
        # We use the model name defined in the configuration
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

        # Set the maximum sequence length
        self.model.max_seq_length = Config.MAX_LENGTH

    def generate_embeddings(self, texts, data_type, load_cached_data=True):
        """
        Generates embeddings for a list of texts, with caching support.

        Args:
            texts (list of str): The list of essay texts to encode.
            data_type (str): The type of data ('train', 'val', 'test') for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: A numpy array of shape (n_samples, embedding_dim).
        """
        # Ensure the cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Construct cache filename
        # We append '_debug' if running in debug mode to avoid polluting the main cache
        suffix = "_debug" if Config.DEBUG else ""
        filename = f"{data_type}_embeddings{suffix}.npy"
        cache_path = os.path.join(Config.CACHE_DIR, filename)

        # 1. Try loading from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                try:
                    print(f"Loading embeddings from cache: {cache_path}")
                    embeddings = np.load(cache_path)

                    # Verify length matches input
                    if len(embeddings) == len(texts):
                        return embeddings
                    else:
                        print(
                            f"Cache size mismatch ({len(embeddings)} vs {len(texts)}). Recomputing."
                        )
                except Exception as e:
                    print(f"Failed to load cache ({e}). Recomputing.")
            else:
                print(f"Cache not found for {data_type}. Computing from scratch.")

        # 2. Compute embeddings
        print(f"Generating embeddings for {len(texts)} texts...")

        # encode() handles batching internally.
        # convert_to_numpy=True returns a numpy array directly.
        # normalize_embeddings=True ensures unit length, often helpful for linear models.
        embeddings = self.model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )

        # 3. Save to cache
        print(f"Saving embeddings to cache: {cache_path}")
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

        return embeddings
