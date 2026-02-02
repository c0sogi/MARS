import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import (
    TRANSFORMER_MODEL_NAME,
    CACHE_DIR,
    BATCH_SIZE,
    MAX_SEQ_LENGTH,
    SEED,
)


class EmbeddingPipeline:
    """
    Manages the Sentence Transformer model for embedding generation and
    feature construction.
    """

    def __init__(self, model_name=TRANSFORMER_MODEL_NAME, device=None):
        """
        Initializes the embedding pipeline.

        Args:
            model_name (str): Name of the pre-trained Sentence Transformer.
            device (str): Device to run the model on ('cuda', 'cpu', etc.).
        """
        # Set seeds for reproducibility
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(
            f"Initializing EmbeddingPipeline with model: {model_name} on {self.device}"
        )
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = MAX_SEQ_LENGTH

        # Set model to evaluation mode
        self.model.eval()

    def encode_texts(self, texts, batch_size=BATCH_SIZE):
        """
        Generates dense embeddings for a list/array of texts.

        Args:
            texts (list or np.ndarray): Input texts.
            batch_size (int): Batch size for inference.

        Returns:
            np.ndarray: Array of embeddings with shape (N, embedding_dim).
        """
        # Ensure input is a list or array
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def create_interaction_features(self, u, v):
        """
        Constructs the interaction feature vector from question embeddings (u)
        and answer embeddings (v).

        Features: [u, v, |u - v|, u * v]

        Args:
            u (np.ndarray): Question embeddings (N, D).
            v (np.ndarray): Answer embeddings (N, D).

        Returns:
            np.ndarray: Combined feature matrix (N, 4*D).
        """
        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch: u {u.shape} vs v {v.shape}")

        diff_feat = np.abs(u - v)
        prod_feat = u * v

        # Concatenate along the feature dimension (axis 1)
        features = np.concatenate([u, v, diff_feat, prod_feat], axis=1)
        return features

    def get_features(self, questions, answers, split_name, load_cached_data=True):
        """
        High-level method to get features for a dataset split.
        Handles caching logic (Load -> Compute -> Save).

        Args:
            questions (np.ndarray): Array of question texts.
            answers (np.ndarray): Array of answer texts.
            split_name (str): Name of the split (e.g., 'train', 'val') for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: The computed feature matrix.
        """
        # Define cache file path (using .npy for efficient array storage)
        cache_path = os.path.join(CACHE_DIR, f"{split_name}_features.npy")

        # 1. IF load_cached_data is True: Try to load.
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(
                    f"Loading cached features for '{split_name}' from {cache_path}..."
                )
                features = np.load(cache_path)

                # Basic validation of shape length (N should match inputs)
                if len(features) == len(questions):
                    return features
                else:
                    print(
                        f"Cache shape mismatch ({len(features)} vs {len(questions)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Failed to load cache for {split_name}: {e}. Recomputing...")

        # 2. IF loading fails OR load_cached_data is False: Compute.
        print(f"Computing features for '{split_name}'...")

        # Encode questions
        print(f"  Encoding {len(questions)} questions...")
        u = self.encode_texts(questions)

        # Encode answers
        print(f"  Encoding {len(answers)} answers...")
        v = self.encode_texts(answers)

        # Create interactions
        print("  Creating interaction features...")
        features = self.create_interaction_features(u, v)

        # Save to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, features)
            print(f"Saved features for '{split_name}' to cache.")
        except Exception as e:
            print(f"Warning: Could not save feature cache: {e}")

        # 3. Return data.
        return features
