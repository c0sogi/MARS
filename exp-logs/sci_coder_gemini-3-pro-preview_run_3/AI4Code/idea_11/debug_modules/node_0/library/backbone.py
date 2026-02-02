import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from library.config import Config, set_seed
from library.data_loader import prepare_relaxed_pairs


class BackboneTrainer:
    """
    Manages the training and inference of the semantic backbone model (SentenceTransformer).
    """

    def __init__(self):
        self.model_name = Config.MODEL_NAME
        self.save_path = Config.MODEL_SAVE_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def _load_model(self):
        """
        Internal helper to load the model.
        Prioritizes the fine-tuned model at save_path.
        Falls back to the base model_name.
        """
        if self.model is not None:
            return

        # Check if a fine-tuned model exists at the save path
        if os.path.exists(self.save_path):
            # Basic check to see if it looks like a model directory
            if os.path.exists(os.path.join(self.save_path, "config.json")):
                print(f"Loading fine-tuned backbone from {self.save_path}")
                self.model = SentenceTransformer(self.save_path, device=self.device)
                return

        # Fallback to base model
        print(f"Loading base backbone {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def train(self, load_cached_data=True):
        """
        Fine-tunes the backbone model using Contrastive Learning (MultipleNegativesRankingLoss).

        Args:
            load_cached_data (bool): Whether to load training pairs from cache.
        """
        set_seed(Config.SEED)

        # 1. Prepare Data
        # We use the relaxed pairs strategy: (Markdown, Nearest Subsequent Code)
        print("Preparing fine-tuning data...")
        pairs_df = prepare_relaxed_pairs(
            metadata_path=Config.TRAIN_PATH,
            sample_size=Config.FT_SAMPLE_SIZE,
            load_cached_data=load_cached_data,
        )

        print(f"Training on {len(pairs_df)} pairs.")

        # 2. Convert to InputExamples
        # SentenceTransformers expects a list of InputExample objects
        train_examples = [
            InputExample(texts=[row["markdown"], row["code"]])
            for _, row in pairs_df.iterrows()
        ]

        # 3. Create DataLoader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=Config.BATCH_SIZE
        )

        # 4. Initialize Model
        # We start fresh from the base model for training
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 5. Define Loss
        # MultipleNegativesRankingLoss is ideal for (anchor, positive) pairs.
        # It treats other samples in the batch as negative examples.
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # 6. Train
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        # Note: We do not use an explicit evaluator here because we only have positive pairs.
        # Standard evaluators (BinaryClassification, EmbeddingSimilarity) often require
        # variance in labels (pos/neg) or a specific retrieval corpus.
        # We rely on the training loss (MNRL) as the primary metric.
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=Config.NUM_EPOCHS,
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={"lr": Config.LEARNING_RATE},
            output_path=self.save_path,
            show_progress_bar=True,
        )

        print(f"Fine-tuning complete. Model saved to {self.save_path}")

    def encode(self, texts, batch_size=Config.BATCH_SIZE, show_progress_bar=False):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list of str): Texts to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show progress.

        Returns:
            np.ndarray: Embeddings matrix of shape (n_texts, hidden_dim).
        """
        self._load_model()

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Encode
        # normalize_embeddings=True is crucial for Cosine Similarity
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings
