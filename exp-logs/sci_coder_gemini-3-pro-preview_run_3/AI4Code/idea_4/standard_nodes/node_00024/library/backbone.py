import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library.config import Config
from library.utils import set_seed
from library.data_loader import prepare_training_pairs, FineTuningDataset


class SemanticModel:
    """
    Wrapper around SentenceTransformer for Domain-Adaptive Semantic Alignment.
    Handles fine-tuning on (Markdown, Next_Code) pairs and encoding of text cells.
    """

    def __init__(self, model_name_or_path=Config.PRETRAINED_MODEL_NAME):
        """
        Args:
            model_name_or_path (str): Path to a local model or a HuggingFace model name.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"Loading SentenceTransformer model from {model_name_or_path} on {self.device}..."
        )
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        self.model.max_seq_length = Config.MAX_SEQ_LENGTH

    def fine_tune(
        self,
        train_metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=Config.FINE_TUNED_MODEL_PATH,
        batch_size=Config.BACKBONE_BATCH_SIZE,
        epochs=Config.BACKBONE_EPOCHS,
        learning_rate=Config.BACKBONE_LEARNING_RATE,
        debug=Config.DEBUG,
        load_cached_data=True,
    ):
        """
        Fine-tunes the backbone model using Multiple Negatives Ranking Loss.

        Args:
            train_metadata_path (str): Path to the training metadata CSV.
            output_path (str): Directory to save the fine-tuned model.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            debug (bool): If True, uses a small subset of data.
            load_cached_data (bool): Whether to load training pairs from parquet cache.
        """
        set_seed(Config.SEED)

        # 1. Prepare Training Data
        cache_name = "train_pairs_debug" if debug else "train_pairs_full"
        df_pairs = prepare_training_pairs(
            metadata_path=train_metadata_path,
            cache_name=cache_name,
            load_cached_data=load_cached_data,
            debug=debug,
        )

        print(f"Initializing dataset with {len(df_pairs)} pairs...")
        train_dataset = FineTuningDataset(df_pairs)

        # SentenceTransformer expects a standard PyTorch DataLoader
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )

        # 2. Define Loss
        # MultipleNegativesRankingLoss is effective for (query, positive) pairs without explicit negatives.
        # It treats other samples in the batch as negatives.
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # 3. Train
        print(f"Starting fine-tuning for {epochs} epoch(s)...")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            optimizer_params={"lr": learning_rate},
            output_path=output_path,
            show_progress_bar=True,
            use_amp=True,  # Use Automatic Mixed Precision for speed
        )

        print(f"Fine-tuning complete. Model saved to {output_path}")

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """
        Generates dense vector embeddings for a list of texts.

        Args:
            texts (list[str]): List of texts to encode.
            batch_size (int): Batch size for encoding.
            show_progress_bar (bool): Whether to show progress.

        Returns:
            np.ndarray: Embeddings matrix of shape (n_texts, hidden_dim).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=self.device,
            convert_to_numpy=True,
        )
