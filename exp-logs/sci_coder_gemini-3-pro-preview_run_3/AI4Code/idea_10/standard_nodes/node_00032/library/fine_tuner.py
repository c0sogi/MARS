import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library.config import Config
from library.data_loader import RelaxedProximityDataset
from library.utils import set_seed


class FineTuner:
    """
    Manages the fine-tuning of the sentence-transformers backbone using
    contrastive learning (Multiple Negatives Ranking Loss).
    """

    def __init__(self):
        """
        Initializes the FineTuner with the pre-trained model defined in Config.
        """
        self.model_name = Config.MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(
        self,
        metadata_path=Config.TRAIN_METADATA_PATH,
        subset_size=Config.FINE_TUNE_SUBSET_SIZE,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        load_cached_data=True,
    ):
        """
        Executes the fine-tuning pipeline.

        Args:
            metadata_path (str): Path to training metadata.
            subset_size (int): Number of notebooks to use for training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for the DataLoader.
            learning_rate (float): Learning rate for the optimizer.
            load_cached_data (bool): Whether to load pre-computed pairs from disk.
        """
        set_seed(Config.SEED)
        print(
            f"Initializing fine-tuning for {self.model_name} on device {self.device}..."
        )

        # 1. Prepare Dataset
        train_dataset = RelaxedProximityDataset(
            metadata_path=metadata_path,
            load_cached_data=load_cached_data,
            subset_size=subset_size,
        )

        if len(train_dataset) == 0:
            print("Warning: Dataset is empty. Skipping training.")
            return

        # 2. Prepare DataLoader
        # SentenceTransformer expects a standard PyTorch DataLoader
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )

        # 3. Define Loss
        # MultipleNegativesRankingLoss is effective for (anchor, positive) pairs
        # It treats other samples in the batch as negative samples.
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        # 4. Train
        output_path = Config.FINE_TUNED_MODEL_PATH
        print(f"Starting training for {epochs} epoch(s)...")

        # We use the built-in fit method of SentenceTransformer
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            optimizer_params={"lr": learning_rate},
            output_path=output_path,
            show_progress_bar=False,  # Silent execution
            use_amp=True,  # Use Automatic Mixed Precision if available
        )

        print(f"Fine-tuning complete. Model saved to {output_path}")
