import os
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import get_artifact_path, seed_everything
from library.data_processing import load_and_group_data, get_tokenizer, NeuralDataset
from library.neural_model import NeuralTrainer


class ModelTrainer:
    """
    Orchestrates the training lifecycle of the Neural Normalizer model.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(Config.SEED)

    def run(self, force_retrain=False):
        """
        Runs the training pipeline: loads data, prepares datasets, and trains the model.

        Args:
            force_retrain (bool): If True, ignores existing checkpoints and forces a fresh training run.

        Returns:
            str: The file path to the best saved model checkpoint.
        """
        # Define the expected path for the best model based on current config hash
        model_path = get_artifact_path("neural_normalizer_best.pt")

        # Check if the model already exists to avoid redundant computation
        if os.path.exists(model_path) and not force_retrain:
            print(f"Trained model found at {model_path}. Skipping training.")
            return model_path

        print(f"Starting training pipeline on {self.device}...")

        # 1. Load and Group Data
        # Relies on library functions which handle caching of the grouped dataframes
        print("Loading training data...")
        train_df = load_and_group_data("train")
        print("Loading validation data...")
        val_df = load_and_group_data("val")

        # 2. Prepare Tokenizer
        # Ensures tokenizer is fit on training data and consistent with config vocab size
        print("Preparing tokenizer...")
        tokenizer = get_tokenizer(train_grouped_df=train_df)
        print(f"Tokenizer ready with vocab size: {len(tokenizer)}")

        # 3. Create Datasets
        print("Creating NeuralDatasets...")
        # Train dataset: Includes all semiotic tokens + digits + 1% random sample of PLAIN text
        # The random sample helps the model learn to copy simple text (identity mapping)
        train_dataset = NeuralDataset(
            grouped_df=train_df,
            tokenizer=tokenizer,
            mode="train",
            context_window=Config.CONTEXT_WINDOW,
            sample_ratio=0.01,
        )

        # Val dataset: Strict validation on semiotic/digit tokens only
        val_dataset = NeuralDataset(
            grouped_df=val_df,
            tokenizer=tokenizer,
            mode="val",
            context_window=Config.CONTEXT_WINDOW,
            sample_ratio=0.0,
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        # 4. Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=NeuralDataset.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=NeuralDataset.collate_fn,
            pin_memory=True,
        )

        # 5. Initialize and Run Trainer
        # NeuralTrainer handles the training loop, validation, early stopping, and saving
        trainer = NeuralTrainer(tokenizer, device=self.device)
        best_model_path = trainer.train(train_loader, val_loader)

        print(f"Training finished. Best model saved to: {best_model_path}")
        return best_model_path
