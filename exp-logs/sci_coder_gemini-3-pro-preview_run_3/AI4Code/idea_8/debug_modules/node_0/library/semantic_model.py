import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library.config import Config
from library.data_processing import prepare_training_pairs, NotebookDataset
from library.utils import set_seed


class FineTuner:
    """
    Manages the lifecycle of the Sentence Transformer backbone.
    Includes methods for training (fine-tuning) and saving the model.
    """

    def __init__(self):
        """
        Initializes the FineTuner by loading the pre-trained model defined in Config.
        """
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        print(f"Initializing Semantic Model: {Config.MODEL_NAME}")
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)

    def train(self, load_cached_data=True):
        """
        Executes the fine-tuning loop using MultipleNegativesRankingLoss.

        Args:
            load_cached_data (bool): Whether to load pre-computed pairs from cache.
        """
        print("Preparing training pairs for fine-tuning...")
        # Load training pairs using the data processing library
        train_examples = prepare_training_pairs(
            load_cached_data=load_cached_data, debug=Config.DEBUG
        )

        if not train_examples:
            print("No training examples available. Skipping fine-tuning.")
            return

        # Wrap examples in the Dataset and DataLoader
        train_dataset = NotebookDataset(train_examples)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=Config.BATCH_SIZE
        )

        # Define the loss function
        # MultipleNegativesRankingLoss is ideal for (anchor, positive) pairs without explicit negatives
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        print(
            f"Starting training for {Config.NUM_EPOCHS} epoch(s) on {len(train_examples)} pairs..."
        )

        # Execute the training loop
        # SentenceTransformer.fit handles the loop, optimizer, scheduler, and saving
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=Config.NUM_EPOCHS,
            warmup_steps=Config.WARMUP_STEPS,
            optimizer_params={"lr": Config.LEARNING_RATE},
            weight_decay=Config.WEIGHT_DECAY,
            show_progress_bar=True,
            output_path=Config.BACKBONE_OUTPUT_DIR,
            use_amp=True if self.device == "cuda" else False,
        )
        print("Fine-tuning complete.")

    def save_model(self, output_path=None):
        """
        Saves the model to the specified path or the default configuration path.

        Args:
            output_path (str, optional): Path to save the model. Defaults to Config.BACKBONE_OUTPUT_DIR.
        """
        if output_path is None:
            output_path = Config.BACKBONE_OUTPUT_DIR

        print(f"Saving fine-tuned model to {output_path}...")
        self.model.save(output_path)

    def encode(self, sentences, batch_size=None, show_progress_bar=False):
        """
        Encodes a list of sentences into embeddings.

        Args:
            sentences (list): List of text strings to encode.
            batch_size (int, optional): Batch size for encoding. Defaults to Config.BATCH_SIZE.
            show_progress_bar (bool): Whether to show a progress bar.

        Returns:
            numpy.ndarray: Array of embeddings.
        """
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=self.device,
            convert_to_numpy=True,
        )
