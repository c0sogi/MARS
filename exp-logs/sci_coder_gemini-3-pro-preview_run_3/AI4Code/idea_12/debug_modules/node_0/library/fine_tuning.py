import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from library.config import Config
from library.pair_generation import generate_bidirectional_pairs


class FineTuner:
    """
    Handles the fine-tuning of the Sentence Transformer backbone using
    contrastive learning on bidirectional markdown-code pairs.
    """

    def __init__(self):
        self.model_name = Config.Model.BASE_MODEL_NAME
        self.output_path = Config.Paths.MODEL_OUTPUT_DIR
        self.batch_size = Config.Training.FINE_TUNE_BATCH_SIZE
        self.epochs = Config.Training.FINE_TUNE_EPOCHS
        self.lr = Config.Training.FINE_TUNE_LR
        self.device = Config.Model.DEVICE
        self.max_seq_len = Config.Model.MAX_SEQ_LEN
        self.num_workers = Config.Training.NUM_WORKERS

    def set_seed(self):
        """Sets random seeds for reproducibility."""
        seed = Config.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        """
        Executes the fine-tuning pipeline:
        1. Loads training pairs.
        2. Initializes the model and loss.
        3. Runs the training loop.
        4. Saves the model.
        """
        self.set_seed()

        print(f"Initializing FineTuner with model: {self.model_name}")
        print(f"Output directory: {self.output_path}")

        # 1. Load Data
        # We rely on the caching mechanism implemented in generate_bidirectional_pairs
        df_pairs = generate_bidirectional_pairs(load_cached_data=True)

        if df_pairs.empty:
            raise ValueError(
                "No training pairs were generated. Check dataset availability."
            )

        print(f"Loaded {len(df_pairs)} training pairs.")

        # Convert DataFrame to InputExample list
        train_examples = []
        for _, row in df_pairs.iterrows():
            # Ensure text is string
            md_text = str(row["markdown"])
            code_text = str(row["code"])
            train_examples.append(InputExample(texts=[md_text, code_text]))

        # Create DataLoader
        # We use a simple DataLoader as SentenceTransformer handles collation internally for InputExamples
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # 2. Initialize Model
        model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = self.max_seq_len

        # 3. Define Loss
        # MultipleNegativesRankingLoss is effective for (anchor, positive) pairs
        # It uses other samples in the batch as negatives.
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        # 4. Train
        print(f"Starting training for {self.epochs} epochs...")

        # We use the fit method provided by SentenceTransformer
        # show_progress_bar is set to False to reduce clutter as per instructions
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            optimizer_params={"lr": self.lr},
            output_path=self.output_path,
            show_progress_bar=False,
            save_best_model=False,  # We don't have a separate dev evaluator here
            use_amp=(
                True if self.device == "cuda" else False
            ),  # Automatic Mixed Precision for speed
        )

        print(f"Fine-tuning complete. Model saved to {self.output_path}")
