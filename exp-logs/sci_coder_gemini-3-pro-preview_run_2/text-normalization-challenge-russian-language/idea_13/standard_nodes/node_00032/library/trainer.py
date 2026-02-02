import os
import torch
from library.config import Config, set_seed
from library.tokenizer import HybridTokenizer
from library.dataset import DatasetManager
from library.transformer_model import Seq2SeqTransformer, TransformerTrainer


class ModelTrainer:
    """
    Orchestrates the training pipeline for the Hybrid Cascade Transformer (Tier 2).
    Integrates Tokenization, Dataset Management, and the Training Loop.
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing hyperparameters and paths.
        """
        self.config = config

    def run(self):
        """
        Executes the full training pipeline:
        1. Fits/Loads Tokenizers.
        2. Prepares DataLoaders (with caching).
        3. Initializes the Seq2Seq Transformer.
        4. Runs the Training Loop with Early Stopping.

        Returns:
            str: Path to the best saved model checkpoint.
        """
        # Ensure reproducibility
        set_seed(self.config.seed)

        print("=== Starting Model Training Pipeline ===")

        # 1. Initialize and Fit Tokenizer
        # The tokenizer handles caching internally via load_cached_data=True
        print("Initializing Tokenizer...")
        tokenizer = HybridTokenizer(self.config)
        tokenizer.fit(load_cached_data=True)

        # 2. Prepare Data
        # DatasetManager handles loading, filtering, balancing, and tokenization
        print("Preparing DataLoaders...")
        dataset_manager = DatasetManager(self.config, tokenizer)
        train_loader, val_loader = dataset_manager.get_dataloaders(
            load_cached_data=True
        )

        # 3. Initialize Model
        # Extract vocabulary sizes and special token IDs needed for the model architecture
        src_vocab_size = tokenizer.char_vocab_size_actual
        tgt_vocab_size = tokenizer.bpe_vocab_size
        src_pad_idx = tokenizer.char2id[tokenizer.PAD_TOKEN]
        tgt_pad_idx = tokenizer.bpe_pad_id

        print(f"Initializing Seq2SeqTransformer...")
        print(f"  Encoder Vocab (Char): {src_vocab_size}")
        print(f"  Decoder Vocab (BPE) : {tgt_vocab_size}")
        print(f"  Encoder Pad ID      : {src_pad_idx}")
        print(f"  Decoder Pad ID      : {tgt_pad_idx}")

        model = Seq2SeqTransformer(
            config=self.config,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
        )

        # 4. Initialize Trainer
        # TransformerTrainer handles the optimization loop, logging, and checkpointing
        print("Initializing Trainer...")
        trainer = TransformerTrainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # 5. Start Training
        print("Starting Training Loop...")
        trainer.train()

        print(f"Training complete. Best model saved at: {trainer.best_model_path}")
        return trainer.best_model_path
