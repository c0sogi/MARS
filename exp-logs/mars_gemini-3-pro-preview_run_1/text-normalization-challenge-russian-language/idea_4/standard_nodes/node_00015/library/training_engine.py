import os
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, ensure_dir
from library.data_processing import (
    prepare_neural_dataset,
    TextNormalizationDataset,
    collate_fn,
)
from library.neural_network import NeuralNormalizer


class Trainer:
    """
    Manages the training lifecycle of the neural model, including data preparation,
    mixed-precision training loop, validation, and checkpointing.
    """

    def __init__(self, config):
        self.config = config
        seed_everything(config.seed)
        self.device = config.device

    def run(self, load_cached_data=True, max_samples=None):
        """
        Executes the training pipeline.

        Args:
            load_cached_data (bool): Whether to load pre-computed data/tokenizer.
            max_samples (int, optional): Limit dataset size for debugging.

        Returns:
            NeuralNormalizer: The trained model wrapper.
        """
        # 1. Data Preparation
        print("Preparing training data...")
        df_train, tokenizer = prepare_neural_dataset(
            self.config, split="train", load_cached_data=load_cached_data
        )

        print("Preparing validation data...")
        df_val, _ = prepare_neural_dataset(
            self.config,
            split="val",
            tokenizer=tokenizer,
            load_cached_data=load_cached_data,
        )

        # Debugging: Limit samples
        if max_samples is not None:
            print(f"Limiting training data to {max_samples} samples.")
            df_train = df_train.head(max_samples)
            df_val = df_val.head(max_samples)

        # 2. Dataset & DataLoader
        train_dataset = TextNormalizationDataset(
            df_train, tokenizer, self.config, mode="train"
        )
        val_dataset = TextNormalizationDataset(
            df_val, tokenizer, self.config, mode="val"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # 3. Model Initialization
        print("Initializing model...")
        normalizer = NeuralNormalizer(self.config, tokenizer)

        # 4. Training Loop (Mixed Precision)
        self._train_loop(normalizer, train_loader, val_loader)

        return normalizer

    def _train_loop(self, normalizer, train_loader, val_loader):
        """
        Custom training loop implementing Mixed Precision (AMP) and Early Stopping.
        Overrides the standard FP32 loop in NeuralNormalizer.
        """
        model = normalizer.model
        optimizer = normalizer.optimizer
        criterion = normalizer.criterion
        config = self.config

        scaler = GradScaler()
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting Mixed-Precision Training on {self.device}...")

        for epoch in range(1, config.num_epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                # Prepare inputs/targets (Teacher Forcing)
                # tgt_input: <bos> ... token
                # tgt_out:   token ... <eos>
                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                # Create masks using helper from normalizer
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                    normalizer._create_mask(src, tgt_input)
                )

                optimizer.zero_grad()

                with autocast():
                    logits = model(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,  # memory_key_padding_mask same as src_padding_mask
                    )

                    # Flatten for loss
                    # logits: [batch, seq_len, vocab]
                    # tgt_out: [batch, seq_len]
                    loss = criterion(
                        logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                    )

                # Scaler Step
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm
                )

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation (FP32 is sufficient and stable for eval)
            avg_val_loss = normalizer.evaluate(val_loader)

            end_time = time.time()
            epoch_duration = end_time - start_time
            epoch_mins = int(epoch_duration // 60)
            epoch_secs = int(epoch_duration % 60)

            print(f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"    Train Loss: {avg_train_loss:.10f}")
            print(f"    Val Loss:   {avg_val_loss:.10f}")

            # Early Stopping & Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                ensure_dir(config.model_best_path)
                torch.save(model.state_dict(), config.model_best_path)
                print("    -> Best model saved.")
            else:
                patience_counter += 1
                print(
                    f"    -> No improvement. Patience: {patience_counter}/{config.early_stopping_patience}"
                )

            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Reload best model weights
        normalizer.load(config.model_best_path)
