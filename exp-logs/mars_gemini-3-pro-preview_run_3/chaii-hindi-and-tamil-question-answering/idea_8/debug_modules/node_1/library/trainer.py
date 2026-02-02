import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from library.config import Config
from library.utils import set_seed
from library.data_loader import QADataset, collate_fn, prepare_features
from library.model_factory import get_model


class Trainer:
    """
    Handles the training and evaluation loop for the Question Answering model.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Optimizer
        # We separate parameters for weight decay handling (standard practice)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

        # Scheduler
        num_training_steps = len(train_loader) * config.epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Loss Function
        # Class weights are moved to device to ensure compatibility
        weights = config.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    def train_epoch(self, epoch_idx: int):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in self.train_loader:
            # Move tensors to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate Loss
            # Flatten logits: (batch * seq_len, num_labels)
            # Flatten labels: (batch * seq_len)
            loss = self.criterion(
                logits.view(-1, self.config.num_labels), labels.view(-1)
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch_idx+1} | Train Loss: {avg_loss} | Time: {duration:.2f}s")
        return avg_loss

    def evaluate(self):
        """Runs evaluation on the validation set."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = self.criterion(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_model(self, path: str):
        """Saves the model state dictionary."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def train_single_seed(config: Config, seed: int, train_df, val_df):
    """
    Executes the training pipeline for a single random seed.
    """
    print(f"\n=== Starting Training for Seed {seed} ===")
    set_seed(seed)

    # 1. Prepare Datasets and Loaders
    train_dataset = QADataset(train_df, is_test=False)
    val_dataset = QADataset(val_df, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = get_model(config)

    # 3. Initialize Trainer
    trainer = Trainer(config, model, train_loader, val_loader)

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    model_save_path = os.path.join(config.model_dir, f"model_seed_{seed}.pt")

    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.evaluate()

        # Print full precision as requested
        print(f"Epoch {epoch+1} | Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Clean up to free memory
    del model, trainer, train_loader, val_loader
    torch.cuda.empty_cache()


def run_training(config: Config):
    """
    Main entry point for the training module.
    Prepares data once and then iterates through all seeds defined in config.
    """
    print("Initializing Training Pipeline...")

    # 1. Load Tokenizer and Data
    # We load data once to avoid redundant processing
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("Preparing Training Features...")
    train_df = prepare_features(config, tokenizer, split="train", load_cached_data=True)

    print("Preparing Validation Features...")
    val_df = prepare_features(config, tokenizer, split="val", load_cached_data=True)

    # 2. Iterate Seeds
    for seed in config.seeds:
        train_single_seed(config, seed, train_df, val_df)

    print("\nAll training runs completed.")
