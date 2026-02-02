import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import library.config as config
from library.utils import set_seed, calculate_f1_score
from library.data import get_dataloaders
from library.model import HybridCNNTransformer


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        save_path,
        patience=config.PATIENCE,
        prediction_threshold=config.PREDICTION_THRESHOLD,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.patience = patience
        self.prediction_threshold = prediction_threshold

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def train_epoch(self, epoch_idx):
        """Runs one training epoch."""
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (tokens, labels) in enumerate(self.train_loader):
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(tokens)
                    loss = self.criterion(logits, labels)

                # Backward Pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tokens)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Update Scheduler
            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch_idx+1} | Train Loss: {avg_loss} | Time: {elapsed:.2f}s")
        return avg_loss

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for tokens, labels in self.val_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(tokens)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(tokens)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate F1 Score
        val_f1 = calculate_f1_score(
            all_preds, all_labels, threshold=self.prediction_threshold
        )

        print(f"Validation | Loss: {avg_loss} | F1-Score: {val_f1}")
        return avg_loss, val_f1

    def fit(self, num_epochs):
        """Main training loop with Early Stopping."""
        best_f1 = 0.0
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            val_loss, val_f1 = self.validate()

            # Checkpoint & Early Stopping
            if val_f1 > best_f1:
                print(f"New best F1! ({best_f1} -> {val_f1}). Saving model...")
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best F1 Score: {best_f1}")

        # Load best model weights before returning
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        return best_f1


def run_training(load_cached_data=True):
    """
    Initializes data, model, and runs the training pipeline.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from disk.
    """
    set_seed(config.SEED)

    # 1. Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, tokenizer, encoder = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing Model...")
    model = HybridCNNTransformer(
        vocab_size=len(tokenizer.vocab) + 1,  # +1 safety margin though vocab handles it
        embed_dim=config.EMBED_DIM,
        cnn_filters=config.CNN_FILTERS,
        cnn_kernel_size=config.CNN_KERNEL_SIZE,
        transformer_layers=config.TRANSFORMER_LAYERS,
        num_heads=config.NUM_HEADS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.DROPOUT,
        num_classes=len(encoder.classes_),
        max_len=config.MAX_LEN,
    )
    model.to(config.DEVICE)

    # 3. Setup Training Components
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=config.DEVICE,
        save_path=config.MODEL_SAVE_PATH,
        patience=config.PATIENCE,
        prediction_threshold=config.PREDICTION_THRESHOLD,
    )

    # 5. Run Training
    trainer.fit(num_epochs=config.NUM_EPOCHS)

    return trainer.model, tokenizer, encoder
