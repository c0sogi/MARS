import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.utils import ModelEMA, save_checkpoint, calculate_accuracy


class Engine:
    """
    Handles the training, validation, and inference processes.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Initialize Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.MAX_EPOCHS, eta_min=1e-6
        )

        # Initialize EMA (Exponential Moving Average)
        # We use the EMA model for validation and inference to improve generalization
        self.ema = ModelEMA(self.model, decay=0.999, device=self.device)

        self.best_score = 0.0
        self.current_epoch = 0

    def train_one_epoch(self, train_dataset):
        """
        Runs one epoch of training using weighted random sampling on GPU-resident data.
        """
        self.model.train()
        total_loss = 0.0

        # Determine number of steps (batches) per epoch
        # Since we use random sampling with replacement, we define an epoch
        # as seeing the equivalent number of samples as the dataset size.
        num_samples = len(train_dataset)
        num_batches = int(np.ceil(num_samples / Config.BATCH_SIZE))

        for _ in range(num_batches):
            # 1. Get Batch Indices (Weighted Random Sampling)
            # This returns a tensor of indices on the GPU
            indices = train_dataset.get_batch_indices(Config.BATCH_SIZE)

            # 2. Get Data (Already on GPU)
            # Indexing the GPU tensors directly avoids CPU-GPU transfer
            waveforms = train_dataset.waveforms[indices]
            targets = train_dataset.labels[indices]

            # 3. Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(waveforms)

            # 4. Loss & Backward
            loss = self.criterion(outputs, targets)
            loss.backward()

            # 5. Optimization Step
            self.optimizer.step()

            # 6. Update EMA
            self.ema.update(self.model)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, val_dataset):
        """
        Evaluates the model on the validation set using the EMA model.
        """
        # Use EMA model for validation
        model_to_eval = self.ema.ema_model
        model_to_eval.eval()

        total_loss = 0.0
        all_outputs = []
        all_targets = []

        num_samples = len(val_dataset)
        indices = torch.arange(num_samples, device=self.device)

        with torch.no_grad():
            # Sequential iteration for validation
            for start_idx in range(0, num_samples, Config.BATCH_SIZE):
                end_idx = min(start_idx + Config.BATCH_SIZE, num_samples)
                batch_indices = indices[start_idx:end_idx]

                waveforms = val_dataset.waveforms[batch_indices]
                targets = val_dataset.labels[batch_indices]

                outputs = model_to_eval(waveforms)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * (end_idx - start_idx)

                all_outputs.append(outputs)
                all_targets.append(targets)

        avg_loss = total_loss / num_samples

        # Concatenate for metric calculation
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        accuracy = calculate_accuracy(all_outputs, all_targets)

        return avg_loss, accuracy

    def fit(self, train_dataset, val_dataset):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        patience_counter = 0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_one_epoch(train_dataset)

            # Validate
            val_loss, val_acc = self.evaluate(val_dataset)

            # Scheduler Step
            self.scheduler.step()

            # Print metrics in full precision
            print(
                f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val Accuracy = {val_acc}"
            )

            # Checkpoint & Early Stopping
            if val_acc > self.best_score:
                self.best_score = val_acc
                patience_counter = 0
                save_checkpoint(
                    self.ema.ema_model,  # Save the EMA model as best
                    self.optimizer,
                    epoch,
                    val_acc,
                    Config.BEST_MODEL_PATH,
                )
                print(f"New best model saved with accuracy: {val_acc}")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    def predict(self, test_dataset):
        """
        Generates predictions for the test set using the best saved model.
        Returns:
            fnames: List of filenames
            predictions: List of predicted labels
        """
        # Load best model weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            try:
                checkpoint = torch.load(
                    Config.BEST_MODEL_PATH, map_location=self.device
                )
                self.ema.ema_model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded best model for prediction.")
            except Exception as e:
                print(f"Error loading best model: {e}. Using current EMA state.")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current EMA model state."
            )

        model_to_eval = self.ema.ema_model
        model_to_eval.eval()

        all_preds = []

        num_samples = len(test_dataset)
        indices = torch.arange(num_samples, device=self.device)

        with torch.no_grad():
            for start_idx in range(0, num_samples, Config.BATCH_SIZE):
                end_idx = min(start_idx + Config.BATCH_SIZE, num_samples)
                batch_indices = indices[start_idx:end_idx]

                waveforms = test_dataset.waveforms[batch_indices]

                outputs = model_to_eval(waveforms)

                # Get predicted class indices
                _, preds = torch.max(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Map indices back to labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

        return test_dataset.fnames, predicted_labels
