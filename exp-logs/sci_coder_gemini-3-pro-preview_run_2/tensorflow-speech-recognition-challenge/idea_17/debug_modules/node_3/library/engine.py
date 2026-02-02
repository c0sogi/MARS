import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    MIN_LR,
    WARMUP_EPOCHS,
    CHECKPOINT_DIR,
    SUBMISSION_PATH,
    TEST_CSV,
    ID2LABEL,
    NUM_CLASSES,
    SEED,
    WORKING_DIR,
)
from library.utils import set_seed, ModelEMA
from library.model import AudioEfficientNetV2
from library.augmentations import GPUBackgroundNoiseMixer


class Trainer:
    def __init__(self, data_dict, device=None):
        """
        Initialize the Trainer with data and model components.

        Args:
            data_dict (dict): Dictionary containing waveforms and labels.
            device (torch.device): Device to run training on.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(SEED)

        print(f"Initializing Trainer on device: {self.device}")

        # 1. Load Data to GPU
        print("Moving datasets to GPU memory...")
        self.train_wavs = data_dict["train_waveforms"].to(self.device)
        self.train_lbls = data_dict["train_labels"].to(self.device)

        self.val_wavs = data_dict["val_waveforms"].to(self.device)
        self.val_lbls = data_dict["val_labels"].to(self.device)

        # Test data might be large, but fits in A100 (40GB).
        # If OOM occurs, one could keep it on CPU and move batches,
        # but strategy dictates GPU residency for speed.
        self.test_wavs = data_dict["test_waveforms"].to(self.device)
        # test_labels are placeholders, not needed for inference logic

        # Background noise for augmentation
        self.bg_noise_list = data_dict["background_noise"]

        # 2. Prepare Weighted Random Sampling
        print("Calculating class weights for balanced sampling...")
        class_counts = torch.bincount(self.train_lbls, minlength=NUM_CLASSES)
        # Avoid division by zero
        class_counts = class_counts.float() + 1e-6
        class_weights = 1.0 / class_counts
        # Assign weight to each sample
        self.sample_weights = class_weights[self.train_lbls]

        # 3. Initialize Model and Components
        print("Building model...")
        self.model = AudioEfficientNetV2(num_classes=NUM_CLASSES).to(self.device)

        # EMA Model
        self.ema = ModelEMA(self.model)

        # Augmentation Module
        self.mixer = GPUBackgroundNoiseMixer(self.bg_noise_list, device=self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=MIN_LR
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # State
        self.best_acc = 0.0
        self.best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training using GPU-resident data and weighted sampling.
        """
        self.model.train()
        self.mixer.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Define number of steps per epoch
        num_samples = self.train_wavs.size(0)
        steps_per_epoch = num_samples // BATCH_SIZE

        for _ in range(steps_per_epoch):
            # 1. Weighted Random Sampling (GPU)
            # replacement=True is standard for weighted sampling
            indices = torch.multinomial(
                self.sample_weights, BATCH_SIZE, replacement=True
            )

            # 2. Fetch Batch
            x = self.train_wavs[indices]
            y = self.train_lbls[indices]

            # 3. Waveform Augmentation (Mix Noise)
            x = self.mixer(x)

            # 4. Forward Pass
            # Model handles Spectrogram generation and SpecAugment internally
            self.optimizer.zero_grad()
            logits = self.model(x)

            # 5. Loss & Backward
            loss = self.criterion(logits, y)
            loss.backward()

            self.optimizer.step()

            # 6. Update EMA
            self.ema.update(self.model)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += BATCH_SIZE

        avg_loss = total_loss / steps_per_epoch
        accuracy = correct / total

        # Update scheduler at epoch end
        self.scheduler.step()

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, waveforms, labels):
        """
        Evaluates the model (using EMA weights) on the provided dataset.
        """
        ema_model = self.ema.get_model()
        ema_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        num_samples = waveforms.size(0)
        # Sequential processing
        indices = torch.arange(num_samples, device=self.device)
        # Split into batches
        batches = torch.split(indices, BATCH_SIZE)

        for batch_idx in batches:
            x = waveforms[batch_idx]
            y = labels[batch_idx]

            # Forward (No Augmentation in Eval)
            logits = ema_model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * len(batch_idx)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(batch_idx)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(self, epochs=EPOCHS, patience=7):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.evaluate(self.val_wavs, self.val_lbls)

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d} | Time: {elapsed:.1f}s | LR: {lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                no_improve_count = 0
                torch.save(self.ema.get_model().state_dict(), self.best_model_path)
                print(f"  -> New Best Model Saved! (Acc: {self.best_acc:.6f})")
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training finished. Best Validation Accuracy: {self.best_acc:.6f}")

    @torch.no_grad()
    def predict_submission(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Generating submission...")

        # Load best weights
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            # Update EMA wrapper to reflect loaded weights (though we could just use self.model in eval)
            self.ema = ModelEMA(self.model)
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        model = self.ema.get_model()
        model.eval()

        all_preds = []

        # Process Test Set
        num_samples = self.test_wavs.size(0)
        indices = torch.arange(num_samples, device=self.device)
        batches = torch.split(indices, BATCH_SIZE)

        for batch_idx in batches:
            x = self.test_wavs[batch_idx]
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Load Test Metadata to get filenames
        df_test = pd.read_csv(TEST_CSV)

        # If predictions are a subset (e.g. debug mode), slice metadata to match
        if len(all_preds) < len(df_test):
            print(
                f"Slicing test metadata from {len(df_test)} to {len(all_preds)} to match predictions."
            )
            df_test = df_test.iloc[: len(all_preds)]

        # Decode Labels
        # ID2LABEL is {0: 'yes', ...}
        pred_labels = [ID2LABEL[p] for p in all_preds]

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"fname": df_test["fname"], "label": pred_labels})

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(df_sub.head())
