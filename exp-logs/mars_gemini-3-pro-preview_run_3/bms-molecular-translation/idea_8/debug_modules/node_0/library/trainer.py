import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import AverageMeter, compute_levenshtein
from library.config import Config
from library.tokenizer import Tokenizer


class Trainer:
    """
    Trainer class to manage model training, validation, and inference.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        config: Config,
        train_loader=None,
        val_loader=None,
        test_loader=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Loss function
        # We ignore the padding token index when calculating loss
        pad_idx = self.tokenizer.stoi[self.config.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr,
            verbose=True,
        )

        # State for early stopping
        self.best_metric = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        start_time = time.time()

        for batch_idx, (images, labels, lengths) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            # labels shape: (B, max_len)
            # outputs shape: (B, max_len, vocab_size)
            outputs = self.model(
                images, labels, teacher_forcing_ratio=self.config.teacher_forcing_ratio
            )

            # Calculate loss
            # We skip the first time step (SOS) in the targets, and the 0th index in outputs (which is 0 initialization)
            # outputs[:, t, :] predicts labels[:, t]
            # So we compare outputs[:, 1:, :] with labels[:, 1:]
            output_logits = outputs[:, 1:, :].reshape(-1, outputs.shape[-1])
            target_labels = labels[:, 1:].reshape(-1)

            loss = self.criterion(output_logits, target_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )

            self.optimizer.step()

            # Record loss
            losses.update(loss.item(), images.size(0))

            if (batch_idx + 1) % self.config.print_freq == 0:
                print(
                    f"Epoch: [{epoch + 1}][{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Time: {time.time() - start_time:.2f}s"
                )

        return losses.avg

    def validate(self):
        """
        Runs validation using greedy decoding and computes Levenshtein distance.
        """
        self.model.eval()
        predictions = []
        ground_truths = []

        sos_idx = self.tokenizer.stoi[self.config.sos_token]
        eos_idx = self.tokenizer.stoi[self.config.eos_token]

        with torch.no_grad():
            for images, labels, lengths in self.val_loader:
                images = images.to(self.device)

                # Generate predictions (indices)
                # Shape: (B, max_len)
                generated_indices = self.model.generate(
                    images,
                    max_len=self.config.max_text_length,
                    sos_token_idx=sos_idx,
                    eos_token_idx=eos_idx,
                )

                # Convert indices to strings
                for i in range(images.size(0)):
                    # Prediction
                    pred_seq = generated_indices[i].cpu().numpy()
                    pred_text = self.tokenizer.sequence_to_text(pred_seq)
                    predictions.append(pred_text)

                    # Ground Truth
                    gt_seq = labels[i].cpu().numpy()
                    gt_text = self.tokenizer.sequence_to_text(gt_seq)
                    ground_truths.append(gt_text)

        # Compute Metric
        score = compute_levenshtein(predictions, ground_truths)
        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.6f}")

            # Validate
            print("Validating...")
            val_score = self.validate()
            print(f"Validation Levenshtein Distance: {val_score}")

            # Scheduler Step
            self.scheduler.step(val_score)

            # Checkpoint & Early Stopping
            if val_score < self.best_metric:
                print(
                    f"Metric improved from {self.best_metric} to {val_score}. Saving model..."
                )
                self.best_metric = val_score
                self.epochs_no_improve = 0

                save_path = os.path.join(self.config.checkpoint_dir, "model_best.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                        "config": self.config.__dict__,  # Save config dict for reference
                    },
                    save_path,
                )
            else:
                self.epochs_no_improve += 1
                print(
                    f"No improvement. Patience: {self.epochs_no_improve}/{self.config.early_stopping_patience}"
                )

            if self.epochs_no_improve >= self.config.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_metric}")

    def predict(self):
        """
        Generates predictions for the test set and saves submission file.
        """
        print("\n--- Starting Inference on Test Set ---")

        # Load best model
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "model_best.pth")
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Best model not found at {checkpoint_path}. Using current model state."
            )
        else:
            print(f"Loading best model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])

        self.model.eval()

        all_preds = []
        all_image_ids = []

        sos_idx = self.tokenizer.stoi[self.config.sos_token]
        eos_idx = self.tokenizer.stoi[self.config.eos_token]

        with torch.no_grad():
            for batch_idx, (images, image_ids) in enumerate(self.test_loader):
                images = images.to(self.device)

                # Generate
                generated_indices = self.model.generate(
                    images,
                    max_len=self.config.max_text_length,
                    sos_token_idx=sos_idx,
                    eos_token_idx=eos_idx,
                )

                # Decode
                for i in range(images.size(0)):
                    pred_seq = generated_indices[i].cpu().numpy()
                    pred_text = self.tokenizer.sequence_to_text(pred_seq)

                    all_preds.append(pred_text)
                    all_image_ids.append(image_ids[i])

                if (batch_idx + 1) % self.config.print_freq == 0:
                    print(f"Inference: [{batch_idx + 1}/{len(self.test_loader)}]")

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"image_id": all_image_ids, "InChI": all_preds})

        # Save
        os.makedirs(self.config.submission_dir, exist_ok=True)
        submission_df.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")
        print(f"Total predictions generated: {len(submission_df)}")
