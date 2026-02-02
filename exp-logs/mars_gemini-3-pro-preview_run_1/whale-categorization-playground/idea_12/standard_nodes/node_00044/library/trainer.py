import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from library.config import Config
from library.loss import ArcFaceLoss
from library.utils import calculate_map5


class Trainer:
    """
    Trainer class to manage the training and validation of a single model instance.
    """

    def __init__(
        self, model, train_loader, val_loader, num_classes, device, model_name
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_classes (int): Total number of classes (identities).
            device (str): 'cuda' or 'cpu'.
            model_name (str): Identifier for the model (used for checkpoint naming).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.model_name = model_name

        # Initialize ArcFace Loss
        # We must optimize the class centers (weights) inside this module.
        # The loss module needs to be moved to the correct device.
        self.criterion = ArcFaceLoss(num_classes=self.num_classes).to(self.device)

        # Optimizer: AdamW
        # We optimize both the backbone/neck parameters AND the ArcFace class centers
        self.optimizer = optim.AdamW(
            [
                {"params": self.model.parameters()},
                {"params": self.criterion.parameters()},
            ],
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.MAX_EPOCHS, eta_min=Config.MIN_LR
        )

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        self.criterion.train()

        running_loss = 0.0
        count = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: Get embeddings from model
            embeddings = self.model(images)

            # Loss pass: Calculate ArcFace loss
            loss = self.criterion(embeddings, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        return epoch_loss

    def validate(self):
        """
        Runs validation with Test-Time Augmentation (Horizontal Flip).
        Calculates MAP@5.
        """
        self.model.eval()
        self.criterion.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            # Retrieve learned class centers from ArcFace module and normalize them
            # Weight shape: [NumClasses, EmbeddingSize]
            class_centers = F.normalize(self.criterion.weight, p=2, dim=1)

            for images, labels in self.val_loader:
                images = images.to(self.device)

                # --- TTA: Original View ---
                emb_orig = self.model(images)
                emb_orig_norm = F.normalize(emb_orig, p=2, dim=1)
                # Compute cosine similarity (logits)
                logits_orig = F.linear(emb_orig_norm, class_centers)

                # --- TTA: Flipped View ---
                # Flip width dimension (dim 3 for NCHW)
                images_flip = torch.flip(images, dims=[3])
                emb_flip = self.model(images_flip)
                emb_flip_norm = F.normalize(emb_flip, p=2, dim=1)
                logits_flip = F.linear(emb_flip_norm, class_centers)

                # --- Ensemble Views ---
                # Average the logits from both views
                avg_logits = (logits_orig + logits_flip) / 2.0

                # --- Prediction ---
                # Get top 5 indices
                _, top_indices = torch.topk(avg_logits, k=5, dim=1)

                # Store results
                # top_indices is [Batch, 5]
                all_preds.extend(top_indices.cpu().numpy().tolist())
                all_targets.extend(labels.numpy().tolist())

        # Calculate MAP@5
        score = calculate_map5(all_preds, all_targets)
        return score

    def train_until_convergence(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {self.model_name}...")

        best_score = -1.0
        patience_counter = 0

        # Ensure checkpoint directory exists
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Update Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics (Full precision for validation score)
            print(
                f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MAP@5: {val_score} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early Stopping & Checkpointing
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0

                # Save Best Model
                save_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"{self.model_name}_best.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "criterion_state_dict": self.criterion.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_score": best_score,
                    },
                    save_path,
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Score: {best_score}"
                )
                break

        return best_score
