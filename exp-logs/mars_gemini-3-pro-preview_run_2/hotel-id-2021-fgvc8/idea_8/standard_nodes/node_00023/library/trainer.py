import os
import time
import torch
import numpy as np
from library.utils import AverageMeter, calc_map5


class Trainer:
    """
    Trainer class for Hotel ID Recognition.
    Manages the training loop, validation loop, optimization, and checkpointing.
    Designed to work with the Dual-Backbone Ensemble and Progressive Resolution strategy.
    """

    def __init__(
        self,
        model,
        device,
        optimizer,
        scheduler,
        criterion,
        checkpoint_path,
    ):
        """
        Args:
            model (nn.Module): The HotelRecognitionModel.
            device (torch.device): Compute device (CPU or CUDA).
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            criterion (nn.Module): Loss function (e.g., SubCenterArcFaceLoss).
            checkpoint_path (str): Path to save the best model weights.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint_path = checkpoint_path
        self.best_score = 0.0

    def train_one_epoch(self, dataloader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with labels:
            # The model head applies the ArcFace margin penalty to the target class logits.
            logits = self.model(images, labels)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

        # Step the scheduler at the end of the epoch
        if self.scheduler is not None:
            self.scheduler.step()

        elapsed = time.time() - start_time
        # Printing full precision as requested for metrics not strictly required here,
        # but good for debugging.
        print(f"Epoch {epoch} | Train Loss: {loss_meter.avg} | Time: {elapsed:.2f}s")
        return loss_meter.avg

    def validate(self, dataloader):
        """
        Runs validation and calculates MAP@5.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                # Forward pass WITHOUT labels:
                # The model returns raw cosine similarities (scaled by s) without margin penalty.
                # This represents the inference-time ranking capability.
                logits = self.model(images, labels=None)

                # Get top 5 predictions
                # logits shape: (Batch, Num_Classes)
                _, top_indices = torch.topk(logits, k=5, dim=1)

                # Collect predictions and targets
                all_preds.extend(top_indices.cpu().numpy().tolist())
                all_targets.extend(labels.tolist())

        # Calculate MAP@5
        map5 = calc_map5(all_preds, all_targets)

        # Print full precision as requested
        print(f"Validation MAP@5: {map5}")
        return map5

    def fit(self, train_loader, val_loader, epochs, patience=5):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Number of epochs to train.
            patience (int): Early stopping patience.

        Returns:
            float: The best MAP@5 score achieved.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            _ = self.train_one_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            # Checkpoint logic
            if val_score > self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Score did not improve. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model weights before returning
        if os.path.exists(self.checkpoint_path):
            print(f"Loading best model from {self.checkpoint_path}")
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )

        return self.best_score
