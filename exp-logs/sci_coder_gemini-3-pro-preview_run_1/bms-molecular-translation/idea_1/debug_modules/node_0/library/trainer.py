import time
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein, save_checkpoint


class Trainer:
    """
    Trainer class to manage training, validation, and prediction loops.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        device=Config.DEVICE,
        patience=Config.PATIENCE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        start_time = time.time()

        for batch_idx, (images, captions) in enumerate(self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # outputs shape: (batch, max_len, vocab_size)
            outputs = self.model(images, captions)

            # Calculate loss
            # We ignore the first token (<sos>) in outputs for loss calculation
            # outputs[:, t, :] predicts captions[:, t]
            # So outputs[:, 1:, :] predicts captions[:, 1:]

            # Flatten outputs and targets for CrossEntropyLoss
            # outputs: (batch, target_len, vocab) -> slice -> (batch, target_len-1, vocab) -> (N, vocab)
            output_dim = outputs.shape[-1]
            loss = self.criterion(
                outputs[:, 1:, :].reshape(-1, output_dim), captions[:, 1:].reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            # Optimizer step
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} - Train Loss: {losses.avg} - Time: {elapsed:.2f}s")
        return losses.avg

    def validate(self, epoch):
        """
        Runs validation loop, computing loss and Levenshtein distance.
        """
        self.model.eval()
        losses = AverageMeter()
        levenshtein_scores = AverageMeter()
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (images, captions) in enumerate(self.val_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                # 1. Calculate Validation Loss
                # Note: This uses the model's forward pass which might use teacher forcing
                # depending on implementation, but we use it for a loss estimate.
                outputs = self.model(images, captions)
                output_dim = outputs.shape[-1]
                loss = self.criterion(
                    outputs[:, 1:, :].reshape(-1, output_dim),
                    captions[:, 1:].reshape(-1),
                )
                losses.update(loss.item(), images.size(0))

                # 2. Calculate Levenshtein Distance (Metric)
                # Use greedy sampling for metric calculation
                preds = self.model.sample(images)

                # Decode strings
                preds_cpu = preds.cpu().numpy()
                captions_cpu = captions.cpu().numpy()

                for i in range(len(images)):
                    pred_str = self.tokenizer.decode(preds_cpu[i])
                    target_str = self.tokenizer.decode(captions_cpu[i])

                    score = compute_levenshtein(pred_str, target_str)
                    levenshtein_scores.update(score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} - Val Loss: {losses.avg} - Val Levenshtein: {levenshtein_scores.avg} - Time: {elapsed:.2f}s"
        )

        return losses.avg, levenshtein_scores.avg

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            _ = self.train_epoch(epoch)
            _, val_score = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step(val_score)

            # Checkpoint and Early Stopping
            is_best = val_score < self.best_score
            if is_best:
                print(
                    f"Validation score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                self.patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_metric": self.best_score,
                    },
                    is_best=True,
                )
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for a test loader.

        Args:
            test_loader: DataLoader for the test set.

        Returns:
            list: List of predicted InChI strings.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                # Use greedy sampling
                preds = self.model.sample(images)
                preds_cpu = preds.cpu().numpy()

                for i in range(len(images)):
                    pred_str = self.tokenizer.decode(preds_cpu[i])
                    predictions.append(pred_str)

        return predictions
