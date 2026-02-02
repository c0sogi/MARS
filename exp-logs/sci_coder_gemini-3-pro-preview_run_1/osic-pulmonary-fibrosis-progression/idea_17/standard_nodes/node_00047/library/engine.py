import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, compute_metric, log_metrics, save_checkpoint


class Engine:
    """
    Encapsulates the training, evaluation, and inference logic for the
    Full-Fidelity Concatenated Dual-Axis Network.
    """

    def __init__(self, model, optimizer, device=Config.DEVICE, scheduler=None):
        """
        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): Compute device ('cuda' or 'cpu').
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_score = -float("inf")

    def _process_batch(self, batch):
        """
        Moves batch data to the configured device.
        """
        return {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }

    def train_one_epoch(self, train_loader, criterion, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            batch = self._process_batch(batch)

            # Unpack inputs
            # The model forward expects the whole batch dict to handle multiple inputs
            outputs = self.model(batch)

            targets = batch["target"]
            meta = batch["meta"]

            # Compute Loss
            loss = criterion(outputs, targets, meta)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), n=targets.size(0))

        log_metrics({"Train Loss": loss_meter.avg}, prefix=f"Epoch {epoch}")
        return loss_meter.avg

    def evaluate(self, val_loader, criterion, epoch=None):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                batch = self._process_batch(batch)

                outputs = self.model(batch)
                targets = batch["target"]
                meta = batch["meta"]

                # Compute Loss
                loss = criterion(outputs, targets, meta)
                loss_meter.update(loss.item(), n=targets.size(0))

                # Reconstruct Predictions for Metric Calculation
                # outputs: [alpha, sigma_base, sigma_growth]
                # meta: [base_fvc, dt]
                alpha = outputs[:, 0]
                sigma_base = outputs[:, 1]
                sigma_growth = outputs[:, 2]

                base_fvc = meta[:, 0]
                dt = meta[:, 1]

                pred_fvc = base_fvc + alpha * dt
                pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

                # Compute Metric
                # Note: compute_metric handles clipping internally
                score = compute_metric(targets, pred_fvc, pred_sigma)
                metric_meter.update(score, n=targets.size(0))

        prefix = f"Epoch {epoch} Validation" if epoch is not None else "Validation"
        log_metrics(
            {"Val Loss": loss_meter.avg, "Val Metric": metric_meter.avg}, prefix=prefix
        )

        return metric_meter.avg

    def fit(
        self,
        train_loader,
        val_loader,
        criterion,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    ):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # Train
            self.train_one_epoch(train_loader, criterion, epoch)

            # Evaluate
            val_score = self.evaluate(val_loader, criterion, epoch)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            # Checkpoint & Early Stopping
            # Metric is negative Laplace Log Likelihood (Higher is better)
            if val_score > self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                epochs_no_improve = 0

                # Construct state dict
                state = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_score": self.best_score,
                }
                save_checkpoint(state, is_best=True)
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Generating predictions for test set...")
        self.model.eval()

        patient_weeks = []
        fvc_preds = []
        conf_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = self._process_batch(batch)

                outputs = self.model(batch)
                meta = batch["meta"]
                p_weeks = batch["patient_week"]

                # Reconstruct Predictions
                alpha = outputs[:, 0]
                sigma_base = outputs[:, 1]
                sigma_growth = outputs[:, 2]

                base_fvc = meta[:, 0]
                dt = meta[:, 1]

                pred_fvc = base_fvc + alpha * dt
                pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

                # Explicitly clip confidence for submission as per requirements
                # "confidence values are clipped at 70 ml"
                pred_sigma = torch.clamp(pred_sigma, min=Config.MIN_CONFIDENCE)

                # Collect results
                patient_weeks.extend(p_weeks)
                fvc_preds.extend(pred_fvc.cpu().numpy())
                conf_preds.extend(pred_sigma.cpu().numpy())

        # Create DataFrame
        submission = pd.DataFrame(
            {"Patient_Week": patient_weeks, "FVC": fvc_preds, "Confidence": conf_preds}
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(submission.head())
