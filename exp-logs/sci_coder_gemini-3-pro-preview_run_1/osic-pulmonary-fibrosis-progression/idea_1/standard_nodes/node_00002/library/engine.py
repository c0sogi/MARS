import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, LaplaceLogLikelihoodLoss


class Trainer:
    """
    Encapsulates the training, evaluation, and inference logic for the Lung Function Decline model.
    """

    def __init__(self, model, optimizer, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = LaplaceLogLikelihoodLoss()
        self.best_val_loss = float("inf")

    def train_one_epoch(self, train_loader, epoch):
        """
        Performs one epoch of training.
        """
        self.model.train()
        meter = AverageMeter()

        for batch in train_loader:
            # Move inputs to device
            images = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            base_fvc = batch["base_fvc"].to(self.device)
            weeks = batch["weeks"].to(self.device)
            true_fvc = batch["fvc_true"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass: Predict Slope and Confidence
            pred_slope, pred_conf = self.model(images, tabular)

            # Calculate Loss
            loss = self.criterion(pred_slope, pred_conf, base_fvc, weeks, true_fvc)

            # Backward pass and Optimization
            loss.backward()
            self.optimizer.step()

            # Update metrics
            meter.update(loss.item(), images.size(0))

        return meter.avg

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                base_fvc = batch["base_fvc"].to(self.device)
                weeks = batch["weeks"].to(self.device)
                true_fvc = batch["fvc_true"].to(self.device)

                # Forward pass
                pred_slope, pred_conf = self.model(images, tabular)

                # Calculate Loss
                loss = self.criterion(pred_slope, pred_conf, base_fvc, weeks, true_fvc)

                meter.update(loss.item(), images.size(0))

        return meter.avg

    def fit(self, train_loader, val_loader, epochs, patience):
        """
        Runs the full training loop with Early Stopping.
        """
        print(f"Starting training: Epochs={epochs}, Patience={patience}")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)

            # Print full precision metrics
            print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}")

            # Update Scheduler (assuming ReduceLROnPlateau based on Config)
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load the best model state for subsequent prediction
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves them to submission.csv.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                base_fvc = batch["base_fvc"].to(self.device)
                weeks = batch["weeks"].to(self.device)
                patient_ids = batch["patient_id"]

                # Forward pass
                pred_slope, pred_conf = self.model(images, tabular)

                # Flatten tensors for processing
                pred_slope = pred_slope.view(-1)
                pred_conf = pred_conf.view(-1)
                base_fvc = base_fvc.view(-1)
                weeks = weeks.view(-1)

                # Reconstruct FVC prediction: Baseline + (Slope * Time)
                pred_fvc = base_fvc + (pred_slope * weeks)

                # Convert to numpy
                p_fvc = pred_fvc.cpu().numpy()
                p_conf = pred_conf.cpu().numpy()
                p_weeks = weeks.cpu().numpy()

                # Collect results
                for i in range(len(patient_ids)):
                    results.append(
                        {
                            "Patient_Week": f"{patient_ids[i]}_{int(p_weeks[i])}",
                            "FVC": p_fvc[i],
                            "Confidence": p_conf[i],
                        }
                    )

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
