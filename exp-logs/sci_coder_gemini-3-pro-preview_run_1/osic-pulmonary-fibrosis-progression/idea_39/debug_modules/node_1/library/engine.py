import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import PAVENet


class Trainer:
    def __init__(self):
        seed_everything(Config.SEED)
        self.device = Config.DEVICE

        # Initialize Model
        self.model = PAVENet().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Loss Function
        self.criterion = LaplaceLogLikelihoodLoss()

        # State tracking
        self.best_score = -np.inf
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(self.device)
            img_cor = batch["image_coronal"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            anchor = batch["anchor"].to(self.device)
            weeks = batch["weeks"].to(self.device)
            target = batch["target"].to(self.device)
            raw_base_fvc = batch["raw_base_fvc"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            fvc_pred, sigma_pred = self.model(
                img_ax, img_cor, tabular, anchor, weeks, raw_base_fvc
            )

            # Loss calculation
            loss = self.criterion(fvc_pred, sigma_pred, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * img_ax.size(0)

        return running_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        all_true = []
        all_pred = []
        all_conf = []

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["image_axial"].to(self.device)
                img_cor = batch["image_coronal"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                anchor = batch["anchor"].to(self.device)
                weeks = batch["weeks"].to(self.device)
                target = batch["target"].to(self.device)
                raw_base_fvc = batch["raw_base_fvc"].to(self.device)

                fvc_pred, sigma_pred = self.model(
                    img_ax, img_cor, tabular, anchor, weeks, raw_base_fvc
                )

                loss = self.criterion(fvc_pred, sigma_pred, target)
                running_loss += loss.item() * img_ax.size(0)

                # Collect for metric calculation
                all_true.extend(target.cpu().numpy().flatten())
                all_pred.extend(fvc_pred.cpu().numpy().flatten())
                all_conf.extend(sigma_pred.cpu().numpy().flatten())

        avg_loss = running_loss / len(val_loader.dataset)
        metric_score = calculate_metric(all_true, all_pred, all_conf)

        return avg_loss, metric_score

    def run(self):
        print(f"Starting training on device: {self.device}")
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_metric = self.validate(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Metric: {val_metric}"
            )

            # Checkpoint & Early Stopping
            if val_metric > self.best_score:
                print(
                    f"Score Improved ({self.best_score} -> {val_metric}). Saving model..."
                )
                self.best_score = val_metric
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {patience_counter} epochs.")
                    break

        print(f"Training complete. Best Metric: {self.best_score}")

    def inference(self):
        print("Starting inference...")
        # Load best model
        if not os.path.exists(self.best_model_path):
            print("No best model found. Please train first.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        _, _, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        # Load test metadata to align predictions
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

        predictions = []
        confidences = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["image_axial"].to(self.device)
                img_cor = batch["image_coronal"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                anchor = batch["anchor"].to(self.device)
                weeks = batch["weeks"].to(self.device)
                raw_base_fvc = batch["raw_base_fvc"].to(self.device)

                fvc_pred, sigma_pred = self.model(
                    img_ax, img_cor, tabular, anchor, weeks, raw_base_fvc
                )

                predictions.extend(fvc_pred.cpu().numpy().flatten())
                confidences.extend(sigma_pred.cpu().numpy().flatten())

        # Create submission dataframe
        submission = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": predictions,
                "Confidence": confidences,
            }
        )

        # Ensure correct types
        submission["FVC"] = submission["FVC"].astype(float)
        submission["Confidence"] = submission["Confidence"].astype(float)

        # Save
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    Config.setup()
    trainer = Trainer()
    trainer.run()
    trainer.inference()


if __name__ == "__main__":
    # This block is here for local testing if needed,
    # but the instructions say "Only implement the module class/functions".
    # The main execution logic is encapsulated in the functions above.
    pass
