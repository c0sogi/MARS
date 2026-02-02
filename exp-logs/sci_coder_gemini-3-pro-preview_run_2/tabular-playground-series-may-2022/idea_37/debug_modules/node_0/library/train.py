import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.model import HybridSwiGLUNet


class Trainer:
    def __init__(self, device):
        self.device = device
        self.model = HybridSwiGLUNet().to(device)
        self.criterion = nn.BCELoss()
        self.best_auc = 0.0

        # Configure Optimizer with strict decoupled weight decay
        opt_params = self.model.get_optimizer_params(
            weight_decay=Config.WEIGHT_DECAY_PARAMS,
            weight_decay_bias_norm=Config.WEIGHT_DECAY_BIAS_NORM,
        )
        self.optimizer = optim.AdamW(opt_params, lr=Config.LEARNING_RATE)

        # Configure Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP_SIZE,
            gamma=Config.SCHEDULER_GAMMA,
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            cont_x = batch["cont_features"].to(self.device)
            cat_x = batch["cat_features"].to(self.device)
            targets = batch["target"].to(self.device).view(-1, 1)

            self.optimizer.zero_grad()
            outputs = self.model(cont_x, cat_x)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * cont_x.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                cont_x = batch["cont_features"].to(self.device)
                cat_x = batch["cat_features"].to(self.device)
                targets = batch["target"].to(self.device).view(-1, 1)

                outputs = self.model(cont_x, cat_x)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * cont_x.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        auc = compute_auc(all_targets, all_preds)
        avg_loss = total_loss / len(val_loader.dataset)

        return avg_loss, auc

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Step the scheduler
            self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                print(f"New best AUC! Saving model to {Config.MODEL_PATH}")
                torch.save(self.model.state_dict(), Config.MODEL_PATH)

        print(f"Training complete. Best Validation AUC: {self.best_auc}")

    def predict(self, test_loader):
        print(f"Loading best model from {Config.MODEL_PATH} for inference...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in test_loader:
                cont_x = batch["cont_features"].to(self.device)
                cat_x = batch["cat_features"].to(self.device)
                ids = batch["id"].numpy()

                outputs = self.model(cont_x, cat_x)
                preds = outputs.cpu().numpy().flatten()

                for id_val, pred_val in zip(ids, preds):
                    results.append({"id": int(id_val), "target": pred_val})

        return pd.DataFrame(results)


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model & Training
    trainer = Trainer(device)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Inference
    submission_df = trainer.predict(test_loader)

    # 5. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    run_training()
