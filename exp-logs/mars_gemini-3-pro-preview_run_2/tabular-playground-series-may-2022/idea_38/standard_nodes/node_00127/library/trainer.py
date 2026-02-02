import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.model import HybridSwiGLUResFunnel
from library.dataset import get_dataloaders
from library.utils import seed_everything, compute_auc, get_optimizer_params


class Trainer:
    """
    Trainer class to manage training, validation, and inference loops.
    """

    def __init__(
        self, model, device, optimizer, scheduler, criterion, checkpoint_dir="./working"
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for batch in train_loader:
            cont = batch["continuous"].to(self.device)
            cat = batch["categorical"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(cont, cat)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        preds = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                cont = batch["continuous"].to(self.device)
                cat = batch["categorical"].to(self.device)
                target = batch["target"].to(self.device)

                pred = self.model(cont, cat)
                loss = self.criterion(pred, target)
                val_loss += loss.item()

                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        auc = compute_auc(targets, preds)
        return val_loss / len(val_loader), auc

    def fit(self, train_loader, val_loader, epochs, patience=5):
        best_auc = 0.0
        patience_counter = 0

        print("Starting training...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if self.scheduler:
                self.scheduler.step()

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Validation AUC: {best_auc}")
        return best_auc

    def predict(self, test_loader):
        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                cont = batch["continuous"].to(self.device)
                cat = batch["categorical"].to(self.device)
                pred = self.model(cont, cat)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)


def run_training_experiment(
    epochs=40, batch_size=1024, patience=10, load_cached_data=True
):
    """
    Main function to setup and run the training experiment.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, num_workers=4
    )

    # 2. Model Setup
    model = PostNormConformerSwiGLU().to(device)

    # 3. Optimizer (Decoupled Weight Decay)
    optimizer_params = get_optimizer_params(model, weight_decay=1e-2)
    optimizer = optim.AdamW(optimizer_params, lr=1e-3)

    # 4. Scheduler (Step Decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 5. Criterion
    criterion = nn.BCELoss()

    # 6. Training
    trainer = Trainer(model, device, optimizer, scheduler, criterion)
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # 7. Inference
    print("Starting inference on test set...")
    predictions = trainer.predict(test_loader)

    # 8. Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    df_sub = pd.DataFrame({"id": test_ids, "target": predictions})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
