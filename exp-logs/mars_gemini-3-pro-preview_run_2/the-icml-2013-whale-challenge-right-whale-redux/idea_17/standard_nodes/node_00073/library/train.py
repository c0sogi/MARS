import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, get_device, seed_everything
from library.dataset import get_dataloaders
from library.models import WhaleModel


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Flatten output and target for BCEWithLogitsLoss
        loss = criterion(output.view(-1), target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

        # Store predictions and targets for AUC calculation
        # Sigmoid applied here for metric calculation only
        preds = torch.sigmoid(output).detach().cpu().numpy()
        targets = target.detach().cpu().numpy()

        all_targets.extend(targets)
        all_preds.extend(preds)

    epoch_auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
    print(f"Train Epoch: {epoch} Loss: {losses.avg:.6f} AUC: {epoch_auc:.6f}")

    return losses.avg, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output.view(-1), target)

            losses.update(loss.item(), data.size(0))

            preds = torch.sigmoid(output).cpu().numpy()
            targets = target.cpu().numpy()

            all_targets.extend(targets)
            all_preds.extend(preds)

    val_auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
    print(f"Validation Loss: {losses.avg:.6f} AUC: {val_auc}")

    return losses.avg, val_auc


class Trainer:
    """
    Manages the training, validation, and prediction for a single model.
    """

    def __init__(self, model_name, train_loader, val_loader, device, debug=False):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.debug = debug
        self.epochs = Config.EPOCHS if not debug else 2

        # Initialize model
        self.model = WhaleModel(model_name, pretrained=True).to(device)

        # Optimization
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        self.best_model_path = os.path.join(
            Config.WORKING_DIR, f"{self.model_name}_best.pth"
        )

    def fit(self):
        print(f"\nStarting training for model: {self.model_name}")
        best_auc = 0.0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_auc = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch,
            )
            val_loss, val_auc = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            self.scheduler.step()

            # Early Stopping based on Validation AUC
            if val_auc > best_auc:
                print(
                    f"AUC improved from {best_auc} to {val_auc}. Saving model to {self.best_model_path}"
                )
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished for {self.model_name}. Best AUC: {best_auc}")
        return best_auc

    def predict(self, loader):
        """
        Loads the best checkpoint and generates predictions.
        """
        # Load best weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print(
                f"Warning: Best model not found for {self.model_name}, using current weights."
            )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds).flatten()


def main(debug=False):
    """
    Main execution function to train the ensemble and generate submission.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # Get DataLoaders
    # load_cached_data=True ensures we use caching logic from dataset.py
    train_loader, val_loader, test_loader, test_clips = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # Collect validation targets for meta-learner training
    # Iterate val_loader to ensure order matches predictions
    val_targets = []
    for _, target in val_loader:
        val_targets.extend(target.numpy())
    val_targets = np.array(val_targets)

    val_meta_features = []
    test_meta_features = []

    # Train each base learner in the ensemble
    for model_name in Config.MODEL_NAMES:
        trainer = Trainer(model_name, train_loader, val_loader, device, debug=debug)
        trainer.fit()

        # Generate predictions for stacking
        print(f"Generating predictions for {model_name}...")
        val_preds = trainer.predict(val_loader)
        test_preds = trainer.predict(test_loader)

        val_meta_features.append(val_preds)
        test_meta_features.append(test_preds)

    # Stack features: (N_samples, N_models)
    X_val = np.column_stack(val_meta_features)
    X_test = np.column_stack(test_meta_features)
    y_val = val_targets

    # Train Meta-Learner (Logistic Regression)
    print("\nTraining Meta-Learner (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=Config.SEED)
    meta_model.fit(X_val, y_val)

    # Validate Meta-Learner
    val_meta_preds = meta_model.predict_proba(X_val)[:, 1]
    meta_auc = calculate_roc_auc(y_val, val_meta_preds)
    print(f"Meta-Learner Validation AUC: {meta_auc}")

    # Generate Final Test Predictions
    final_test_preds = meta_model.predict_proba(X_test)[:, 1]

    # Create Submission File
    submission = pd.DataFrame({"clip": test_clips, "probability": final_test_preds})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
