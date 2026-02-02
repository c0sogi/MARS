import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_logger, calculate_auc
from library.dataset import WhaleDataset, prepare_data
from library.model import DualStreamEfficientNet

# -----------------------------------------------------------------------------
# Mixup Utilities
# -----------------------------------------------------------------------------


def mixup_data(x1, x2, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to both streams synchronously.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(device)

    # Apply same mixing to both streams
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]

    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss for mixed inputs.
    """
    # y_a and y_b need to be unsqueezed to match pred shape (B, 1) if using BCEWithLogitsLoss
    return lam * criterion(pred, y_a.unsqueeze(1)) + (1 - lam) * criterion(
        pred, y_b.unsqueeze(1)
    )


# -----------------------------------------------------------------------------
# Trainer Class
# -----------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = get_logger("Trainer")
        self.best_auc = 0.0

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for x1, x2, y in self.train_loader:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device)

            current_batch_size = x1.size(0)

            # Apply Synchronized Mixup
            x1_mix, x2_mix, y_a, y_b, lam = mixup_data(
                x1, x2, y, self.config.MIXUP_ALPHA, self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x1_mix, x2_mix)

            # Compute Loss
            loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * current_batch_size
            dataset_size += current_batch_size

        return running_loss / dataset_size if dataset_size > 0 else 0.0

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x1, x2, y in self.val_loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                y = y.to(self.device)

                current_batch_size = x1.size(0)

                outputs = self.model(x1, x2)
                loss = self.criterion(outputs, y.unsqueeze(1))

                running_loss += loss.item() * current_batch_size
                dataset_size += current_batch_size

                # Apply sigmoid for probabilities
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        # Calculate AUC
        # Flatten preds to match target shape if necessary
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets)

        auc_score = calculate_auc(all_targets, all_preds)

        return avg_loss, auc_score

    def fit(self):
        patience_counter = 0
        best_model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")

        self.logger.info(f"Starting training on {self.device}...")

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_one_epoch()
            val_loss, val_auc = self.validate()

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc} | "
                f"Time: {elapsed}s"
            )

            # Checkpoint & Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), best_model_path)
                patience_counter = 0
                self.logger.info(f"  -> New Best AUC! Model saved to {best_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= self.config.PATIENCE:
                self.logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

        self.logger.info(f"Training complete. Best Val AUC: {self.best_auc}")
        return best_model_path


# -----------------------------------------------------------------------------
# Main Execution Functions
# -----------------------------------------------------------------------------


def run_training(config=Config, load_cached_data=True):
    """
    Orchestrates the training pipeline.
    """
    seed_everything(config.SEED)
    Config.setup()
    device = torch.device(config.DEVICE)
    logger = get_logger("TrainingPipeline")

    # Load Metadata
    train_csv_path = os.path.join(config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(config.METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Prepare Data
    logger.info("Preparing Training Data...")
    X1_train, X2_train, Y_train, _ = prepare_data(
        train_df, config, cache_name="train", load_cached_data=load_cached_data
    )

    logger.info("Preparing Validation Data...")
    X1_val, X2_val, Y_val, _ = prepare_data(
        val_df, config, cache_name="val", load_cached_data=load_cached_data
    )

    # Datasets
    train_dataset = WhaleDataset(X1_train, X2_train, Y_train, augment=True)
    val_dataset = WhaleDataset(X1_val, X2_val, Y_val, augment=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Initialization
    model = DualStreamEfficientNet(config).to(device)

    # Loss Function (Weighted BCE)
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    best_model_path = trainer.fit()
    return best_model_path


def generate_submission(model_path, config=Config, load_cached_data=True):
    """
    Generates predictions for the test set using the trained model.
    """
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    logger = get_logger("Inference")

    # Load Test Metadata
    test_csv_path = os.path.join(config.METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    logger.info("Preparing Test Data...")
    X1_test, X2_test, Y_test, clips = prepare_data(
        test_df, config, cache_name="test", load_cached_data=load_cached_data
    )

    test_dataset = WhaleDataset(X1_test, X2_test, Y_test, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = DualStreamEfficientNet(config).to(device)
    logger.info(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    all_preds = []
    logger.info("Running Inference...")

    with torch.no_grad():
        for x1, x2, _ in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            outputs = model(x1, x2)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds).flatten()

    # Save Submission
    submission = pd.DataFrame({"clip": clips, "probability": all_preds})

    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")
