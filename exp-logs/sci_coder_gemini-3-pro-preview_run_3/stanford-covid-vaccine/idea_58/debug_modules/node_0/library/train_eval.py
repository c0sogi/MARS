import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RNADataset
from library.model import HCSDBiGRU
from library.loss_metric import MCRMSELoss, compute_competition_metric


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = MCRMSELoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )

        self.best_score = float("inf")
        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(self.device)
            bpp_indices = batch["bpp_indices"].to(self.device)
            bpp_mask = batch["bpp_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features, bpp_indices, bpp_mask)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient Clipping (Mandatory for hybrid architecture)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                bpp_indices = batch["bpp_indices"].to(self.device)
                bpp_mask = batch["bpp_mask"].to(self.device)
                targets = batch[
                    "targets"
                ]  # Keep on CPU for metric calc later or move back

                outputs = self.model(features, bpp_indices, bpp_mask)

                all_preds.append(outputs.cpu())
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute competition metric (MCRMSE on 3 scored columns)
        score = compute_competition_metric(all_preds, all_targets)
        return score

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            self.scheduler.step()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE (Scored): {val_score} | "
                f"Time: {duration:.2f}s"
            )

            # Early Stopping / Model Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score: {self.best_score}")

        print(f"Training complete. Best Validation Score: {self.best_score}")


def run_training(max_samples=None, epochs=None):
    """
    Main entry point to run the training pipeline.

    Args:
        max_samples (int, optional): Limit dataset size for debugging.
        epochs (int, optional): Override config epochs.
    """
    set_seed(Config.SEED)

    # Initialize Datasets
    train_dataset = RNADataset(
        mode="train", load_cached_data=True, max_samples=max_samples
    )
    val_dataset = RNADataset(mode="val", load_cached_data=True, max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    device = torch.device(Config.DEVICE)
    model = HCSDBiGRU().to(device)

    # Initialize Trainer
    trainer = Trainer(model, device, Config)

    # Run Training
    num_epochs = epochs if epochs is not None else Config.NUM_EPOCHS
    trainer.fit(train_loader, val_loader, num_epochs)


def generate_submission(max_samples=None):
    """
    Generates the submission file using the best trained model.
    """
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    model = HCSDBiGRU().to(device)

    # Load best weights
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_dataset = RNADataset(
        mode="test", load_cached_data=True, max_samples=max_samples
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            ids = batch["ids"]

            outputs = model(features, bpp_indices, bpp_mask)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission DataFrame
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []
    target_cols = Config.TARGET_COLS

    seq_len = Config.SEQ_LENGTH

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(target_cols):
                row_dict[col] = float(row_preds[j])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Save Submission
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
