import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    seed_everything,
    weighted_log_loss,
    AverageMeter,
    load_or_generate_cache,
)
from library.data import get_loaders, get_test_loader
from library.model import DualAttentionNetwork
from library.loss import TriLevelLoss


class Trainer:
    """
    Manages the training and validation lifecycle of the 2.5D Dual-Attention Network.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = DualAttentionNetwork(config)
        self.model.to(self.device)

        # Initialize Loss
        self.criterion = TriLevelLoss(config)
        self.criterion.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.SCHEDULER_T_MAX,
            eta_min=config.SCHEDULER_MIN_LR,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Tracking
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training with Gradient Accumulation and Mixed Precision.
        """
        self.model.train()
        losses = AverageMeter()

        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move data to device
            images = batch["image"].to(self.device, non_blocking=True)

            targets = {
                "label_study": batch["label_study"].to(self.device, non_blocking=True),
                "label_slice": batch["label_slice"].to(self.device, non_blocking=True),
                "label_spatial": batch["label_spatial"].to(
                    self.device, non_blocking=True
                ),
            }

            batch_size = images.size(0)

            # Forward pass with Automatic Mixed Precision
            with autocast():
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                # Normalize loss for gradient accumulation
                loss = loss / self.config.ACCUMULATION_STEPS

            # Backward pass
            self.scaler.scale(loss).backward()

            # Optimization Step (only every N steps)
            if (step + 1) % self.config.ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update metrics (multiply back to get actual loss value)
            losses.update(loss.item() * self.config.ACCUMULATION_STEPS, batch_size)

        return losses.avg

    def validate(self, val_loader):
        """
        Runs validation and computes both the composite loss and the competition metric.
        """
        self.model.eval()
        losses = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device, non_blocking=True)

                targets = {
                    "label_study": batch["label_study"].to(
                        self.device, non_blocking=True
                    ),
                    "label_slice": batch["label_slice"].to(
                        self.device, non_blocking=True
                    ),
                    "label_spatial": batch["label_spatial"].to(
                        self.device, non_blocking=True
                    ),
                }

                batch_size = images.size(0)

                # Forward pass
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)

                losses.update(loss.item(), batch_size)

                # Collect predictions for metric calculation
                # Apply sigmoid to study logits
                all_preds.append(torch.sigmoid(preds["study_logits"]).cpu())
                all_targets.append(targets["label_study"].cpu())

        # Check for empty validation set
        if len(all_preds) == 0:
            return 0.0, 0.0

        # Compute Weighted Log Loss (Competition Metric)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Construct weights matching the target columns
        metric_weights = torch.ones(len(self.config.TARGET_COLS))
        if "patient_overall" in self.config.TARGET_COLS:
            idx = self.config.TARGET_COLS.index("patient_overall")
            metric_weights[idx] = 7.0

        metric = weighted_log_loss(
            all_targets, all_preds, weights=metric_weights
        ).item()

        return losses.avg, metric

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        seed_everything(self.config.SEED)

        print(f"Initializing Data Loaders...")
        train_loader, val_loader = get_loaders()

        print(f"Starting Training on {self.device} for {self.config.EPOCHS} epochs.")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metric = self.validate(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Metric: {val_metric:.6f}"
            )

            # Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                save_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  [+] Saved best model to {save_path}")
            else:
                self.patience_counter += 1
                print(
                    f"  [!] No improvement. Patience: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            # Early Stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model for potential immediate use
        best_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        return self.best_loss


def inference(config=Config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device(config.DEVICE)
    weights_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return

    print("Loading model for inference...")
    model = DualAttentionNetwork(config)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print("Loading test data...")
    test_loader = get_test_loader()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            with autocast():
                outputs = model(images)
                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs["study_logits"]).cpu().numpy()
                all_preds.append(probs)

    if not all_preds:
        print("No predictions generated.")
        return

    all_preds = np.concatenate(all_preds, axis=0)

    # --- Format Submission ---

    # 1. Get Study UIDs (order matches loader)
    test_meta = pd.read_csv(config.TEST_METADATA)
    study_uids = test_meta["StudyInstanceUID"].values

    if len(study_uids) != len(all_preds):
        print(
            f"Warning: Count mismatch. Metadata: {len(study_uids)}, Preds: {len(all_preds)}"
        )

    # 2. Create DataFrame with predictions
    pred_df = pd.DataFrame(all_preds, columns=config.TARGET_COLS)
    pred_df["StudyInstanceUID"] = study_uids

    # 3. Melt to long format (StudyUID_Target)
    melted = pred_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=config.TARGET_COLS,
        var_name="target_name",
        value_name="fractured",
    )

    # Create row_id
    melted["row_id"] = melted["StudyInstanceUID"] + "_" + melted["target_name"]

    # 4. Merge with sample submission to ensure correct row order and existence
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION)

    submission = sample_sub[["row_id"]].merge(
        melted[["row_id", "fractured"]], on="row_id", how="left"
    )

    # Fill missing values (safeguard)
    submission["fractured"] = submission["fractured"].fillna(0.5)

    # 5. Save
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {config.SUBMISSION_PATH}")
