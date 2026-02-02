import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_NORM_CLIP,
    NUM_WORKERS,
    PIN_MEMORY,
    CLASS_NAMES,
    TRAIN_SUBSET_SIZE,
    VAL_SUBSET_SIZE,
)
from library.dataset import NuScenesDataset, custom_collate_fn
from library.model import PointPillarsResNetFPN
from library.loss import CenterLoss
from library.utils import decode_predictions


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Engine:
    def __init__(self):
        # Ensure reproducibility
        set_seed(SEED)

        self.device = torch.device(DEVICE)

        # Initialize Model
        print("Initializing model...")
        self.model = PointPillarsResNetFPN().to(self.device)

        # Initialize Loss
        self.criterion = CenterLoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.patience = 3  # Early stopping patience
        self.counter = 0

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

    def get_dataloaders(self):
        """Setup DataLoaders for Train and Val."""
        print("Loading datasets...")
        # Train Dataset
        train_ds = NuScenesDataset(
            metadata_path="./metadata/train_metadata.csv", split="train"
        )
        if TRAIN_SUBSET_SIZE is not None:
            indices = list(range(min(len(train_ds), TRAIN_SUBSET_SIZE)))
            train_ds = torch.utils.data.Subset(train_ds, indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate_fn,
            pin_memory=PIN_MEMORY,
        )

        # Val Dataset
        val_ds = NuScenesDataset(
            metadata_path="./metadata/val_metadata.csv", split="val"
        )
        if VAL_SUBSET_SIZE is not None:
            indices = list(range(min(len(val_ds), VAL_SUBSET_SIZE)))
            val_ds = torch.utils.data.Subset(val_ds, indices)

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate_fn,
            pin_memory=PIN_MEMORY,
        )

        return train_loader, val_loader

    def train_one_epoch(self, loader, scheduler, epoch_idx):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        stats_sum = {}
        num_batches = 0

        for batch in loader:
            # Move inputs to device
            points = batch["points"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            # Forward
            self.optimizer.zero_grad()
            preds = self.model(points)

            # Loss
            loss, stats = self.criterion(preds, targets)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_NORM_CLIP)
            self.optimizer.step()
            scheduler.step()

            # Logging
            total_loss += loss.item()
            for k, v in stats.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + v

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_stats = {k: v / num_batches for k, v in stats_sum.items()}

        print(f"Epoch {epoch_idx+1} | Train Loss: {avg_loss:.8f}")
        # Print detailed stats in a clean way
        stats_str = " | ".join([f"{k}: {v:.6f}" for k, v in avg_stats.items()])
        print(f"  Details: {stats_str}")

        return avg_loss

    def evaluate(self, loader):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                points = batch["points"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                preds = self.model(points)
                loss, _ = self.criterion(preds, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def run_training(self):
        """Main training execution loop."""
        train_loader, val_loader = self.get_dataloaders()

        # Scheduler (OneCycleLR)
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100,
        )

        print(f"Starting training for {NUM_EPOCHS} epochs on {self.device}...")

        for epoch in range(NUM_EPOCHS):
            _ = self.train_one_epoch(train_loader, scheduler, epoch)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch {epoch+1} | Val Loss: {val_loss:.8f}")

            # Checkpoint & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                save_path = os.path.join(WORKING_DIR, "model_checkpoint.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  [+] Saved best model to {save_path}")
            else:
                self.counter += 1
                print(f"  [-] EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def generate_submission(self):
        """Generate predictions for test set and save to CSV."""
        print("\nGenerating submission...")

        # Load best weights
        checkpoint_path = os.path.join(WORKING_DIR, "model_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"Loaded model from {checkpoint_path}")
        else:
            print("Warning: No checkpoint found. Using current weights.")

        self.model.eval()

        # Test Dataset
        test_ds = NuScenesDataset(
            metadata_path="./metadata/test_metadata.csv", split="test"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate_fn,
            pin_memory=PIN_MEMORY,
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                points = batch["points"].to(self.device)
                metadata = batch["metadata"]

                preds = self.model(points)

                # Decode predictions
                decoded_batch = decode_predictions(
                    preds["heatmap"],
                    preds["rot"],
                    preds["dim"],
                    preds["height"],
                    preds["reg"],
                    K=75,  # Keep top 75 objects per sample
                    score_threshold=0.1,
                )

                # Format for submission
                for i, boxes in enumerate(decoded_batch):
                    s_token = metadata[i]["sample_token"]
                    pred_str_list = []

                    if boxes is not None and len(boxes) > 0:
                        boxes_cpu = boxes.cpu().numpy()
                        for box in boxes_cpu:
                            # Box format from decode_predictions: [x, y, z, w, l, h, yaw, score, class_id]
                            x, y, z, w, l, h, yaw, score, cls_id = box

                            cls_name = CLASS_NAMES[int(cls_id)]

                            # Submission Format: confidence center_x center_y center_z width length height yaw class_name
                            s = f"{score:.4f} {x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw:.4f} {cls_name}"
                            pred_str_list.append(s)

                    full_str = " ".join(pred_str_list)
                    results.append({"Id": s_token, "PredictionString": full_str})

        # Save CSV
        df = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
