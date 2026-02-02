import os
import time
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_NORM_CLIP,
    SEED,
    WARMUP_EPOCHS,
)
from library.dataset import LidarDataset
from library.model import PointPillars
from library.utils import setup_logger


class Solver:
    def __init__(self):
        # 1. Setup Environment
        self._set_random_seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(os.path.join(WORKING_DIR, "train.log"))

        self.logger.info(f"Initializing Solver on device: {self.device}")

        # 2. Data Loaders
        self.logger.info("Loading Datasets...")
        self.train_dataset = LidarDataset(split="train", load_cached_data=True)
        self.val_dataset = LidarDataset(split="val", load_cached_data=True)
        self.test_dataset = LidarDataset(split="test", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=LidarDataset.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=LidarDataset.collate_fn,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=LidarDataset.collate_fn,
            pin_memory=True,
        )

        # 3. Model
        self.logger.info("Building Model...")
        self.model = PointPillars().to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # OneCycleLR
        steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE,
            total_steps=EPOCHS * steps_per_epoch,
            pct_start=WARMUP_EPOCHS / EPOCHS,
            div_factor=10.0,
            final_div_factor=100.0,
        )

        self.best_val_loss = float("inf")
        self.patience = 5
        self.counter = 0

    def _set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True # Can slow down training

    def _to_device(self, batch_dict):
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.to(self.device)
            elif isinstance(value, list):
                # Handle list of tensors (gt_boxes, gt_labels)
                batch_dict[key] = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
        return batch_dict

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_dir_loss = 0.0

        start_time = time.time()

        for batch_idx, batch_dict in enumerate(self.train_loader):
            batch_dict = self._to_device(batch_dict)

            self.optimizer.zero_grad()

            # Forward pass (PointPillars.forward returns loss dict when training=True)
            loss_dict = self.model(batch_dict)

            loss = loss_dict["loss"]

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_NORM_CLIP)
            self.optimizer.step()
            self.scheduler.step()

            # Logging
            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_box_loss += loss_dict["box_loss"].item()
            total_dir_loss += loss_dict["dir_loss"].item()

        avg_loss = total_loss / len(self.train_loader)
        avg_cls = total_cls_loss / len(self.train_loader)
        avg_box = total_box_loss / len(self.train_loader)
        avg_dir = total_dir_loss / len(self.train_loader)
        duration = time.time() - start_time

        self.logger.info(
            f"Epoch [{epoch_idx+1}/{EPOCHS}] Train Loss: {avg_loss:.6f} "
            f"(Cls: {avg_cls:.6f}, Box: {avg_box:.6f}, Dir: {avg_dir:.6f}) "
            f"Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self, epoch_idx):
        """
        Calculates validation loss.
        We manually run the model components to get loss while in eval mode.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_dict in self.val_loader:
                batch_dict = self._to_device(batch_dict)

                # Manual forward pass to get loss in eval mode
                # 1. PFN
                x = self.model.pfn(
                    batch_dict["pillar_features"],
                    batch_dict["num_points"],
                    batch_dict["pillar_coords"],
                )
                # 2. Scatter
                x = self.model.scatter(
                    x, batch_dict["pillar_coords"], batch_dict["batch_size"]
                )
                # 3. Backbone
                x = self.model.backbone(x)
                # 4. Head
                cls_preds, box_preds, dir_preds = self.model.head(x)

                # 5. Loss
                loss_dict = self.model.get_loss(
                    cls_preds, box_preds, dir_preds, batch_dict
                )
                total_loss += loss_dict["loss"].item()

        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch [{epoch_idx+1}/{EPOCHS}] Val Loss: {avg_loss:.10f}")
        return avg_loss

    def fit(self):
        self.logger.info("Starting Training...")

        for epoch in range(EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # Checkpoint & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                save_path = os.path.join(WORKING_DIR, "model_checkpoint.pth")
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(
                    f"Validation loss improved. Model saved to {save_path}"
                )
            else:
                self.counter += 1
                self.logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
                if self.counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        self.logger.info("Training Complete.")

    def inference(self):
        self.logger.info("Starting Inference on Test Set...")

        # Load best model
        checkpoint_path = os.path.join(WORKING_DIR, "model_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            self.logger.info(f"Loaded best model from {checkpoint_path}")
        else:
            self.logger.warning("No checkpoint found! Using current model weights.")

        self.model.eval()

        results = []
        ids = []

        with torch.no_grad():
            for batch_dict in self.test_loader:
                batch_dict = self._to_device(batch_dict)

                # Forward pass (PointPillars.forward returns predictions list when training=False)
                # Note: model.eval() sets training=False in nn.Module,
                # but PointPillars.forward checks self.training.
                # self.training is managed by model.train() / model.eval()

                preds = self.model(batch_dict)

                # Collect results
                batch_ids = batch_dict["sample_tokens"]
                ids.extend(batch_ids)
                results.extend(preds)

        # Create Submission DataFrame
        df = pd.DataFrame({"Id": ids, "PredictionString": results})

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        df.to_csv(SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {SUBMISSION_PATH} with {len(df)} rows.")


if __name__ == "__main__":
    # This block is for testing the module independently if needed,
    # but the prompt asks not to include the execution block.
    # However, to run the task, we usually need an entry point.
    # The prompt says: "DO NOT include an if __name__ == '__main__': block."
    # So I will just define the class.
    pass
