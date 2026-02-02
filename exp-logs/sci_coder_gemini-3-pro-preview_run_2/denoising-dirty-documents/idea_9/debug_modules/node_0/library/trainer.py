import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    set_seed,
    calculate_rmse,
    create_submission_file,
    print_metrics,
    save_image,
    denormalize,
)
from library.model import CoRes2NetUNet
from library.dataset import DenoisingDataset


class Trainer:
    """
    Trainer class for the CoRes2Net-UNet denoising model.
    Handles training, validation, and submission generation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = CoRes2NetUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
        ).to(self.device)

        # Optimizer and Loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.criterion = nn.MSELoss()

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        # Best Metric for Checkpointing
        self.best_rmse = float("inf")

    def train_one_epoch(self, train_loader: DataLoader, epoch: int):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        # Disable progress bar for submission environment compliance if needed,
        # but keeping it simple for logging.
        # Using enumerate to avoid tqdm if strict silence is required,
        # but prompt allows "required information".

        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Target is the noise residual
            target_noise = noisy - clean

            self.optimizer.zero_grad()

            # Predict noise
            pred_noise = self.model(noisy)

            loss = self.criterion(pred_noise, target_noise)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def _apply_tta(self, x):
        """
        Applies 8 geometric transformations (D4 group) to the input batch.
        Returns a tensor of shape (B*8, C, H, W).
        """
        # x: [B, C, H, W]
        out = []
        # 0: Identity
        out.append(x)
        # 1: Rot90
        out.append(torch.rot90(x, 1, [2, 3]))
        # 2: Rot180
        out.append(torch.rot90(x, 2, [2, 3]))
        # 3: Rot270
        out.append(torch.rot90(x, 3, [2, 3]))
        # 4: HFlip
        out.append(torch.flip(x, [3]))
        # 5: VFlip
        out.append(torch.flip(x, [2]))
        # 6: Transpose (Rot90 + Flip)
        out.append(torch.flip(torch.rot90(x, 1, [2, 3]), [2]))
        # 7: Anti-Transpose (Rot90 + Flip)
        out.append(torch.flip(torch.rot90(x, 1, [2, 3]), [3]))

        return torch.cat(out, dim=0)

    def _reverse_tta(self, x, batch_size):
        """
        Reverses the TTA transformations and averages the results.
        x: [B*8, C, H, W]
        Returns: [B, C, H, W]
        """
        # Split into the 8 groups
        chunks = torch.chunk(x, 8, dim=0)

        # Inverse transforms
        # 0: Identity
        c0 = chunks[0]
        # 1: Rot90 -> Rot270 (3)
        c1 = torch.rot90(chunks[1], 3, [2, 3])
        # 2: Rot180 -> Rot180 (2)
        c2 = torch.rot90(chunks[2], 2, [2, 3])
        # 3: Rot270 -> Rot90 (1)
        c3 = torch.rot90(chunks[3], 1, [2, 3])
        # 4: HFlip -> HFlip
        c4 = torch.flip(chunks[4], [3])
        # 5: VFlip -> VFlip
        c5 = torch.flip(chunks[5], [2])
        # 6: Transpose -> Transpose
        c6 = torch.rot90(torch.flip(chunks[6], [2]), 3, [2, 3])
        # 7: Anti-Transpose
        c7 = torch.rot90(torch.flip(chunks[7], [3]), 3, [2, 3])

        # Stack and mean
        stacked = torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=0)
        return torch.mean(stacked, dim=0)

    def predict_image_tiled(self, image_tensor):
        """
        Performs tiled inference with overlap and optional TTA.
        image_tensor: [1, C, H, W]
        """
        self.model.eval()

        b, c, h, w = image_tensor.shape
        patch_size = self.config.PATCH_SIZE
        overlap = self.config.OVERLAP_RATIO
        stride = int(patch_size * (1 - overlap))

        # Output containers
        output = torch.zeros((b, c, h, w), device=self.device)
        count = torch.zeros((b, c, h, w), device=self.device)

        # Calculate padding to ensure coverage
        # We need to pad such that the last patch ends exactly at or beyond W/H
        pad_h = 0
        pad_w = 0
        if h < patch_size:
            pad_h = patch_size - h
        if w < patch_size:
            pad_w = patch_size - w

        # Add extra padding for stride alignment if desired, but simple coverage logic is better
        # Just pad right/bottom
        padded_img = torch.nn.functional.pad(
            image_tensor, (0, pad_w, 0, pad_h), mode="reflect"
        )
        _, _, h_pad, w_pad = padded_img.shape

        patches = []
        coords = []

        # Extract patches
        y_range = list(range(0, h_pad - patch_size + 1, stride))
        if (h_pad - patch_size) % stride != 0:
            y_range.append(h_pad - patch_size)

        x_range = list(range(0, w_pad - patch_size + 1, stride))
        if (w_pad - patch_size) % stride != 0:
            x_range.append(w_pad - patch_size)

        # We process patches in batches to maximize GPU usage
        batch_patches = []
        batch_coords = []
        inference_batch_size = self.config.BATCH_SIZE

        # If TTA is enabled, effective batch size is smaller
        if self.config.TTA_ENABLED:
            inference_batch_size = max(1, inference_batch_size // 8)

        with torch.no_grad():
            for y in y_range:
                for x in x_range:
                    patch = padded_img[:, :, y : y + patch_size, x : x + patch_size]
                    batch_patches.append(patch)
                    batch_coords.append((y, x))

                    if len(batch_patches) >= inference_batch_size:
                        # Stack
                        inp = torch.cat(batch_patches, dim=0)  # [B, C, H, W]

                        if self.config.TTA_ENABLED:
                            inp_aug = self._apply_tta(inp)
                            pred_aug = self.model(inp_aug)
                            pred = self._reverse_tta(pred_aug, inp.shape[0])
                        else:
                            pred = self.model(inp)

                        # Accumulate
                        for i, (py, px) in enumerate(batch_coords):
                            output[
                                :, :, py : py + patch_size, px : px + patch_size
                            ] += pred[i : i + 1]
                            count[
                                :, :, py : py + patch_size, px : px + patch_size
                            ] += 1.0

                        batch_patches = []
                        batch_coords = []

            # Process remaining
            if len(batch_patches) > 0:
                inp = torch.cat(batch_patches, dim=0)
                if self.config.TTA_ENABLED:
                    inp_aug = self._apply_tta(inp)
                    pred_aug = self.model(inp_aug)
                    pred = self._reverse_tta(pred_aug, inp.shape[0])
                else:
                    pred = self.model(inp)

                for i, (py, px) in enumerate(batch_coords):
                    output[:, :, py : py + patch_size, px : px + patch_size] += pred[
                        i : i + 1
                    ]
                    count[:, :, py : py + patch_size, px : px + patch_size] += 1.0

        # Average
        output = output / count

        # Crop back
        output = output[:, :, :h, :w]

        return output

    def validate_epoch(self, val_loader: DataLoader):
        """
        Validates the model on the validation set.
        Returns average RMSE.
        """
        self.model.eval()
        rmse_list = []

        with torch.no_grad():
            for noisy, clean, img_id in val_loader:
                noisy = noisy.to(self.device)
                clean_target = clean.numpy()  # Keep on CPU for metric calc

                # Predict noise
                pred_noise = self.predict_image_tiled(noisy)

                # Reconstruct clean image: Clean = Noisy - Noise
                pred_clean = noisy - pred_noise

                # Clip to valid range
                pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

                # Calculate RMSE
                val_rmse = calculate_rmse(clean_target, pred_clean)
                rmse_list.append(val_rmse)

        return np.mean(rmse_list)

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device}...")

        # Datasets
        train_dataset = DenoisingDataset("train", self.config)
        val_dataset = DenoisingDataset("val", self.config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Validate one image at a time
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        patience = 10
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_rmse = self.validate_epoch(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Logging
            print(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.10f}"
            )

            # Checkpoint
            if val_rmse < self.best_rmse:
                self.best_rmse = val_rmse
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_PATH)
                print(f"New best model saved with RMSE: {self.best_rmse:.10f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best RMSE: {self.best_rmse:.10f}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best model.
        """
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(self.config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(self.config.CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded model from {self.config.CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()

        test_dataset = DenoisingDataset("test", self.config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        predictions = {}

        with torch.no_grad():
            for noisy, img_id_tuple in test_loader:
                img_id = img_id_tuple[0]
                noisy = noisy.to(self.device)

                # Predict
                pred_noise = self.predict_image_tiled(noisy)
                pred_clean = noisy - pred_noise
                pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

                # Convert to numpy
                pred_clean_np = pred_clean.squeeze().cpu().numpy()

                predictions[img_id] = pred_clean_np

        create_submission_file(predictions, self.config.SUBMISSION_PATH)
        print("Submission file created.")


def run_training():
    """
    Entry point helper.
    """
    config = Config()
    # Ensure reproducibility
    set_seed(config.SEED)

    trainer = Trainer(config)
    trainer.fit()
    trainer.generate_submission()
