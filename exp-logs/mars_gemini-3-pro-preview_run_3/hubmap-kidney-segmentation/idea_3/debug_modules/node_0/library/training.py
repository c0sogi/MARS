import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scipy.stats import norm

# Import from library
from library.utils import set_seed, get_logger, rle_encode, compute_dice_score
from library.dataset import prepare_datasets, prepare_test_dataset
from library.model import build_model
from library.losses import DeepSupervisionLoss

# Initialize Logger
logger = get_logger("TrainingModule")


def get_gaussian_window(size, sigma=None):
    """
    Generates a 2D Gaussian window for weighting tile predictions during stitching.
    """
    if sigma is None:
        sigma = size / 4.0  # Heuristic: sigma = size / 4 covers the tile well

    x = np.linspace(0, size, size)
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y)

    center = size / 2.0
    gauss = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

    return gauss.astype(np.float32)


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, scaler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.config = config
        self.best_dice = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        accumulation_steps = self.config.get("accumulation_steps", 1)

        self.optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # Mixed Precision Forward
            with autocast(enabled=True):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient Accumulation Step
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        dice_scores = []

        # We compute Dice per batch to save memory, then average
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # In eval mode, model returns single tensor (final output)
                outputs = self.model(images)

                # Sigmoid & Threshold
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                # Compute Dice for this batch
                # masks is (B, H, W), preds is (B, 1, H, W)
                masks_np = masks.cpu().numpy()
                preds_np = preds.squeeze(1).cpu().numpy()

                for i in range(len(masks_np)):
                    score = compute_dice_score(masks_np[i], preds_np[i])
                    dice_scores.append(score)

        avg_dice = np.mean(dice_scores)
        logger.info(f"Epoch {epoch} | Val Dice: {avg_dice:.6f}")
        return avg_dice

    def fit(self, train_loader, val_loader, epochs, save_path):
        logger.info("Starting training...")

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            val_dice = self.validate_epoch(val_loader, epoch)

            if self.scheduler:
                self.scheduler.step()

            # Checkpointing
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved with Dice: {val_dice:.6f}")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= self.config.get("patience", 5):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(f"Training complete. Best Val Dice: {self.best_dice:.6f}")
        return self.best_dice


def run_training(
    tile_size=1024,
    batch_size=4,
    accumulation_steps=8,
    epochs=20,
    lr=1e-4,
    patience=5,
    debug=False,
):
    """
    Main function to run the training pipeline.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    config = {
        "tile_size": tile_size,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
    }

    # 1. Prepare Data
    train_dataset, val_dataset = prepare_datasets(
        tile_size=tile_size,
        overlap=0.5,
        do_normalization=True,
        load_cached_data=True,
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Build Model
    model = build_model(encoder_name="convnext_base", in_channels=3, classes=1)
    model = model.to(device)

    # 3. Setup Training Components
    criterion = DeepSupervisionLoss(
        weights=[1.0, 0.5, 0.25], bce_weight=0.5, dice_weight=0.5
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    scaler = GradScaler()

    # 4. Train
    trainer = Trainer(model, optimizer, scheduler, criterion, device, scaler, config)

    save_dir = "./working/idea_3"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    trainer.fit(train_loader, val_loader, epochs, save_path)

    # Clear memory
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return save_path


def predict_and_submit(
    model_path, output_path="./submission/submission.csv", tile_size=1024, batch_size=4
):
    """
    Runs inference on the test set, stitches tiles, and generates submission CSV.
    """
    logger.info("Starting Inference and Submission Generation...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    test_dataset, df_test_tiles = prepare_test_dataset(
        tile_size=tile_size, overlap=0.5, do_normalization=True, load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Load Model
    model = build_model(encoder_name="convnext_base", classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Prepare Stitching Resources
    # We need to reconstruct full images.
    # Group tiles by image_id
    image_ids = df_test_tiles["id"].unique()

    # Load metadata to get image dimensions
    meta_df = pd.read_csv("./metadata/test.csv")
    meta_lookup = meta_df.set_index("id")

    submission_rows = []

    # Gaussian Window
    gaussian_weight = get_gaussian_window(tile_size)

    # We process image by image to save memory, but the loader yields mixed batches if we shuffled.
    # But we set shuffle=False, so tiles are ordered by the dataframe.
    # However, dataframe might interleave if not sorted. process_dataset appends sequentially.
    # To be safe, we'll iterate the loader and accumulate results into a dictionary of accumulators.
    # Given memory constraints, we should process one image at a time if possible.
    # But the loader is already built. We will use a pointer to the dataframe row.

    # Map for current images being processed: image_id -> { 'prob': float_array, 'weight': float_array }
    # Since test set is small (3 images), we can probably keep them in memory or process sequentially.
    # Let's verify if df_test_tiles is sorted by image_id.
    # process_dataset iterates metadata rows, so yes, it is grouped by image.

    # We will iterate through the loader and fill buffers.
    # Since batches might cross image boundaries, we handle that.

    # Initialize buffers for all test images
    buffers = {}
    for img_id in image_ids:
        if img_id not in meta_lookup.index:
            logger.warning(f"Metadata missing for {img_id}, skipping.")
            continue

        h = int(meta_lookup.loc[img_id, "height_pixels"])
        w = int(meta_lookup.loc[img_id, "width_pixels"])

        # Use float16 to save memory if needed, but float32 is safer for accumulation
        buffers[img_id] = {
            "prob": np.zeros((h, w), dtype=np.float32),
            "weight": np.zeros((h, w), dtype=np.float32),
            "count": 0,
        }

    logger.info(f"Initialized buffers for {len(buffers)} images.")

    # 4. Inference Loop
    tile_idx = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)  # (B, 1, H, W)
            preds_np = preds.squeeze(1).cpu().numpy()

            batch_len = images.size(0)

            for i in range(batch_len):
                # Get tile info
                row = df_test_tiles.iloc[tile_idx]
                img_id = row["id"]
                x, y = row["x"], row["y"]

                if img_id in buffers:
                    # Add to buffer with Gaussian Weighting
                    # Handle edge cases where tile might be smaller (if we didn't pad)
                    # But dataset guarantees fixed size tiles (padded if necessary or shifted)
                    # HubmapDataset returns fixed size.

                    # We need to be careful: if the tile goes out of bounds of the original image buffer?
                    # The buffer is (H, W). The tile is at (x, y).
                    # x, y are top-left.

                    h_buf, w_buf = buffers[img_id]["prob"].shape

                    # Crop the prediction if it extends beyond image (shouldn't happen with valid tiling logic)
                    h_pred, w_pred = preds_np[i].shape

                    y_end = min(y + h_pred, h_buf)
                    x_end = min(x + w_pred, w_buf)

                    h_eff = y_end - y
                    w_eff = x_end - x

                    if h_eff > 0 and w_eff > 0:
                        buffers[img_id]["prob"][y:y_end, x:x_end] += (
                            preds_np[i, :h_eff, :w_eff]
                            * gaussian_weight[:h_eff, :w_eff]
                        )
                        buffers[img_id]["weight"][y:y_end, x:x_end] += gaussian_weight[
                            :h_eff, :w_eff
                        ]
                        buffers[img_id]["count"] += 1

                tile_idx += 1

            if tile_idx % 100 == 0:
                logger.info(f"Processed {tile_idx} tiles...")

    # 5. Post-processing and Encoding
    logger.info("Finalizing predictions...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for img_id, buf in buffers.items():
        # Normalize
        # Avoid division by zero
        mask = buf["weight"] > 0
        buf["prob"][mask] /= buf["weight"][mask]

        # Threshold
        binary_mask = (buf["prob"] > 0.5).astype(np.uint8)

        # Encode
        rle = rle_encode(binary_mask)
        submission_rows.append({"id": img_id, "predicted": rle})

        # Free memory
        del buffers[img_id]
        gc.collect()

    # 6. Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
