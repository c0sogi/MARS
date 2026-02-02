import os
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.losses import DeepSupervisionLoss
from library.metrics import calculate_iou_map


def train_model():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing Training on device: {device}")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # 3. Model, Loss, Optimizer
    model = DeepResUNet(
        in_channels=2, out_channels=1, deep_supervision=Config.DEEP_SUPERVISION
    )
    model = model.to(device)

    criterion = DeepSupervisionLoss(weights=Config.DS_WEIGHTS)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.T_0,
        T_mult=Config.T_MULT,
        eta_min=Config.MIN_LR,
    )

    # 4. Training Loop Variables
    best_map = 0.0
    early_stopping_patience = 20
    patience_counter = 0

    # Padding offset for cropping (128 - 101) // 2 = 13
    pad_offset = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    crop_slice = slice(pad_offset, pad_offset + Config.ORIG_SIZE)

    print("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # --- Train Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass (returns list if deep supervision is on)
            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        val_map_accum = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass (returns single tensor in eval mode)
                output = model(images)

                # Calculate Validation Loss
                loss = criterion(output, masks)
                val_loss_accum += loss.item()

                # Calculate mAP
                # 1. Sigmoid to get probabilities
                probs = torch.sigmoid(output)

                # 2. Crop back to 101x101
                # Output shape: (B, 1, 128, 128) -> Crop spatial dims
                probs_cropped = probs[:, :, crop_slice, crop_slice]
                masks_cropped = masks[:, :, crop_slice, crop_slice]

                # 3. Calculate metric
                batch_map = calculate_iou_map(probs_cropped, masks_cropped)
                val_map_accum += batch_map

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_map = val_map_accum / len(val_loader)

        # --- Logging & Checkpointing ---
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {epoch_time:.2f}s | LR: {current_lr}"
        )
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val mAP: {avg_val_map}")

        # Scheduler Step (CosineAnnealing)
        scheduler.step()

        # Save Best Model (Monitor mAP)
        if avg_val_map > best_map:
            print(
                f"Validation mAP improved from {best_map} to {avg_val_map}. Saving checkpoint."
            )
            best_map = avg_val_map
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    # 5. Inference & Submission
    print("\nStarting Inference on Test Set...")

    # Load Best Model
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()
    submission_data = []

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)

            # Test Time Augmentation (TTA)
            # 1. Original Prediction
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Flipped Prediction
            images_flipped = torch.flip(images, dims=[3])  # Flip width (dim 3)
            out_flipped = model(images_flipped)
            prob_flipped = torch.sigmoid(out_flipped)
            prob_flipped = torch.flip(prob_flipped, dims=[3])  # Flip back

            # Average
            probs = (prob_orig + prob_flipped) / 2.0

            # Crop to 101x101
            probs_cropped = probs[:, 0, crop_slice, crop_slice]  # (B, 101, 101)

            # Convert to binary mask
            preds_bin = (probs_cropped > 0.5).byte().cpu().numpy()

            # Get IDs for this batch
            # The test loader iterates sequentially. We need to map back to IDs.
            # We can access the dataset from the loader, but indices are cleaner.
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_ids = test_loader.dataset.ids[
                start_idx:end_idx
            ]  # Assuming dataset has .ids attribute exposed or we access underlying data
            # Note: SaltDataset stores ids in the underlying dict if we passed them,
            # but the provided SaltDataset class doesn't explicitly expose 'ids' as a public attribute easily
            # unless we modify it or access the cache.
            # However, `load_data_and_cache` returns a dict with 'ids'.
            # Let's reconstruct IDs from the metadata file to be safe and order-preserving.

    # Re-loading test metadata to ensure correct ID mapping order
    df_test = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    all_preds = []

    # Re-run inference loop strictly collecting predictions
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # TTA
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            images_flipped = torch.flip(images, dims=[3])
            out_flipped = model(images_flipped)
            prob_flipped = torch.sigmoid(out_flipped)
            prob_flipped = torch.flip(prob_flipped, dims=[3])

            probs = (prob_orig + prob_flipped) / 2.0
            probs_cropped = probs[:, 0, crop_slice, crop_slice]
            preds_bin = (probs_cropped > 0.5).byte().cpu().numpy()

            for i in range(len(preds_bin)):
                all_preds.append(preds_bin[i])

    # Generate RLE
    print("Generating RLE masks...")
    rle_masks = []
    for mask in all_preds:
        rle_masks.append(rle_encode(mask))

    # Create Submission DataFrame
    sub_df = pd.DataFrame({"id": df_test["id"].values, "rle_mask": rle_masks})

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    train_model()
