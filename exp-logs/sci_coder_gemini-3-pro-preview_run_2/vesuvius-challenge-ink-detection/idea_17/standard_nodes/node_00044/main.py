import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
import gc
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, rle_encoding, get_transforms
from library.dataset import InkDataset, get_test_patches
from library.model import build_segformer_model
from library.engine import fit, evaluate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALID_METADATA_PATH)

    # Create Datasets
    # Training uses the fixed Z-start defined in Config (Z=20)
    train_dataset = InkDataset(
        train_df,
        split="train",
        z_start=Config.Z_START_TRAIN,
        transforms=get_transforms("train"),
    )

    val_dataset = InkDataset(
        val_df,
        split="valid",
        z_start=Config.Z_START_TRAIN,
        transforms=get_transforms("valid"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("Building SegFormer MiT-B2 model...")
    model = build_segformer_model()
    model = model.to(device)

    # 4. Training
    print("Starting training...")
    best_score = fit(
        model, train_loader, val_loader, epochs=Config.NUM_EPOCHS, device=device
    )

    # Clean up training memory
    del train_loader, train_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Final Validation & Failure Analysis
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # We need a custom evaluation loop to gather data for failure analysis
    criterion = torch.nn.BCEWithLogitsLoss()  # Just for loss calculation

    all_errors = []
    all_intensities = []

    # Re-instantiate val loader for analysis
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Metric accumulation
            all_preds.append(probs.cpu().numpy().flatten())
            all_targets.append(labels.cpu().numpy().flatten())

            # Failure Analysis Data Collection
            # Error magnitude: |pred - target|
            error_batch = torch.abs(probs - labels).cpu().numpy().flatten()

            # Input intensity: mean across channels (H, W, 3) -> (H, W)
            # Images are (B, 3, H, W)
            intensity_batch = images.mean(dim=1).cpu().numpy().flatten()

            all_errors.append(error_batch)
            all_intensities.append(intensity_batch)

    # Calculate Final Metric
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)

    from library.utils import fbeta_score

    final_f05 = fbeta_score(
        all_preds_flat, all_targets_flat, beta=0.5, threshold=Config.THRESHOLD
    )

    print(f"Final Validation Metric: {final_f05}")

    # Calculate Correlation
    all_errors_flat = np.concatenate(all_errors)
    all_intensities_flat = np.concatenate(all_intensities)

    # Downsample for correlation calculation if too large (to save time/memory)
    if len(all_errors_flat) > 1_000_000:
        indices = np.random.choice(len(all_errors_flat), 1_000_000, replace=False)
        corr, _ = pearsonr(all_errors_flat[indices], all_intensities_flat[indices])
    else:
        corr, _ = pearsonr(all_errors_flat, all_intensities_flat)

    print(
        f"Failure Analysis: Correlation between Error Magnitude and Input Intensity: {corr:.4f}"
    )

    # 6. Submission (Conditional)
    SUBMISSION_THRESHOLD = 0.597622633

    if final_f05 > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric {final_f05} > {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        # get_test_patches returns a dataframe of all patches in test set
        all_test_patches = get_test_patches(test_meta)

        submission_data = []

        # Group by fragment to process one full image at a time
        for frag_id, frag_patches in all_test_patches.groupby("fragment_id"):
            print(f"Processing fragment {frag_id}...")

            # Get fragment dimensions from the mask
            mask_path = os.path.join(
                Config.INPUT_DIR, frag_patches.iloc[0]["mask_path"]
            )
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Mask not found for {frag_id}")
                continue

            h_img, w_img = mask.shape

            # Initialize accumulator for Max-Fusion
            fused_probs = np.zeros((h_img, w_img), dtype=np.float32)

            # Decoupled Z-Scanning: Iterate over Z-offsets
            for z_offset in Config.Z_OFFSETS_INFERENCE:
                # Create a dataset for this specific Z-offset
                # We reuse the patch definitions but shift the Z-window
                ds = InkDataset(
                    frag_patches,
                    split="test",
                    z_start=z_offset,
                    transforms=get_transforms("test"),
                )

                dl = DataLoader(
                    ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    drop_last=False,
                )

                # Temp buffer for this Z-level
                z_level_probs = np.zeros((h_img, w_img), dtype=np.float32)

                with torch.no_grad():
                    for batch in dl:
                        imgs = batch["image"].to(device)
                        preds = model(imgs)
                        preds = torch.sigmoid(preds).cpu().numpy()  # (B, 1, H, W)

                        # Place predictions into the buffer
                        # We need to handle the coordinates
                        xs = batch["x"].numpy()
                        ys = batch["y"].numpy()

                        for i in range(len(preds)):
                            px, py = xs[i], ys[i]
                            pred_patch = preds[i, 0]  # (512, 512)

                            # Calculate valid region (handling edges)
                            valid_h = min(Config.TILE_SIZE, h_img - py)
                            valid_w = min(Config.TILE_SIZE, w_img - px)

                            # Assign
                            z_level_probs[py : py + valid_h, px : px + valid_w] = (
                                pred_patch[:valid_h, :valid_w]
                            )

                # Max-Fusion Update
                fused_probs = np.maximum(fused_probs, z_level_probs)

                # Cleanup
                del ds, dl, z_level_probs
                gc.collect()

            # Apply Mask and Threshold
            # Ensure we don't predict outside the valid fragment area
            fused_probs = fused_probs * (mask > 0).astype(np.float32)

            binary_pred = (fused_probs > Config.THRESHOLD).astype(np.uint8)

            # RLE Encode
            rle_str = rle_encoding(binary_pred)
            submission_data.append({"Id": frag_id, "Predicted": rle_str})

        # Write Submission
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric {final_f05} <= {SUBMISSION_THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
