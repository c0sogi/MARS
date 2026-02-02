import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats as stats
import cv2

# Import provided library components
from library.config import Config
from library.utils import set_seed, rle_encoding
from library.model import InkSegFormer
from library.data import get_dataloaders, get_test_dataloader
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # load_cached_data=True to use pre-processed .npy files if available
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = InkSegFormer().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_f05 = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        metrics = validate(model, val_loader, device)
        val_f05 = metrics["f0.5"]

        # Save Best Model
        if val_f05 > best_f05:
            best_f05 = val_f05
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! F0.5: {val_f05:.6f}")

    # Required Output Format
    print(f"Final Validation Metric: {best_f05}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    errors = []
    intensities = []

    # Iterate validation set to collect error stats
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)  # (B, 3, H, W)
            labels = batch["label"].to(device)  # (B, 1, H, W)
            masks = batch["valid_mask"].to(device)  # (B, 1, H, W)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Calculate pixel-wise L1 error
            # Only consider valid pixels defined by the mask
            abs_diff = torch.abs(probs - labels) * masks

            # Average error per patch (sum of error / count of valid pixels)
            # Add epsilon to avoid division by zero
            valid_counts = masks.sum(dim=(1, 2, 3))
            batch_errors = abs_diff.sum(dim=(1, 2, 3)) / (valid_counts + 1e-6)

            # Average intensity per patch (mean of 3 channels)
            # We compute global mean of the image tensor
            batch_intensities = images.mean(dim=(1, 2, 3))

            errors.extend(batch_errors.cpu().tolist())
            intensities.extend(batch_intensities.cpu().tolist())

    if len(errors) > 0:
        correlation, p_value = stats.pearsonr(errors, intensities)
        print(
            f"Correlation between Error Magnitude and Input Intensity: {correlation:.4f} (p={p_value:.4f})"
        )
    else:
        print("Insufficient data for failure analysis.")

    # 6. Submission Logic
    THRESHOLD_SCORE = 0.597622633

    if best_f05 > THRESHOLD_SCORE:
        print(
            f"\nValidation score {best_f05:.6f} > {THRESHOLD_SCORE}. Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nValidation score {best_f05:.6f} did not exceed threshold {THRESHOLD_SCORE}. Skipping submission."
        )


def generate_submission(model, device):
    """
    Generates submission using Multi-View Ensemble Scanning and TTA.
    """
    model.eval()

    # Store original config to restore later
    original_view_b_start = Config.VIEW_B_START

    # Define the 3 discrete views
    views = [Config.VIEW_A_START, Config.VIEW_B_START, Config.VIEW_C_START]
    view_names = ["A", "B", "C"]

    # We will store predictions for all patches for all views
    # Since the test loader is deterministic, the order of patches is preserved.
    # We collect lists of numpy arrays.
    all_view_preds = []  # List of lists

    # Metadata list to reconstruct fragments later
    meta_list = []

    # --- Multi-View Inference Loop ---
    for i, view_start in enumerate(views):
        print(f"Running Inference for View {view_names[i]} (Start Z={view_start})...")

        # HACK: Modify Config global state to force InkDataset to load the specific view
        # InkDataset uses Config.VIEW_B_START for test split
        Config.VIEW_B_START = view_start

        # Re-initialize loader to pick up the new Config value
        test_loader = get_test_dataloader(load_cached_data=True)

        view_preds = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images = batch["image"].to(device)

                # Collect metadata only on the first pass
                if i == 0:
                    fids = batch["fragment_id"]
                    xs = batch["x"]
                    ys = batch["y"]
                    for k in range(len(fids)):
                        meta_list.append(
                            {"fid": fids[k], "x": xs[k].item(), "y": ys[k].item()}
                        )

                # --- Test Time Augmentation (TTA) ---
                # 1. Original
                out = torch.sigmoid(model(images))

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                out_h = torch.sigmoid(model(images_h))
                out_h = torch.flip(out_h, [3])

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                out_v = torch.sigmoid(model(images_v))
                out_v = torch.flip(out_v, [2])

                # Average TTA
                out_avg = (out + out_h + out_v) / 3.0

                # Squeeze channel dim and move to CPU
                preds_numpy = out_avg.squeeze(1).cpu().numpy()

                for p in preds_numpy:
                    view_preds.append(p)

        all_view_preds.append(view_preds)

    # Restore Config
    Config.VIEW_B_START = original_view_b_start

    # --- Fusion and Reconstruction ---
    print("Fusing views and reconstructing fragments...")

    # Initialize buffers for fragments
    # We need to know the size of each fragment.
    # We can read the masks using the metadata paths.
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    fragment_buffers = {}
    fragment_masks = {}

    for _, row in test_df.iterrows():
        fid = row["fragment_id"]
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        fragment_buffers[fid] = np.zeros((h, w), dtype=np.float32)
        fragment_masks[fid] = mask  # Keep mask for final masking

    # Iterate through all patches and fuse
    num_patches = len(meta_list)
    for idx in range(num_patches):
        meta = meta_list[idx]
        fid = meta["fid"]
        x = meta["x"]
        y = meta["y"]

        # Get predictions from all 3 views
        p_a = all_view_preds[0][idx]
        p_b = all_view_preds[1][idx]
        p_c = all_view_preds[2][idx]

        # Max-Fusion
        p_fused = np.maximum(p_a, np.maximum(p_b, p_c))

        # Place into fragment buffer
        h_frag, w_frag = fragment_buffers[fid].shape
        h_patch, w_patch = p_fused.shape

        # Calculate valid region (handle edge cropping/padding)
        y_end = min(y + h_patch, h_frag)
        x_end = min(x + w_patch, w_frag)

        valid_h = y_end - y
        valid_w = x_end - x

        # Assign
        fragment_buffers[fid][y:y_end, x:x_end] = p_fused[:valid_h, :valid_w]

    # --- RLE Encoding and Saving ---
    submission_rows = []

    for fid, pred_map in fragment_buffers.items():
        # Threshold
        binary_map = (pred_map > Config.THRESHOLD).astype(np.uint8)

        # Apply fragment validity mask
        valid_mask = (fragment_masks[fid] > 0).astype(np.uint8)
        binary_map = binary_map * valid_mask

        # Encode
        rle = rle_encoding(binary_map)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # Save
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
