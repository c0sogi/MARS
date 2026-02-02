import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import set_seed, dice_coefficient, hausdorff_distance_3d
from library.dataset import UWMadisonDataset
from library.model import UNetResNet18
from library.train import run_training
from library.inference import run_inference

# --- Configuration ---
SEED = 42
BATCH_SIZE = 32
IMG_SIZE = 256
EPOCHS = 5
FRACTION = 0.5  # Use 50% of data for a fast but representative baseline
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "./working/idea_1/best_model.pth"
SUBMISSION_DIR = "./submission"


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Starting execution on device: {DEVICE}")

    # 2. Training Phase
    print("\n=== Phase 1: Training ===")
    # Train the model using the provided training module
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        fraction=FRACTION,
        lr=1e-3,
        patience=3,
        img_size=IMG_SIZE,
    )

    # 3. Validation & Failure Analysis Phase
    print("\n=== Phase 2: Validation & Failure Analysis ===")

    # Initialize model and load best weights
    model = UNetResNet18(num_classes=3).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded best model checkpoint for validation.")
    else:
        print("Warning: Checkpoint not found. Using untrained model.")

    model.eval()

    # Load the full validation dataset
    val_dataset = UWMadisonDataset(mode="val", fraction=1.0, img_size=IMG_SIZE)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Evaluating on {len(val_dataset)} validation samples...")

    # Storage for 3D reconstruction
    volume_data = {}

    with torch.no_grad():
        # Dataset returns (images, masks, metadata), so we unpack 3 values
        # Cite debug_lesson_1: Correctly unpacking metadata to handle interface change
        for images, masks, metadata in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            preds = (outputs > 0.5).float()

            # Move to CPU/Numpy for storage
            # Cite debug_lesson_2: Explicitly moving to CPU before numpy conversion
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            images_np = images.cpu().numpy()
            meta_np = metadata.numpy()

            for i in range(images.size(0)):
                case_id = int(meta_np[i, 0])
                day_id = int(meta_np[i, 1])
                slice_id = int(meta_np[i, 2])

                key = (case_id, day_id)
                if key not in volume_data:
                    volume_data[key] = []

                # Store slice data for volume reconstruction
                volume_data[key].append(
                    {
                        "slice": slice_id,
                        "pred": preds_np[i],
                        "true": masks_np[i],
                        "img_mean": np.mean(images_np[i]),
                    }
                )

    # Compute 3D Metrics
    dice_scores = []
    hd_scores = []
    combined_scores = []

    # Lists for failure analysis (Volume level)
    error_magnitudes = []
    feat_mask_areas = []
    feat_img_intensities = []

    for key, slices in volume_data.items():
        # Sort by slice index to form proper volume
        slices.sort(key=lambda x: x["slice"])

        # Stack to create volumes: (D, 3, H, W)
        pred_vol = np.stack([s["pred"] for s in slices], axis=0)
        true_vol = np.stack([s["true"] for s in slices], axis=0)

        # Calculate features for analysis
        vol_mean_intensity = np.mean([s["img_mean"] for s in slices])
        vol_mask_area = np.sum(true_vol)

        # Calculate metric per class
        case_dice = 0
        case_hd = 0

        for c in range(3):
            p_c = pred_vol[:, c, :, :]  # (D, H, W)
            t_c = true_vol[:, c, :, :]  # (D, H, W)

            case_dice += dice_coefficient(t_c, p_c)
            case_hd += hausdorff_distance_3d(t_c, p_c)

        case_dice /= 3.0
        case_hd /= 3.0

        # Competition Metric: 0.4 * Dice + 0.6 * (1 - HD)
        # HD is normalized [0,1], so (1-HD) is similarity
        h_score_component = max(0.0, 1.0 - case_hd)
        score = 0.4 * case_dice + 0.6 * h_score_component

        dice_scores.append(case_dice)
        hd_scores.append(case_hd)
        combined_scores.append(score)

        # Failure Analysis Data
        error_magnitudes.append(1.0 - score)
        feat_mask_areas.append(vol_mask_area)
        feat_img_intensities.append(vol_mean_intensity)

    # Compute and Print Final Metrics
    final_metric = np.mean(combined_scores)
    final_dice = np.mean(dice_scores)
    final_hd = np.mean(hd_scores)

    print(f"Final Validation Metric: {final_metric:.10f}")
    print(
        f"Breakdown -> Mean Dice: {final_dice:.5f}, Mean Hausdorff Dist: {final_hd:.5f}"
    )

    # Failure Analysis: Correlations
    if len(error_magnitudes) > 1:
        corr_area, _ = pearsonr(error_magnitudes, feat_mask_areas)
        corr_intensity, _ = pearsonr(error_magnitudes, feat_img_intensities)

        print("\n--- Failure Analysis Correlations ---")
        print(f"Correlation (Error vs Mask Volume): {corr_area:.4f}")
        print(f"Correlation (Error vs Volume Intensity): {corr_intensity:.4f}")

        if abs(corr_area) > 0.2:
            print(
                "Insight: Model performance is sensitive to the size of the segmentation mask."
            )
        if abs(corr_intensity) > 0.2:
            print(
                "Insight: Model performance is sensitive to image brightness/contrast."
            )

    # 4. Inference Phase
    print("\n=== Phase 3: Inference & Submission ===")
    run_inference(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        checkpoint_path=CHECKPOINT_PATH,
        submission_dir=SUBMISSION_DIR,
    )

    print("Runfile Execution Complete.")


if __name__ == "__main__":
    main()
