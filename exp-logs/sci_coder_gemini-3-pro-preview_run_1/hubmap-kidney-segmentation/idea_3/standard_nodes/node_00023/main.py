import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    compute_intersection_union,
    calculate_global_dice,
    rle_decode,
)
from library.data import get_dataloaders
from library.arch import ResNet34UNetPlusPlus
from library.engine import run_training, validate
from library.predict import generate_submission, predict_slide


def analyze_failures(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Computes Dice scores per image and correlates error (1 - Dice) with metadata.
    """
    print("\n--- Starting Failure Analysis ---")

    # Access the coordinate dataframe to map tiles back to image IDs
    # val_loader is assumed to be sequential (shuffle=False)
    coords_df = val_loader.dataset.coords_df

    # Dictionary to accumulate intersection and union per image
    # Structure: {image_id: {'inter': 0.0, 'union': 0.0}}
    image_stats = {}

    model.eval()
    current_idx = 0

    # 1. Inference and Accumulation
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Thresholding
            preds = (preds > Config.MASK_THRESHOLD).float()

            batch_size = images.size(0)

            # Process each sample in the batch
            for i in range(batch_size):
                # Retrieve image ID corresponding to this tile
                # The loader iterates sequentially, so we track the global index
                img_id = coords_df.iloc[current_idx + i]["id"]

                p = preds[i].cpu().numpy()
                t = masks[i].cpu().numpy()

                # Compute intersection and union for this tile
                inter = np.sum(p * t)
                union = np.sum(p) + np.sum(t)

                if img_id not in image_stats:
                    image_stats[img_id] = {"inter": 0.0, "union": 0.0}

                image_stats[img_id]["inter"] += inter
                image_stats[img_id]["union"] += union

            current_idx += batch_size

    # 2. Compute Dice per Image
    results = []
    for img_id, stats in image_stats.items():
        u = stats["union"]
        i = stats["inter"]
        # Dice = 2*I / U. If U is 0 (empty pred and empty truth), Dice is 1.0
        dice = (2.0 * i) / u if u > 0 else 1.0
        results.append({"id": img_id, "dice": dice, "error": 1.0 - dice})

    results_df = pd.DataFrame(results)

    # 3. Merge with Metadata
    # We join on 'id' to get patient/image features
    analysis_df = pd.merge(results_df, val_df, on="id", how="left")

    # 4. Correlation Analysis
    # Define numerical columns of interest
    num_cols = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
        "width_pixels",
        "height_pixels",
    ]

    print("\nCorrelation between Model Error (1 - Dice) and Features:")
    corrs = {}
    for col in num_cols:
        if col in analysis_df.columns:
            # Check for variance to avoid warnings
            if analysis_df[col].nunique() > 1:
                corr = analysis_df["error"].corr(analysis_df[col])
                corrs[col] = corr
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: N/A (No variance)")

    if corrs:
        strongest = max(corrs, key=lambda k: abs(corrs[k]))
        print(
            f"\nStrongest predictor of failure: {strongest} (Correlation: {corrs[strongest]:.4f})"
        )
    else:
        print("\nNo valid correlations computed.")


def validate_full_images(model, val_df, device):
    """
    Performs validation on full images using the inference pipeline (TTA + Overlap).
    This matches the submission process and provides a more accurate metric.
    """
    print("\nPerforming Full-Image Validation with TTA...")
    model.eval()

    total_inter = 0.0
    total_union = 0.0

    # Ensure mask directory exists for loading GT
    mask_dir = os.path.join(Config.WORKING_DIR, "masks")

    for _, row in val_df.iterrows():
        image_id = row["id"]
        image_path = row["image_path"]
        anat_path = row["anatomical_json_path"]
        h, w = row["height_pixels"], row["width_pixels"]

        # 1. Run Inference (Predict Slide)
        # This applies TTA, Gaussian weighting, and Anatomical filtering
        pred_mask = predict_slide(model, image_path, anat_path, device)

        # 2. Load Ground Truth
        # Try loading from npy first (faster), else decode
        npy_path = os.path.join(mask_dir, f"{image_id}.npy")
        if os.path.exists(npy_path):
            gt_mask = np.load(npy_path)
        else:
            gt_mask = rle_decode(row["encoding"], (h, w))

        # 3. Accumulate Stats
        inter, union = compute_intersection_union(pred_mask, gt_mask)
        total_inter += inter
        total_union += union

        # Cleanup
        del pred_mask, gt_mask

    dice = calculate_global_dice(total_inter, total_union)
    return dice


def main():
    # --- 1. Configuration & Setup ---
    # Adjust configuration for improved performance
    # Cite solution_lesson_node_00006: Increase batch size and training duration
    Config.EPOCHS = 20
    Config.TRAIN_NUM_SAMPLES = 10000
    Config.BATCH_SIZE = 12  # Increased from 4 to 12 to fit A100

    Config.setup()
    set_seed(Config.SEED)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Train Samples={Config.TRAIN_NUM_SAMPLES}, Batch={Config.BATCH_SIZE}"
    )

    # --- 2. Data Loading ---
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    print("Initializing DataLoaders...")
    # load_cached_data=True allows skipping coordinate generation if already done
    dataloaders = get_dataloaders(train_df, val_df, test_df)

    # --- 3. Model Initialization ---
    print("Initializing Model (U-Net++ with ResNet34)...")
    device = torch.device(Config.DEVICE)
    model = ResNet34UNetPlusPlus(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # --- 4. Training ---
    print("Starting Training Loop...")
    best_model_path = run_training(
        model,
        dataloaders,
        optimizer,
        scheduler,
        num_epochs=Config.EPOCHS,
        device=device,
    )

    # --- 5. Validation ---
    print("Performing Final Validation...")
    # Load the best model checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Calculate Global Dice Score on Validation Set using Full Image Inference
    # Cite solution_lesson_node_00012: Use TTA and sliding window for final validation
    final_metric = validate_full_images(model, val_df, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    analyze_failures(model, dataloaders["val"], val_df, device)

    # --- 7. Submission ---
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.9347

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.5f}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # Free up memory before inference
        del model, dataloaders, optimizer
        torch.cuda.empty_cache()

        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.5f}) does not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
