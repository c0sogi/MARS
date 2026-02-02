import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import configuration and utilities from the provided library
from library.config import (
    SEED,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    NUM_EPOCHS,
    WORKING_DIR,
    INPUT_DIR,
    METADATA_DIR,
    THRESHOLD,
)
from library.utils import seed_everything, fbeta_score
from library.dataset import InkDataset
from library.model import InkDetectorFCN
from library.engine import train_model
from library.inference import (
    generate_submission,
    load_volume,
    tiled_inference,
    load_normalization_stats,
)


def run():
    # 1. Initialization
    seed_everything(SEED)
    print("Starting Ink Detection Pipeline...")

    # 2. Data Loading
    # Increased patches_per_epoch to provide more data variety per epoch.
    print("Loading Datasets...")
    train_dataset = InkDataset("train", patches_per_epoch=12000, load_cached_data=True)
    val_dataset = InkDataset("val", patches_per_epoch=4000, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = InkDetectorFCN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Path to save the best model weights
    save_path = WORKING_DIR / "best_model.pth"

    # 4. Training
    print("Starting Training...")
    # Cite solution_lesson_node_00006: The Inefficacy of Extending Patience on Noisy Validation Metrics.
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        patience=3,  # Reduced patience to prevent overfitting/wasted compute
        save_path=save_path,
    )

    # 5. Full Validation Evaluation & Failure Analysis
    print("Performing Full Validation Inference...")

    # Load the best model for evaluation
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    # Load validation metadata
    val_meta_path = METADATA_DIR / "val.csv"
    if not val_meta_path.exists():
        print("Validation metadata not found. Skipping full validation.")
        return

    val_df = pd.read_csv(val_meta_path)

    # Load normalization stats generated during training
    mean, std = load_normalization_stats()

    total_score = 0.0
    count = 0

    # Containers for failure analysis
    all_errors = []
    all_means = []
    all_stds = []

    for _, row in val_df.iterrows():
        frag_id = str(row["fragment_id"])
        vol_path = row["surface_volume_path"]
        mask_path = row["mask_path"]
        label_path = row["inklabels_path"]

        print(f"Evaluating Validation Fragment {frag_id}...")

        # Load and Normalize Volume
        try:
            volume = load_volume(vol_path)
        except Exception as e:
            print(f"Error loading volume {vol_path}: {e}")
            continue

        volume_norm = (volume.astype(np.float32) - mean) / (std + 1e-6)

        # Run Inference on the full volume
        prob_map = tiled_inference(model, volume_norm, DEVICE)

        # Load Ground Truth
        valid_mask = cv2.imread(str(INPUT_DIR / mask_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(INPUT_DIR / label_path), cv2.IMREAD_GRAYSCALE)

        # Ensure dimensions match (resize if necessary, though unlikely)
        if valid_mask.shape != prob_map.shape:
            valid_mask = cv2.resize(
                valid_mask,
                (prob_map.shape[1], prob_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if label.shape != prob_map.shape:
            label = cv2.resize(
                label,
                (prob_map.shape[1], prob_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply Valid Mask
        valid_mask_bin = valid_mask > 0
        prob_map = prob_map * valid_mask_bin
        label_bin = (label > 0).astype(np.float32)

        # Compute Metric
        prob_t = torch.from_numpy(prob_map).unsqueeze(0)  # (1, H, W)
        label_t = torch.from_numpy(label_bin).unsqueeze(0)

        score = fbeta_score(prob_t, label_t, beta=0.5, threshold=THRESHOLD)
        total_score += score
        count += 1

        # --- Failure Analysis Data Collection ---
        # Calculate pixel-wise L1 error
        error_map = np.abs(prob_map - label_bin)

        # Calculate input features (Mean and Std intensity across Z-depth)
        # We use the raw volume for feature extraction
        vol_mean = np.mean(volume, axis=0)
        vol_std = np.std(volume, axis=0)

        # Select indices within the valid mask to sample
        valid_indices = np.where(valid_mask_bin)
        n_valid = len(valid_indices[0])

        # Subsample to keep analysis fast (e.g., 100,000 pixels max per fragment)
        if n_valid > 100000:
            idx = np.random.choice(n_valid, 100000, replace=False)
            sampled_y = valid_indices[0][idx]
            sampled_x = valid_indices[1][idx]
        else:
            sampled_y = valid_indices[0]
            sampled_x = valid_indices[1]

        # Collect samples
        all_errors.extend(error_map[sampled_y, sampled_x])
        all_means.extend(vol_mean[sampled_y, sampled_x])
        all_stds.extend(vol_std[sampled_y, sampled_x])

    # Print Final Metric
    final_metric = total_score / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Print Failure Analysis
    print("Performing Failure Analysis...")
    if len(all_errors) > 0:
        errors = np.array(all_errors)
        means = np.array(all_means)
        stds = np.array(all_stds)

        # Calculate correlations
        # We check if error correlates with pixel intensity or contrast (std)
        corr_mean, _ = pearsonr(errors, means)
        corr_std, _ = pearsonr(errors, stds)

        print(f"Correlation (Error vs Pixel Intensity Mean): {corr_mean:.10f}")
        print(f"Correlation (Error vs Pixel Intensity Std): {corr_std:.10f}")
    else:
        print("Insufficient data for failure analysis.")

    # 6. Submission Generation
    baseline_score = 0.38412588834762573
    if final_metric > baseline_score:
        print(
            f"Validation score {final_metric:.4f} > baseline {baseline_score:.4f}. Generating Submission..."
        )
        generate_submission(save_path)
    else:
        print(
            f"Validation score {final_metric:.4f} did not improve upon baseline {baseline_score:.4f}. Skipping submission."
        )

    print("Pipeline Completed.")


if __name__ == "__main__":
    run()
