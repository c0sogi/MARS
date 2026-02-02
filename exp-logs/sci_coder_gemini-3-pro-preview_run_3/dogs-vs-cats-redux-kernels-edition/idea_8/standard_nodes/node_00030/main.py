import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, setup_directories
from library import utils, engine, models, dataset, inference
from library.dataset import DogCatDataset, get_transforms
from torch.utils.data import DataLoader
from library.inference import predict_with_tta


def main():
    # 1. Setup and Configuration
    setup_directories()
    utils.set_seed(Config.SEED)

    # Adjust configuration for a fast baseline execution
    # Reducing epochs to 5 ensures the pipeline completes within the 2-hour limit
    # while utilizing the A100 GPU efficiently.
    Config.NUM_EPOCHS = 5

    print("Starting Decoupled Multi-Resolution Hybrid Ensemble Pipeline")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs per model: {Config.NUM_EPOCHS}")
    print(f"Models: {list(Config.MODEL_SPECS.keys())}")

    # 2. Training Phase
    # Train each model independently. engine.train_model handles the loop and saving.
    for model_key in Config.MODEL_SPECS.keys():
        try:
            engine.train_model(model_key, device=Config.DEVICE)
        except Exception as e:
            print(f"Critical error training {model_key}: {e}")
            # We continue to try other models to salvage the run if one fails
            continue

    # 3. Ensemble Validation Phase
    print("\n=== Performing Ensemble Validation ===")

    if not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(f"Validation metadata not found at {Config.VAL_CSV}")

    val_df = pd.read_csv(Config.VAL_CSV)
    val_labels = val_df["label"].values

    # Array to store accumulated probabilities from all models
    ensemble_probs = np.zeros(len(val_df))
    valid_models_count = 0

    for model_key, spec in Config.MODEL_SPECS.items():
        ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_key}_best.pth")

        if not os.path.exists(ckpt_path):
            print(
                f"Warning: Checkpoint for {model_key} not found. Skipping in validation."
            )
            continue

        print(f"Validating {model_key}...")

        # Setup Validation Data Loader (Resolution Specific)
        img_size = spec["img_size"]
        val_dataset = DogCatDataset(
            split="val",
            img_size=img_size,
            transform=get_transforms(img_size, is_train=False),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Load Model
        model = models.create_model(model_key, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model = model.to(Config.DEVICE)

        # Run Inference with TTA
        # predict_with_tta returns (ids, preds). For validation, 'ids' corresponds to labels.
        # We only need the probabilities here.
        _, probs = predict_with_tta(model, val_loader, Config.DEVICE)

        ensemble_probs += np.array(probs)
        valid_models_count += 1

    if valid_models_count == 0:
        raise RuntimeError("No models were successfully trained. Cannot proceed.")

    # Compute Ensemble Average
    ensemble_probs /= valid_models_count

    # Calculate and Print Final Metric
    final_metric = log_loss(val_labels, ensemble_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis Phase
    print("\n=== Failure Analysis ===")

    # Calculate absolute prediction error
    errors = np.abs(val_labels - ensemble_probs)

    # Extract image metadata for correlation analysis
    print("Extracting metadata features for correlation analysis...")
    widths = []
    heights = []
    file_sizes = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        w, h, s = 0, 0, 0
        if os.path.exists(full_path):
            s = os.path.getsize(full_path)
            try:
                # Read image dimensions
                img = cv2.imread(full_path)
                if img is not None:
                    h, w, _ = img.shape
            except Exception:
                pass

        widths.append(w)
        heights.append(h)
        file_sizes.append(s)

    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    file_sizes = np.array(file_sizes)

    # Compute Pearson correlations
    # We check if error is correlated with image dimensions or file size
    corr_w, _ = pearsonr(errors, widths)
    corr_h, _ = pearsonr(errors, heights)
    corr_s, _ = pearsonr(errors, file_sizes)

    print(f"Correlation between Error and Image Width: {corr_w:.6f}")
    print(f"Correlation between Error and Image Height: {corr_h:.6f}")
    print(f"Correlation between Error and File Size: {corr_s:.6f}")

    # 5. Submission Phase
    THRESHOLD = 0.009311713870561527

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )
        # Use the library function to generate submission
        # It handles loading models, TTA, caching, and saving to CSV
        inference.ensemble_predictions(device=Config.DEVICE, load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT lower than threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
