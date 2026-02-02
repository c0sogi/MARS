import os
import sys
import random
import numpy as np
import torch
import pandas as pd
import cv2
import nltk
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloader
from library.utils import LevenshteinMetric


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # --- 1. Setup & Configuration ---
    print("Initializing Fast Baseline Run...")

    # Force a fresh start by removing old checkpoints
    if os.path.exists(Config.CHECKPOINT_PATH):
        os.remove(Config.CHECKPOINT_PATH)
    if os.path.exists(Config.BEST_MODEL_PATH):
        os.remove(Config.BEST_MODEL_PATH)

    # Override Config for fast baseline execution
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000  # Train on 50k samples (~3-4% of data)
    Config.BATCH_SIZE = 32  # Reduced from 64 to avoid CUDA OOM
    Config.NUM_WORKERS = 4

    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Training Samples: {Config.DEBUG_SAMPLE_SIZE}")

    # --- 2. Training ---
    # Initialize Trainer (builds model, optimizer, tokenizer)
    trainer = Trainer(load_cached_data=True)

    # Execute training loop
    # Note: We pass debug=True to use the subsampled training set defined in Config
    trainer.fit(epochs=Config.EPOCHS, debug=True)

    # --- 3. Full Validation Assessment ---
    print("\n--- Executing Full Validation Assessment ---")

    # Create a dataloader for the FULL validation set (debug=False)
    # shuffle=False to ensure alignment with metadata for failure analysis
    val_loader = get_dataloader(
        Config.VAL_METADATA, trainer.tokenizer, mode="val", shuffle=False, debug=False
    )

    trainer.model.eval()
    metric = LevenshteinMetric()

    val_preds = []
    val_targets = []
    val_distances = []

    print(f"Validating on {len(val_loader.dataset)} samples...")

    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(val_loader):
            images = images.to(trainer.device)

            # Predict using greedy decoding
            pred_indices = trainer.model.predict(images)

            # Decode to strings
            batch_preds = [trainer.tokenizer.decode(p) for p in pred_indices]
            batch_targets = [trainer.tokenizer.decode(l) for l in labels]

            val_preds.extend(batch_preds)
            val_targets.extend(batch_targets)

            # Update metric
            metric.update(batch_preds, batch_targets)

            # Calculate individual distances for failure analysis
            for p, t in zip(batch_preds, batch_targets):
                dist = nltk.edit_distance(p, t)
                val_distances.append(dist)

            if i % 100 == 0:
                print(f"Validation progress: {i}/{len(val_loader)} batches")

    final_metric = metric.get_avg_score()
    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # Load validation metadata to get input features
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Ensure dataframe length matches predictions (should match if shuffle=False and no drop_last)
    if len(val_df) != len(val_distances):
        print(
            f"Warning: Mismatch in lengths (DF: {len(val_df)}, Preds: {len(val_distances)}). Truncating to min."
        )
        min_len = min(len(val_df), len(val_distances))
        val_df = val_df.iloc[:min_len].copy()
        val_distances = val_distances[:min_len]

    val_df["error_magnitude"] = val_distances
    val_df["target_len"] = val_df["InChI"].str.len()

    # To analyze image features (width, height, aspect ratio), we need to read the images.
    # Reading all 380k images is too slow. We will sample 5000 images for this specific analysis.
    analysis_sample = val_df.sample(n=5000, random_state=Config.SEED).copy()

    widths = []
    heights = []

    print("Extracting image features from sample subset...")
    for _, row in analysis_sample.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image to get dimensions
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)

    analysis_sample["width"] = widths
    analysis_sample["height"] = heights
    analysis_sample["aspect_ratio"] = (
        analysis_sample["width"] / analysis_sample["height"]
    )

    # Drop any failed reads
    analysis_sample = analysis_sample.dropna()

    # Calculate Correlations
    print("Correlations with Error Magnitude (Levenshtein Distance):")

    # 1. Target Sequence Length
    corr_len = analysis_sample["error_magnitude"].corr(analysis_sample["target_len"])
    print(f"  Target Length: {corr_len:.4f}")

    # 2. Image Width
    corr_width = analysis_sample["error_magnitude"].corr(analysis_sample["width"])
    print(f"  Image Width:   {corr_width:.4f}")

    # 3. Image Height
    corr_height = analysis_sample["error_magnitude"].corr(analysis_sample["height"])
    print(f"  Image Height:  {corr_height:.4f}")

    # 4. Aspect Ratio
    corr_ar = analysis_sample["error_magnitude"].corr(analysis_sample["aspect_ratio"])
    print(f"  Aspect Ratio:  {corr_ar:.4f}")

    # --- 5. Submission Generation ---
    print("\n--- Generating Submission ---")
    # Predict on the full test set (debug=False)
    trainer.predict_test(debug=False)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
