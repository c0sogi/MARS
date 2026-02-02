import os
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloader
from library.model import PlantConvNeXt
from library.train import run_training
from library.inference import generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print("Initializing pipeline...")
    set_seed(Config.SEED)

    # Define constraints for Fast Baseline within 2 hours
    # We estimate we can train on ~150,000 images for 3 epochs.
    # 150k images * 3 epochs / 32 batch size * ~0.35s/batch ~= 82 mins training time.
    SUBSET_SIZE = 150000
    NUM_EPOCHS = 3

    # Create a training subset to ensure we finish in time
    print(f"Creating training subset of {SUBSET_SIZE} samples...")
    if os.path.exists(Config.TRAIN_CSV):
        df_train = pd.read_csv(Config.TRAIN_CSV)
        if len(df_train) > SUBSET_SIZE:
            df_subset = df_train.sample(
                n=SUBSET_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)
        else:
            df_subset = df_train

        subset_csv_path = os.path.join(Config.WORK_DIR, "train_subset.csv")
        df_subset.to_csv(subset_csv_path, index=False)

        # Override Config to use the subset
        Config.TRAIN_CSV = subset_csv_path
        print(f"Training data set to {subset_csv_path}")
    else:
        print(f"Error: Train metadata not found at {Config.TRAIN_CSV}")
        return

    # Override Epochs and Scheduler settings for short run
    Config.EPOCHS = NUM_EPOCHS
    Config.PCT_START = 0.3

    print(f"Configured for {Config.EPOCHS} epochs.")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    print("\n==== Starting Training ====")
    # run_training will use the modified Config values
    run_training(epochs=Config.EPOCHS, save_path=Config.BEST_MODEL_PATH)

    # ---------------------------------------------------------
    # 3. Validation & Metrics
    # ---------------------------------------------------------
    print("\n==== Starting Validation ====")
    device = Config.DEVICE

    # Load Best Model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model not found.")
        return

    model = PlantConvNeXt()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Data
    val_loader = get_dataloader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Use AMP for faster inference
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute Metric
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    # Print full precision as requested
    print(f"Final Validation Metric: {val_f1}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n==== Failure Analysis ====")
    # Calculate error (0 for correct, 1 for incorrect)
    errors = (all_preds != all_targets).astype(int)

    # We analyze a subset of validation samples to save time on image I/O
    # Sample 2000 indices for feature extraction
    num_samples = len(all_targets)
    analysis_size = min(2000, num_samples)
    indices = np.random.choice(num_samples, size=analysis_size, replace=False)

    print(f"Analyzing {analysis_size} samples for feature correlation...")

    # Get file paths from dataset
    val_dataset = val_loader.dataset
    file_paths = val_dataset.file_paths

    widths = []
    heights = []
    file_sizes = []
    subset_errors = errors[indices]

    for idx in indices:
        rel_path = file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        w, h, size = 0, 0, 0
        try:
            # File size
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                # Read image for dimensions
                img = cv2.imread(full_path)
                if img is not None:
                    h, w, _ = img.shape
        except Exception:
            pass

        widths.append(w)
        heights.append(h)
        file_sizes.append(size)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": subset_errors,
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
        }
    )

    # Filter out failed reads (width=0)
    df_analysis = df_analysis[df_analysis["width"] > 0]

    if not df_analysis.empty:
        corr_w = df_analysis["error"].corr(df_analysis["width"])
        corr_h = df_analysis["error"].corr(df_analysis["height"])
        corr_s = df_analysis["error"].corr(df_analysis["file_size"])

        print("Correlation between Error Magnitude and Input Features:")
        print(f"  Width: {corr_w}")
        print(f"  Height: {corr_h}")
        print(f"  File Size: {corr_s}")
    else:
        print("Could not compute correlations (no valid images read).")

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    print("\n==== Submission Generation ====")
    THRESHOLD = 0.5930838412243743

    if val_f1 > THRESHOLD:
        print(
            f"Validation F1 ({val_f1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model_path=Config.BEST_MODEL_PATH)
    else:
        print(
            f"Validation F1 ({val_f1}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
