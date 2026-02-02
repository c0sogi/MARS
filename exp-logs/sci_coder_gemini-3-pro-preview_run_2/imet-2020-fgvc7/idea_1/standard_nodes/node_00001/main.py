import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import set_seed
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkClassifier
from library.train import fit
from library.inference import predict_and_submit


def main():
    # --- 1. Setup ---
    # Initialize configuration and set seeds for reproducibility
    Config.setup()
    set_seed(Config.seed)

    print("Starting execution of runfile.py...")

    # --- 2. Training ---
    print("\n--- Phase 1: Training ---")
    # Train the model using the library's fit function.
    # This will train for Config.epochs (5) and save the best model to Config.MODEL_PATH.
    fit(
        epochs=Config.epochs,
        batch_size=Config.batch_size,
        learning_rate=Config.learning_rate,
        debug=Config.debug,
        num_workers=Config.num_workers,
    )

    # --- 3. Validation & Failure Analysis ---
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    device = torch.device(Config.device)

    # Load the best model checkpoint saved during training
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    print(f"Loading best model from {Config.MODEL_PATH}")
    model = ArtworkClassifier(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=False,  # Load structure only, weights come from checkpoint
    )
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Prepare Validation Loader
    # We use shuffle=False to ensure order matches metadata for analysis
    val_dataset = ArtworkDataset(
        metadata_path=Config.VAL_METADATA,
        input_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="valid", image_size=Config.image_size),
        mode="valid",
        num_classes=Config.num_classes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)

            # Use mixed precision for inference speed
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Move to CPU and convert to float32 for metric calculation
            all_preds.append(probs.float().cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # Binarize predictions with threshold 0.5
    binary_preds = (all_preds > 0.5).astype(int)
    val_micro_f1 = f1_score(all_targets, binary_preds, average="micro")

    # Print required metric with full precision
    print(f"Final Validation Metric: {val_micro_f1}")

    # Failure Analysis
    print("Performing failure analysis...")

    # Calculate error magnitude per sample (Hamming distance)
    # This is the count of incorrect labels (False Positives + False Negatives) per image
    error_magnitude = np.sum(np.abs(binary_preds - all_targets), axis=1)

    # Load metadata to get features for correlation
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Feature 1: Ground Truth Label Count (Complexity of the annotation)
    gt_label_counts = np.sum(all_targets, axis=1)

    # Feature 2: File Size (Proxy for image complexity/quality)
    # Construct full paths to read file stats
    file_paths = [os.path.join(Config.INPUT_DIR, p) for p in val_df["file_path"]]
    file_sizes = [os.path.getsize(p) for p in file_paths]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": error_magnitude,
            "gt_label_count": gt_label_counts,
            "file_size": file_sizes,
        }
    )

    # Calculate correlations
    corr_count = analysis_df["error"].corr(analysis_df["gt_label_count"])
    corr_size = analysis_df["error"].corr(analysis_df["file_size"])

    print(f"Correlation (Error vs GT Label Count): {corr_count}")
    print(f"Correlation (Error vs File Size): {corr_size}")

    # --- 4. Submission ---
    print("\n--- Phase 3: Submission ---")
    # Generate predictions for the test set and save submission.csv
    predict_and_submit(
        model_path=Config.MODEL_PATH,
        metadata_path=Config.TEST_METADATA,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        device=Config.device,
        threshold=0.5,
    )

    print("Runfile execution complete.")


if __name__ == "__main__":
    main()
