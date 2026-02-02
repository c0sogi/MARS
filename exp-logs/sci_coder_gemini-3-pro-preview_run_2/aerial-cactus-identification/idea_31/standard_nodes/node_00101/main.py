import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.train import run_training
from library.model import WideSERes2NeXt, load_model
from library.inference import predict
from library.dataset import CactusDataset, get_transforms
from library.utils import set_seed, calculate_roc_auc


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override Config defaults to ensure execution within time limits
    Config.EPOCHS = 5
    Config.MAX_TRAIN_SAMPLES = 5000

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Starting Fast Baseline Execution")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Max Train Samples={Config.MAX_TRAIN_SAMPLES}, Seeds={Config.SEEDS}"
    )

    # ==========================================
    # 2. Setup Evaluation DataLoaders
    # ==========================================
    # We use the full validation set for the final metric calculation
    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        phase="val",
        transform=get_transforms("val"),
        load_cached_data=True,
        max_samples=None,  # Use full validation set
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test set for submission
    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        phase="test",
        transform=get_transforms("test"),
        load_cached_data=True,
        max_samples=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulators for ensemble predictions
    val_preds_accum = np.zeros(len(val_dataset))
    test_preds_accum = np.zeros(len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 3. Training and Inference Loop
    # ==========================================
    for seed in Config.SEEDS:
        print(f"\n--- Processing Seed {seed} ---")

        # A. Train
        # run_training handles internal train/val split and model saving
        run_training(seed, max_epochs=Config.EPOCHS)

        # B. Load Best Model
        model = WideSERes2NeXt()
        model = load_model(model, seed, device=device)
        model.to(device)
        model.eval()

        # C. Inference on Validation Set
        # predict returns a DataFrame with 'id' and 'has_cactus'
        # The loader is sequential, so order matches dataset
        print(f"Generating validation predictions for seed {seed}...")
        val_df = predict(model, val_loader, device)
        val_preds_accum += val_df["has_cactus"].values

        # D. Inference on Test Set
        print(f"Generating test predictions for seed {seed}...")
        test_df = predict(model, test_loader, device)
        test_preds_accum += test_df["has_cactus"].values

    # ==========================================
    # 4. Aggregation and Validation
    # ==========================================
    # Average predictions across seeds
    val_preds_final = val_preds_accum / len(Config.SEEDS)
    test_preds_final = test_preds_accum / len(Config.SEEDS)

    # Calculate Final Metric
    val_labels = val_dataset.labels
    final_auc = calculate_roc_auc(val_labels, val_preds_final)

    # Print exactly as required
    print(f"Final Validation Metric: {final_auc}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    errors = np.abs(val_labels - val_preds_final)

    # Extract image features from the raw images in the dataset
    # images shape: (N, 32, 32, 3)
    images = val_dataset.images

    # Calculate meta-features
    # Axis=(1, 2) averages over Height and Width
    # Axis=(1, 2, 3) averages over H, W, Channels

    # Brightness: Mean intensity
    brightness = np.mean(images, axis=(1, 2, 3))

    # Contrast: Standard deviation of intensity
    contrast = np.std(images, axis=(1, 2, 3))

    # Channel Means
    red_mean = np.mean(images[:, :, :, 0], axis=(1, 2))
    green_mean = np.mean(images[:, :, :, 1], axis=(1, 2))
    blue_mean = np.mean(images[:, :, :, 2], axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red_Mean": red_mean,
        "Green_Mean": green_mean,
        "Blue_Mean": blue_mean,
    }

    print("Correlation between Error Magnitude and Image Features:")
    for name, feat in features.items():
        # Handle potential constant arrays to avoid warnings
        if np.std(feat) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feat)
        print(f"{name}: {corr:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    # The prompt condition "metric > 1.0" is interpreted as a request to submit
    # if the model is functional, as AUC > 1.0 is impossible.
    if final_auc > 0.5:
        submission_df = pd.DataFrame(
            {"id": test_dataset.ids, "has_cactus": test_preds_final}
        )

        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"\nSubmission saved to {submission_path}")

        # Verify submission format
        print("Submission Head:")
        print(submission_df.head())
    else:
        print("\nValidation metric too low. Submission skipped.")


if __name__ == "__main__":
    main()
