import sys
import os
import numpy as np
import torch
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.dataset import CactusDataset, get_valid_transforms
from library.model import CustomSEResNet
from library.utils import seed_everything, calculate_roc_auc


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast but effective baseline
    # 20 epochs is sufficient for convergence on this small dataset
    Config.EPOCHS = 20

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Seeds: {Config.SEEDS}")
    print(f"  Model: {Config.MODEL_NAME}")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n>>> Starting Training Pipeline...")
    run_training()

    # ==========================================
    # 3. Validation & Evaluation
    # ==========================================
    print("\n>>> Starting Validation & Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load Validation Dataset
    # We use get_valid_transforms (no augmentation) for consistent metric calculation
    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_valid_transforms(),
        mode="val",
        load_cached_data=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Perform Ensemble Prediction on Validation Set
    # We aggregate predictions from all available seeds
    ensemble_probs = np.zeros(len(val_dataset))
    successful_seeds = 0

    for seed in Config.SEEDS:
        model_path = Config.get_model_path(seed)
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        # Load Model
        model = CustomSEResNet(**Config.MODEL_PARAMS)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Generate predictions
        seed_probs = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                seed_probs.append(probs.cpu().numpy())

        ensemble_probs += np.concatenate(seed_probs).flatten()
        successful_seeds += 1

    if successful_seeds == 0:
        print("Error: No models trained successfully. Exiting.")
        return

    # Average probabilities
    ensemble_probs /= successful_seeds

    # Calculate Final Metric (ROC AUC)
    val_labels = val_dataset.labels
    final_metric = calculate_roc_auc(val_labels, ensemble_probs)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n>>> Performing Failure Analysis...")

    # Calculate error magnitude (Absolute Error)
    # Label is 0 or 1, Prob is [0, 1]
    errors = np.abs(val_labels - ensemble_probs)

    # Extract meta-features from validation images
    # val_dataset.images is (N, 32, 32, 3) uint8
    images = val_dataset.images.astype(np.float32)

    # Compute image statistics
    # Mean intensity (Brightness)
    brightness = images.mean(axis=(1, 2, 3))
    # Standard deviation (Contrast)
    contrast = images.std(axis=(1, 2, 3))
    # Channel means
    red_mean = images[:, :, :, 0].mean(axis=(1, 2))
    green_mean = images[:, :, :, 1].mean(axis=(1, 2))
    blue_mean = images[:, :, :, 2].mean(axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    print("Correlation between Error Magnitude and Image Features:")
    for name, feat_vals in features.items():
        # Calculate Pearson correlation
        corr, _ = pearsonr(feat_vals, errors)
        print(f"  {name}: {corr:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    # The prompt requires metric > 1.0 which is impossible for AUC (max 1.0).
    # Proceeding with submission if metric is reasonable (> 0.5).
    if final_metric > 0.5:
        print("\n>>> Generating Submission File...")
        run_inference()
    else:
        print("\nValidation metric is too low. Skipping submission.")


if __name__ == "__main__":
    main()
