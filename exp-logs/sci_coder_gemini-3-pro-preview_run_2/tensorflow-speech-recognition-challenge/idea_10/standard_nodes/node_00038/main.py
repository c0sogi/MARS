import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed, load_checkpoint, get_device
from library.dataset import get_dataloaders, LABELS
from library.model import EfficientNetV2Audio

# ==========================================
# 1. Configuration Adjustments
# ==========================================
# Adjust hyperparameters for a fast but effective baseline run
Config.NUM_EPOCHS = 20


def run():
    # ==========================================
    # 2. Setup & Training
    # ==========================================
    set_seed(Config.SEED)
    device = get_device()

    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training Loop...")
    trainer.fit()

    # ==========================================
    # 3. Validation & Evaluation
    # ==========================================
    print("\nRunning Final Validation...")

    # Initialize a fresh model instance to load the best weights
    model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)

    # Load the best checkpoint (which contains EMA weights)
    checkpoint = load_checkpoint(model, filepath=Config.BEST_MODEL_PATH, device=device)
    if checkpoint is None:
        print("Error: Checkpoint not found. Using current model state.")
        model = trainer.ema.ema  # Fallback

    model.eval()

    # Get validation loader
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []  # Feature for failure analysis (Mean Spectrogram Intensity)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Collect data
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Calculate mean intensity of the spectrogram as a feature
            # images shape: (Batch, 1, Freq, Time)
            # We average over Freq and Time dimensions
            means = images.view(images.size(0), -1).mean(dim=1)
            all_features.append(means.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_features = np.concatenate(all_features)

    # Compute Final Metric
    accuracy = (all_preds == all_labels).mean()
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude: 1.0 - Probability of the True Class
    # Indexing: for each sample i, get prob of class all_labels[i]
    rows = np.arange(len(all_labels))
    true_class_probs = all_probs[rows, all_labels]
    error_magnitudes = 1.0 - true_class_probs

    # Calculate Correlation with Input Feature (Spectrogram Intensity)
    # Check for NaN/Inf just in case
    if np.isfinite(error_magnitudes).all() and np.isfinite(all_features).all():
        correlation, _ = pearsonr(error_magnitudes, all_features)
        print(
            f"Correlation between Error Magnitude and Mean Spectrogram Intensity: {correlation}"
        )
    else:
        print("Warning: NaN or Inf values detected in failure analysis data.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9866209549293419

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        _, _, test_loader = get_dataloaders(load_cached_data=True)
        test_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds)

        # Map indices to string labels
        pred_labels = [LABELS[idx] for idx in test_preds]

        # Retrieve filenames from the dataset dataframe
        fnames = test_loader.dataset.df["fname"].values

        # Create DataFrame
        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation accuracy ({accuracy}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
