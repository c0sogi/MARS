import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import CatDogDataset
from library.models import build_model
from library.train import train_all_models
from library.inference import run_ensemble


def validate_ensemble():
    """
    Performs validation on the ensemble of trained models.
    Returns the Log Loss metric, the ensemble predictions, and the targets.
    """
    print("Starting Ensemble Validation...")
    device = torch.device(Config.DEVICE)

    # Load Validation Dataset
    val_dataset = CatDogDataset(Config.VAL_CSV, phase="val", debug=Config.DEBUG)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Store predictions from each model
    model_predictions = []

    for model_name in Config.MODELS:
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for {model_name} not found. Skipping.")
            continue

        print(f"Validating model: {model_name}")

        # Build and load model
        model = build_model(model_name, pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()

        preds = []

        # Inference with TTA (Horizontal Flip)
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)

                # Forward pass 1: Original
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # Forward pass 2: Flipped
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

                # Average
                probs_avg = (probs_orig + probs_flip) / 2.0
                preds.extend(probs_avg.cpu().view(-1).numpy())

        model_predictions.append(preds)

        # Clean up
        del model
        torch.cuda.empty_cache()

    if not model_predictions:
        print("No models available for validation.")
        return 1.0, None, None

    # Ensemble Averaging
    ensemble_preds = np.mean(model_predictions, axis=0)

    # Get Ground Truth
    targets = val_dataset.df["label"].values

    # Ensure lengths match (debug mode might truncate dataset but loader follows dataset)
    if len(ensemble_preds) != len(targets):
        print(f"Shape mismatch: Preds {len(ensemble_preds)} vs Targets {len(targets)}")
        min_len = min(len(ensemble_preds), len(targets))
        ensemble_preds = ensemble_preds[:min_len]
        targets = targets[:min_len]

    # Calculate Log Loss
    # Clip predictions to avoid log(0)
    ensemble_preds_clipped = np.clip(ensemble_preds, 1e-15, 1 - 1e-15)
    metric = log_loss(targets, ensemble_preds_clipped)

    return metric, ensemble_preds, targets


def analyze_failures(preds, targets):
    """
    Analyzes the correlation between prediction error and image metadata.
    """
    print("\n--- Failure Analysis ---")

    # Load metadata to get filepaths
    val_df = pd.read_csv(Config.VAL_CSV)
    if Config.DEBUG:
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Calculate Error Magnitude
    errors = np.abs(targets - preds)

    # Extract features on-the-fly
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []

    print(f"Extracting metadata for {len(val_df)} validation images...")

    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        if os.path.exists(full_path):
            # File Size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Compute Correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "File Size": file_sizes,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, data in features.items():
        if len(data) == len(errors):
            # Pearson correlation
            corr = np.corrcoef(errors, data)[0, 1]
            print(f"  {name}: {corr:.8f}")
        else:
            print(
                f"  {name}: Length mismatch (Data: {len(data)}, Errors: {len(errors)})"
            )


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Train Models
    print("--- Starting Training Phase ---")
    train_all_models()

    # 3. Validate Ensemble
    print("\n--- Starting Validation Phase ---")
    metric, preds, targets = validate_ensemble()

    # Print required metric format
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    if preds is not None:
        analyze_failures(preds, targets)

    # 5. Submission Logic
    # Strict threshold from task description
    THRESHOLD = 0.009311713870561527

    print("\n--- Submission Decision ---")
    if metric < THRESHOLD:
        print(
            f"Metric {metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        run_ensemble()
    else:
        print(
            f"Metric {metric} is NOT lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
