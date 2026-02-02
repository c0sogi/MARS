"""
Main execution script for Whale Species Prediction.
Orchestrates training, validation, failure analysis, and submission.
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, map5_metric, WhaleLabelEncoder
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet
from library.train import train_model, get_logits_inference
from library.inference import run_inference

# -------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -------------------------------------------------------------------------
# We override the default configuration to ensure the run completes within
# the time limits while still providing a robust baseline.
# Reducing epochs and using a single seed for the "Fast Baseline" requirement.
Config.STAGE_1_EPOCHS = 8
Config.STAGE_2_EPOCHS = 4
Config.ENSEMBLE_SEEDS = [42]  # Single model for baseline


def analyze_failures(val_df, preds, targets):
    """
    Performs failure analysis on the validation set.
    Calculates the correlation between model error and input image features.

    Args:
        val_df (pd.DataFrame): Validation metadata.
        preds (list): List of predicted top-5 indices.
        targets (list): List of true target indices.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate per-sample error
    # Score is 1/(rank+1) if target in preds, else 0
    # Error = 1.0 - Score
    errors = []

    for p, t in zip(preds, targets):
        score = 0.0
        # p is a numpy array or list of indices
        p_list = p.tolist() if isinstance(p, np.ndarray) else p

        if t in p_list:
            rank = p_list.index(t)
            score = 1.0 / (rank + 1)

        errors.append(1.0 - score)

    errors = np.array(errors)

    # 2. Extract Image Metadata (Features)
    # We need to read the original images to get dimensions and intensity
    feature_stats = []

    print("Extracting metadata features for correlation analysis...")

    for idx, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Default values in case of read failure
        w, h, intensity = 0, 0, 0.0

        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                # Convert to RGB for consistency in intensity calc (though mean is same)
                # Just use simple mean
                intensity = img.mean() / 255.0
        except Exception:
            pass

        feature_stats.append(
            {
                "Width": w,
                "Height": h,
                "AspectRatio": (w / h) if h > 0 else 0,
                "MeanIntensity": intensity,
            }
        )

    meta_df = pd.DataFrame(feature_stats)
    meta_df["Error"] = errors

    # 3. Calculate Correlations
    print("Correlation between Model Error and Input Features:")
    results = {}
    for feature in ["Width", "Height", "AspectRatio", "MeanIntensity"]:
        corr = meta_df[feature].corr(meta_df["Error"])
        results[feature] = corr
        print(f"  {feature}: {corr:.6f}")

    return results


def main():
    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("Starting Training Phase...")
    # Train the model(s) defined in Config
    for seed in Config.ENSEMBLE_SEEDS:
        train_model(seed)

    # -------------------------------------------------------------------------
    # 2. Validation Phase
    # -------------------------------------------------------------------------
    print("\nStarting Validation Phase...")
    device = torch.device(Config.DEVICE)

    # Load Label Encoder (cached from training)
    label_encoder = WhaleLabelEncoder()
    label_encoder.fit([], load_cached_data=True)
    num_classes = label_encoder.num_classes()

    # Setup Validation Data
    val_dataset = WhaleDataset(
        Config.VAL_CSV,
        label_encoder=label_encoder,
        transform=get_transforms("val", Config.STAGE_2_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the Trained Model
    # We use the first seed since we only trained one for the baseline
    seed = Config.ENSEMBLE_SEEDS[0]
    model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    if not os.path.exists(model_path):
        print(f"Critical Error: Model file not found at {model_path}")
        return

    model = WhaleDenseNet(
        num_classes=num_classes,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=False,
        dropout_rate=Config.DROPOUT_RATE,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
    )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)

            # Get logits (Cosine Similarity) with TTA
            logits = get_logits_inference(model, images, device, tta=Config.TTA_FLIP)

            # Get Top 5 predictions
            preds = torch.topk(logits, 5)[1].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    # Compute Final Metric
    final_metric = map5_metric(all_preds, all_targets)

    # Print Metric (Strict Format)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(val_dataset.df, all_preds, all_targets)

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6545824094604581

    if final_metric > THRESHOLD:
        print(f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Proceeding to generate submission for Test Set...")

        # Clear memory
        del model, val_loader, val_dataset, images, logits
        torch.cuda.empty_cache()

        # Run Inference Module
        # Note: run_inference uses Config.ENSEMBLE_SEEDS, which we overrode to [42]
        run_inference(debug=Config.DEBUG, load_cached_data=True)

    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
