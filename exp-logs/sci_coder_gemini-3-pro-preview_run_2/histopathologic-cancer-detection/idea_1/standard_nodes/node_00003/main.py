import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import necessary components from the provided library
from library.config import Config
from library.train import run_training
from library.predict import generate_submission
from library.dataset import PathologyDataset, get_transforms
from library.model import TumorClassifier
from library.utils import set_seed, compute_metrics


def analyze_failures(model, device):
    """
    Performs inference on the validation set to compute the final metric
    and analyzes correlations between error magnitude and input features.
    """
    # Initialize validation dataset and loader
    val_dataset = PathologyDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_transforms("val"),
        debug=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Storage for predictions, labels, and features
    all_preds = []
    all_labels = []

    # Feature lists for correlation analysis
    feat_brightness = []
    feat_contrast = []
    feat_r_mean = []
    feat_g_mean = []
    feat_b_mean = []

    model.eval()

    # Inference loop without gradient computation
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # labels are loaded as float tensors

            # Cite {solution_lesson_node_00002}
            # Test Time Augmentation: Original, HFlip, VFlip, Rot180
            logits_orig = model(images)
            logits_h = model(torch.flip(images, dims=[3]))
            logits_v = model(torch.flip(images, dims=[2]))
            logits_hv = model(torch.flip(images, dims=[2, 3]))

            # Average probabilities
            probs_tensor = (
                torch.sigmoid(logits_orig)
                + torch.sigmoid(logits_h)
                + torch.sigmoid(logits_v)
                + torch.sigmoid(logits_hv)
            ) / 4.0

            probs = probs_tensor.squeeze(1).cpu().numpy()
            labels_np = labels.numpy()

            all_preds.extend(probs)
            all_labels.extend(labels_np)

            # Calculate image features for failure analysis
            # Images are (B, C, H, W)

            # Mean per channel (B, 3)
            batch_means = images.mean(dim=(2, 3)).cpu().numpy()

            # Std per image (B,) - used as a proxy for contrast
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            feat_r_mean.extend(batch_means[:, 0])
            feat_g_mean.extend(batch_means[:, 1])
            feat_b_mean.extend(batch_means[:, 2])

            # Brightness: average of the channel means
            feat_brightness.extend(batch_means.mean(axis=1))

            # Contrast: standard deviation of the image
            feat_contrast.extend(batch_stds)

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. Compute and print Final Validation Metric
    val_auc = compute_metrics(all_labels, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 2. Failure Analysis
    # Calculate error magnitude
    errors = np.abs(all_labels - all_preds)

    print("Failure Analysis (Correlation with Error Magnitude):")
    features = {
        "Brightness": np.array(feat_brightness),
        "Contrast": np.array(feat_contrast),
        "Red Mean": np.array(feat_r_mean),
        "Green Mean": np.array(feat_g_mean),
        "Blue Mean": np.array(feat_b_mean),
    }

    # Compute correlations
    for name, vals in features.items():
        # Ensure variance exists to avoid warnings/NaNs
        if np.std(vals) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(errors, vals)[0, 1]
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: 0.0000 (Insufficient variance)")

    return val_auc


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # We limit epochs to 2 for a fast baseline.
    # With A100, this is sufficient to verify the pipeline and get a decent score.
    print("Starting training pipeline...")
    best_model_path = run_training(epochs=2, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Validation & Failure Analysis
    print("Loading best model for validation and analysis...")
    model = TumorClassifier(pretrained=False)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    val_auc = analyze_failures(model, device)

    # 4. Generate Submission
    # Cite {solution_lesson_node_00002}
    threshold = 0.9808827205496372
    if val_auc > threshold:
        print(
            f"Validation AUC ({val_auc:.6f}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=best_model_path,
            output_path=Config.PREDICTION_FILE,
            device=Config.DEVICE,
            debug=False,
        )
    else:
        print(
            f"Validation AUC ({val_auc:.6f}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
