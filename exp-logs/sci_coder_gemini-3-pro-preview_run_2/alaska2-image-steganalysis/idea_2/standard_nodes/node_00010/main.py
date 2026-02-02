import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.dataset import AlaskaDataset, get_transforms
from library.model import StegoNet
from library.train import fit
from library.inference import generate_submission
from library.utils import seed_everything, alaska_weighted_auc


def perform_failure_analysis(val_df, errors):
    """
    Calculates and prints correlations between error magnitude and input features
    (File Size and Mean Intensity) on the validation set.
    """
    print("Performing failure analysis...")

    file_sizes = []
    mean_intensities = []

    # Iterate through validation dataframe to extract features
    for idx, row in val_df.iterrows():
        file_path = os.path.join(Config.input_root, row["file_path"])

        # Feature 1: File Size
        try:
            f_size = os.path.getsize(file_path)
        except OSError:
            f_size = 0
        file_sizes.append(f_size)

        # Feature 2: Mean Intensity
        try:
            # Read image to calculate mean intensity
            # Using cv2 for speed
            img = cv2.imread(file_path)
            if img is not None:
                # Normalize to [0, 1]
                mean_val = img.mean() / 255.0
            else:
                mean_val = 0.0
        except Exception:
            mean_val = 0.0
        mean_intensities.append(mean_val)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "error_magnitude": errors,
            "file_size": file_sizes,
            "mean_intensity": mean_intensities,
        }
    )

    # Compute correlation of features with error_magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)


def main():
    # 1. Setup & Configuration
    seed_everything(Config.seed)

    # Override Config for Optimized Run
    # Cite solution_lesson_node_00005: Prolonged Convergence Schedules
    Config.epochs = 20

    # Ensure GPU usage
    if torch.cuda.is_available():
        Config.device = torch.device("cuda")
    else:
        Config.device = torch.device("cpu")

    # 2. Train Model
    # fit() handles the training loop, validation monitoring, and saving the best checkpoint
    fit(epochs=Config.epochs)

    # 3. Validation & Metrics
    # Reload the best model to evaluate it properly for the final metric
    best_model_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found. Training may have failed.")
        return

    # Initialize Model Architecture
    model = StegoNet(
        backbone_name=Config.backbone_name,
        pretrained=False,
        num_classes=Config.num_classes,
    )

    # Load Weights
    model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()

    # Prepare Validation Loader
    val_dataset = AlaskaDataset("val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Inference on Validation Set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.device)

            # Forward pass
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(labels.numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    # Calculate Weighted AUC
    metric = alaska_weighted_auc(val_targets, val_probs)

    # Print Metric (Required Format)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    # Calculate error magnitude: |label - prediction|
    errors = np.abs(val_targets - val_probs)
    perform_failure_analysis(val_dataset.df, errors)

    # 5. Submission
    # Generate submission only if metric exceeds threshold
    THRESHOLD = 0.8416

    if metric > THRESHOLD:
        generate_submission(checkpoint_path=best_model_path, debug=Config.debug)
    else:
        print(
            f"Metric ({metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
