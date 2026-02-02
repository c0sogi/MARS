import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import cv2
import torchvision.transforms.functional as TF

# Import from provided library files
from library.config import (
    MODEL_CONFIGS,
    data_config,
    train_config,
    WORKING_DIR,
    SUBMISSION_DIR,
    INPUT_DIR,
    SEED,
)
from library.utils import seed_everything, get_device
from library.engine import train_model
from library.models import CustomEnsembleModel
from library.transforms import get_transforms
from library.dataset import DogCatDataset


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Preparation (Subsampling for Fast Baseline)
    print("Preparing training data...")
    # Load original train csv
    full_train_df = pd.read_csv(data_config.train_csv)

    # Subsample to 5000 images to ensure quick execution as per requirements
    mini_train_df = full_train_df.sample(n=5000, random_state=SEED).reset_index(
        drop=True
    )

    # Save mini train csv to working directory
    mini_train_path = os.path.join(WORKING_DIR, "mini_train.csv")
    mini_train_df.to_csv(mini_train_path, index=False)

    # Update data config to point to the subsampled data
    data_config.train_csv = mini_train_path

    # Update train config for fast baseline
    train_config.epochs = 5  # Reduced epochs for speed

    # 3. Train Models
    trained_models = []

    for config in MODEL_CONFIGS:
        print(f"\n=== Training Model: {config.model_name} ===")
        # Train the model
        train_model(config, data_config, train_config)

        # Load the best checkpoint for inference
        safe_model_name = config.model_name.replace(".", "_")
        checkpoint_path = os.path.join(WORKING_DIR, safe_model_name, "model_best.pth")

        print(f"Loading best weights from {checkpoint_path}")
        model = CustomEnsembleModel(
            config=config, num_classes=data_config.num_classes, pretrained=False
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        trained_models.append((config, model))

    # 4. Validation Inference (Ensemble + TTA)
    print("\n=== Running Validation Inference ===")
    val_df = pd.read_csv(data_config.val_csv)
    val_preds = run_inference(trained_models, val_df, mode="val")

    # Calculate Metric
    val_labels = val_df["label"].values
    # Clip predictions to avoid log(0)
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_labels, val_preds_clipped)

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    analyze_failures(val_df, val_labels, val_preds_clipped)

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.009241249605204765

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        test_df = pd.read_csv(data_config.test_csv)
        test_preds = run_inference(trained_models, test_df, mode="test")

        submission = pd.DataFrame({"id": test_df["id"], "label": test_preds})

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission generation."
        )


def run_inference(models, df, mode="val"):
    """
    Runs inference using an ensemble of models with Test Time Augmentation (TTA).
    """
    device = get_device()
    ensemble_preds = np.zeros(len(df))

    for config, model in models:
        # Create a dataloader specific to this model's input size
        # Validation/Test transform is deterministic resize
        transform = get_transforms(config.input_size, mode="val")
        dataset = DogCatDataset(df, transform=transform, mode=mode)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size * 2,  # Double batch size for inference
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model_preds = []
        with torch.no_grad():
            for batch in loader:
                if mode == "test":
                    images, _ = batch
                else:
                    images, _ = batch

                images = images.to(device)

                # TTA Strategy: Average (Original + Horizontal Flip)

                # 1. Original
                logits = model(images)
                probs = torch.sigmoid(logits)

                # 2. Flipped
                images_flipped = TF.hflip(images)
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                avg_probs = (probs + probs_flipped) / 2.0
                model_preds.append(avg_probs.cpu().numpy())

        # Concatenate predictions for this model
        model_preds = np.concatenate(model_preds).flatten()
        ensemble_preds += model_preds

    # Average across all models in the ensemble
    return ensemble_preds / len(models)


def analyze_failures(df, labels, preds):
    """
    Analyzes correlations between error magnitude and image metadata.
    """
    # Calculate absolute error
    errors = np.abs(labels - preds)

    widths = []
    heights = []
    file_sizes = []
    aspect_ratios = []

    print("Collecting metadata for failure analysis...")
    # Iterate through validation set to get image stats
    # Using a simple loop as dataset size is manageable (4500)
    for idx, row in df.iterrows():
        path = os.path.join(INPUT_DIR, row["filepath"])
        try:
            # File size
            fsize = os.path.getsize(path)

            # Dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w, _ = img.shape
            else:
                h, w = 0, 0

            widths.append(w)
            heights.append(h)
            file_sizes.append(fsize)

            ar = w / h if h > 0 else 0
            aspect_ratios.append(ar)

        except Exception:
            widths.append(0)
            heights.append(0)
            file_sizes.append(0)
            aspect_ratios.append(0)

    # Calculate correlations using NumPy
    meta_features = {
        "width": widths,
        "height": heights,
        "file_size": file_sizes,
        "aspect_ratio": aspect_ratios,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in meta_features.items():
        if len(values) != len(errors):
            continue

        # Pearson correlation
        # np.corrcoef returns a matrix, [0,1] is the correlation between x and y
        corr = np.corrcoef(values, errors)[0, 1]
        print(f"  {name}: {corr:.4f}")


if __name__ == "__main__":
    main()
