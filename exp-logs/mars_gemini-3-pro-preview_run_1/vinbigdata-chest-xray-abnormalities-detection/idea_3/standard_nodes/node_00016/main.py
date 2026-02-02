import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import SpatiallyAwareCenterNet
from library.engine import train_model, evaluate, predict_and_submit
from library.loss import CenterNetLoss


def analyze_failures(model, dataloader, device):
    """
    Performs failure analysis on the validation set by correlating
    model loss with input features (e.g., number of boxes, area).
    """
    print("Running Failure Analysis...")
    model.eval()
    criterion = CenterNetLoss()

    results = []

    # Disable gradient calculation for analysis
    with torch.no_grad():
        for batch_idx, (images, targets, image_ids) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # We calculate loss per image to estimate "error magnitude"
            # Since the loss function is designed for batches, we iterate
            # through the batch items individually for this analysis.
            for i in range(len(images)):
                # Slice output tensors to maintain (1, C, H, W) dimensions
                single_output = {k: v[i : i + 1] for k, v in outputs.items()}

                # Slice target: list of dicts -> list containing one dict
                single_target = [targets[i]]

                # Compute loss for this single image
                loss_stats = criterion(single_output, single_target)
                total_loss = loss_stats["loss"].item()

                # Extract Metadata Features
                tgt = targets[i]
                n_boxes = len(tgt["boxes"])

                # Calculate average area of GT boxes (if any)
                if n_boxes > 0:
                    # Boxes are [x1, y1, x2, y2]
                    areas = (tgt["boxes"][:, 2] - tgt["boxes"][:, 0]) * (
                        tgt["boxes"][:, 3] - tgt["boxes"][:, 1]
                    )
                    avg_area = areas.mean().item()
                else:
                    avg_area = 0.0

                # Aspect ratio of original image
                orig_w, orig_h = tgt["orig_size"]
                aspect_ratio = float(orig_w) / float(orig_h)

                results.append(
                    {
                        "image_id": image_ids[i],
                        "loss": total_loss,
                        "num_gt_boxes": n_boxes,
                        "avg_gt_area": avg_area,
                        "img_aspect_ratio": aspect_ratio,
                        "has_finding": 1 if n_boxes > 0 else 0,
                    }
                )

            # Limit analysis to a subset of validation to save time (e.g., first 500 images)
            if len(results) >= 500:
                break

    df_analysis = pd.DataFrame(results)

    if len(df_analysis) == 0:
        print("No analysis data collected.")
        return

    # Calculate Correlations
    print("Correlation between Error (Loss) and Input Features:")
    features = ["num_gt_boxes", "avg_gt_area", "img_aspect_ratio", "has_finding"]
    correlations = df_analysis[features].corrwith(df_analysis["loss"])

    print(correlations)

    # Identify highest correlation
    max_corr_feat = correlations.abs().idxmax()
    print(
        f"Feature most associated with error: {max_corr_feat} (Corr: {correlations[max_corr_feat]:.4f})"
    )


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation (Subsampling for Fast Baseline)
    # To ensure the script completes within the time limit, we train on a subset.
    full_train_df = pd.read_csv(Config.TRAIN_META_PATH)

    # Sample ~2500 images (approx 25% of training data)
    unique_img_ids = full_train_df["image_id"].unique()
    sample_size = min(2500, len(unique_img_ids))

    # Random sampling of image IDs
    sampled_img_ids = np.random.choice(unique_img_ids, size=sample_size, replace=False)
    train_subset_df = full_train_df[
        full_train_df["image_id"].isin(sampled_img_ids)
    ].copy()

    # Save temporary subset metadata
    subset_train_path = os.path.join(Config.WORK_DIR, "train_meta_subset.csv")
    train_subset_df.to_csv(subset_train_path, index=False)
    print(f"Training on subset of {len(sampled_img_ids)} images.")

    # Create DataLoaders
    # We use the FULL validation set to ensure the metric is comparable and valid
    dataloaders = get_dataloaders(
        train_meta=subset_train_path,
        val_meta=Config.VAL_META_PATH,
        test_meta=Config.TEST_META_PATH,
    )

    # 3. Model Initialization
    model = SpatiallyAwareCenterNet(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 4. Training
    # We limit to 5 epochs for the fast baseline check
    n_epochs = 5
    train_model(model, dataloaders, device, epochs=n_epochs)

    # 5. Final Evaluation
    # Load the best model checkpoint saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint.get('epoch')}")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    print("Running final validation on full validation set...")
    val_map = evaluate(model, dataloaders["val"], device)

    # REQUIRED: Print the final metric in the exact format
    print(f"Final Validation Metric: {val_map}")

    # 6. Failure Analysis
    analyze_failures(model, dataloaders["val"], device)

    # 7. Submission
    # Generate submission only if metric exceeds threshold
    threshold = 0.1783551866

    if val_map > threshold:
        print(f"\nValidation metric {val_map} > {threshold}. Generating submission...")
        predict_and_submit(model, dataloaders["test"], device)
    else:
        print(f"\nValidation metric {val_map} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
