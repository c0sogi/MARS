import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, calculate_log_loss, load_checkpoint
from library.data import get_dataloaders, load_metadata
from library.modeling import create_model
from library.engine import train_model, generate_submission


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Define the ensemble configuration keys
    model_keys = ["resnet", "convnext", "vit_ssl"]

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("=== Training Phase ===")
    for key in model_keys:
        # train_model handles the full training loop, validation monitoring,
        # and saving the best checkpoint to ./working/{key}_best.pth
        train_model(config_name=key, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 2. Ensemble Validation Phase
    # -------------------------------------------------------------------------
    print("\n=== Ensemble Validation Phase ===")

    # Load ground truth from metadata
    val_df = load_metadata("val", load_cached_data=True)
    y_true = val_df["label"].values.astype(float)

    # Initialize array for ensemble probabilities
    ensemble_preds = np.zeros(len(val_df), dtype=np.float64)

    # Generate predictions for each model on the validation set
    for key in model_keys:
        print(f"Generating validation predictions for: {key}")
        cfg = Config.MODEL_CONFIGS[key]

        # Get validation dataloader (shuffle=False guarantees order matches val_df)
        _, val_loader, _ = get_dataloaders(
            img_size=cfg["img_size"],
            batch_size=cfg["batch_size"],
            load_cached_data=True,
        )

        # Initialize model architecture
        model = create_model(
            model_name=cfg["model_name"],
            num_classes=Config.NUM_CLASSES,
            pretrained=False,  # We load our trained weights
            img_size=cfg["img_size"],
        )
        model.to(Config.DEVICE)
        model.eval()

        # Load the best checkpoint saved during training
        ckpt_path = os.path.join(Config.WORKING_DIR, f"{key}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint for {key} not found. Skipping.")
            continue

        load_checkpoint(ckpt_path, model, device=Config.DEVICE)

        # Inference Loop
        model_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(Config.DEVICE)

                # Forward pass
                outputs = model(images)
                probs = torch.sigmoid(outputs)

                # Test Time Augmentation (Horizontal Flip)
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    outputs_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(outputs_flipped)
                    probs = (probs + probs_flipped) / 2.0

                model_probs.append(probs.cpu().numpy())

        # Flatten and accumulate
        model_probs = np.concatenate(model_probs).flatten()
        ensemble_preds += model_probs

    # Compute Arithmetic Mean
    ensemble_preds /= len(model_keys)

    # Compute and Print Metric
    final_metric = calculate_log_loss(y_true, ensemble_preds)
    print(f"Final Validation Metric: {final_metric:.15f}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis Phase
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis Phase ===")

    # Calculate error magnitude
    errors = np.abs(y_true - ensemble_preds)

    # Extract meta-features from validation images
    print("Extracting image features...")
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []

    for idx, row in val_df.iterrows():
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        if os.path.exists(filepath):
            # File Size
            file_sizes.append(os.path.getsize(filepath))

            # Image Dimensions
            img = cv2.imread(filepath)
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

    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "File Size": file_sizes,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, data in features.items():
        # Using Pearson correlation
        if len(data) == len(errors):
            corr, _ = pearsonr(errors, data)
            print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission Phase
    # -------------------------------------------------------------------------
    print("\n=== Submission Phase ===")
    threshold = 0.009241249605204765

    if final_metric < threshold:
        print(f"Metric meets threshold ({threshold}). Generating submission...")
        generate_submission(model_keys, load_cached_data=True)
    else:
        print(
            f"Metric {final_metric:.15f} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
