import os
import sys
import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, do_kaggle_metric, calculate_iou_batch
from library.train import Trainer
from library.predict import Predictor
from library.dataset import SaltDataset
from library.model import DeepResUNet


def main():
    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    print("--- Starting Full Training Pipeline ---")
    print(
        f"Configuration: {Config.NUM_EPOCHS} Epochs, {Config.EPOCHS_PER_CYCLE} Epochs/Cycle"
    )

    # Execute Training
    trainer = Trainer()
    trainer.run()

    # -------------------------------------------------------------------------
    # 3. Validation Assessment (Ensemble)
    # -------------------------------------------------------------------------
    print("\n--- Performing Final Validation (Ensemble) ---")

    device = torch.device(Config.DEVICE)
    val_dataset = SaltDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Models for Ensemble (Same logic as Predictor)
    models = []
    ckpt_names = ["best_cycle_2.pth", "swa_model.pth"]

    for name in ckpt_names:
        path = os.path.join(Config.CHECKPOINT_DIR, name)
        if os.path.exists(path):
            m = DeepResUNet().to(device)
            state_dict = torch.load(path, map_location=device)

            # Handle 'module.' prefix from SWA/DataParallel wrappers
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                elif k == "n_averaged":
                    continue
                else:
                    new_state_dict[k] = v

            m.load_state_dict(new_state_dict)
            m.eval()
            models.append(m)

    # Fallback
    if not models:
        path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(path):
            m = DeepResUNet().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)

    if not models:
        raise RuntimeError("No trained models found for validation.")

    # Inference Loop
    all_preds = []
    all_targets = []
    all_depths = []

    # Padding calculations for cropping (128 -> 101)
    pad_total = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_top = pad_total // 2
    pad_left = pad_total // 2
    crop_h = Config.IMG_SIZE - (pad_total - pad_top)
    crop_w = Config.IMG_SIZE - (pad_total - pad_left)

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)

            # Extract depth for failure analysis (Channel 1, any spatial location)
            # images shape: (B, 2, 128, 128)
            batch_depths = images[:, 1, 0, 0].cpu().numpy()
            all_depths.append(batch_depths)

            # Ensemble Prediction
            batch_probs_sum = None
            for model in models:
                # Standard
                logits = model(images)
                probs = torch.sigmoid(logits)

                # TTA (Horizontal Flip)
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip_back = torch.flip(probs_flip, dims=[3])

                probs = (probs + probs_flip_back) / 2.0

                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs

            avg_probs = batch_probs_sum / len(models)

            # Crop to original size
            cropped_probs = avg_probs[..., pad_top:crop_h, pad_left:crop_w]
            cropped_masks = masks[..., pad_top:crop_h, pad_left:crop_w]

            all_preds.append(cropped_probs.cpu().numpy())
            all_targets.append(cropped_masks.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    all_targets = np.concatenate(all_targets, axis=0).squeeze(1)
    all_depths = np.concatenate(all_depths, axis=0)

    # Calculate Final Metric
    final_metric = do_kaggle_metric(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Performing Failure Analysis ---")

    # Error Metric: 1 - IoU (at threshold 0.5)
    ious = calculate_iou_batch(all_preds, all_targets)
    errors = 1.0 - ious

    # Feature: Salt Coverage (Ground Truth)
    # Mean of binary mask gives proportion of salt pixels
    coverages = all_targets.mean(axis=(1, 2))

    # Correlations
    # Note: all_depths is normalized [0,1] due to dataset preprocessing, which is fine for correlation
    if len(errors) > 1:
        corr_depth, _ = pearsonr(errors, all_depths)
        corr_cov, _ = pearsonr(errors, coverages)
    else:
        corr_depth, corr_cov = 0.0, 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    if final_metric > 0.833:
        print("\nMetric > 0.833. Generating Submission...")
        predictor = Predictor()
        predictor.predict()
    else:
        print(f"\nMetric {final_metric} <= 0.833. Skipping Submission.")


if __name__ == "__main__":
    main()
