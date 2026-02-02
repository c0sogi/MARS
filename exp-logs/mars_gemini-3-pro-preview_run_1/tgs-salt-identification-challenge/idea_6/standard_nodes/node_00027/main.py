import os
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from provided library files
from library.trainer import Trainer
from library.utils import set_seed, do_kaggle_metric, unpad_image_101
from library.dataset import get_dataloaders

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(42)

    print("Initializing Trainer...")
    # We use 150 epochs to allow the 3-cycle Cosine Annealing schedule to complete.
    # Cycle 1: 0-50, Cycle 2: 50-100, Cycle 3: 100-150.
    # On A100, this is very fast.
    trainer = Trainer(
        base_dir="./working/idea_6",
        batch_size=32,
        num_workers=2,
        lr=1e-3,
        epochs=150,
        debug=False,
    )

    # 2. Training
    print("Starting Training...")
    trainer.fit()

    # 3. Validation & Failure Analysis
    print("Performing Validation and Failure Analysis...")

    # Load the best global model for analysis
    best_model_path = os.path.join(trainer.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )

    trainer.model.eval()

    val_loader = trainer.val_loader
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, masks, img_ids in val_loader:
            images = images.to(trainer.device)

            # Forward pass
            logits = trainer.model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
            masks_np = masks.cpu().numpy().squeeze(1)

            # Unpad and store
            for i in range(len(img_ids)):
                p_unpad = unpad_image_101(probs[i])
                t_unpad = unpad_image_101(masks_np[i])
                all_preds.append(p_unpad)
                all_targets.append(t_unpad)
                all_ids.append(img_ids[i])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Final Metric
    final_metric = do_kaggle_metric(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Metadata
    # Load metadata to get features
    try:
        df_val = pd.read_csv("./metadata/val.csv")
        id_to_depth = dict(zip(df_val["id"], df_val["z"]))
        id_to_cov = dict(zip(df_val["id"], df_val["coverage"]))

        scores = []
        depths = []
        coverages = []

        thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)

        # Calculate per-image score (mAP)
        for i in range(len(all_ids)):
            p = all_preds[i]
            t = all_targets[i]

            # Binarize prediction for IoU calculation
            p_bin = (p > 0.5).astype(np.uint8)

            intersection = np.logical_and(t, p_bin).sum()
            union = np.logical_or(t, p_bin).sum()

            if union == 0:
                iou = 1.0 if t.sum() == 0 else 0.0
            else:
                iou = intersection / union

            # Calculate score over thresholds
            # Score is 1 if IoU > threshold, else 0
            pass_thresholds = (iou > thresholds).astype(float)
            img_score = pass_thresholds.mean()

            scores.append(img_score)
            depths.append(id_to_depth.get(all_ids[i], 0))
            coverages.append(id_to_cov.get(all_ids[i], 0))

        scores = np.array(scores)
        errors = 1.0 - scores  # Error magnitude

        # Calculate correlations
        if len(errors) > 1:
            corr_depth, _ = pearsonr(errors, depths)
            corr_cov, _ = pearsonr(errors, coverages)
            print(f"Correlation (Error vs Depth): {corr_depth}")
            print(f"Correlation (Error vs Salt Coverage): {corr_cov}")
        else:
            print("Insufficient data for correlation analysis.")

    except Exception as e:
        print(f"Failure analysis skipped due to error: {e}")

    # 4. Submission
    # Threshold check
    if final_metric > 0.8156666666666668:
        trainer.generate_submission()
    else:
        print(
            f"Validation metric {final_metric} is not higher than 0.8156666666666668. Skipping submission."
        )


if __name__ == "__main__":
    main()
