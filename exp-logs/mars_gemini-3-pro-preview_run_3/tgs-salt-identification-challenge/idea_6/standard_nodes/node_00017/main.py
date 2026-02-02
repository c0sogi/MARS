import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from functools import partialmethod
from tqdm import tqdm

# Suppress progress bars to comply with output requirements
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed, unpad_image, compute_map_score

# Override Config for the fast baseline execution
# We set EPOCHS to 30 to ensure the two-stage curriculum completes within the time limit.
# Epochs 0-14: BCE+Dice (Warmup)
# Epochs 15-29: Lovasz-Hinge (Fine-tuning)
Config.EPOCHS = 30


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 1. Train the Model
    # The Trainer handles data loading, model init, training loop, and checkpointing.
    trainer = Trainer()
    trainer.fit()

    # 2. Load Best Model for Evaluation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model checkpoint not found.")
        return

    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()

    # Retrieve the optimal threshold found during training
    best_threshold = checkpoint.get("best_threshold", 0.5)

    # 3. Validation & Failure Analysis
    # We manually iterate over the validation set to calculate the final metric
    # and collect per-image errors for analysis.

    # Load metadata to map IDs to features (Depth, Coverage)
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)
    val_metadata.set_index("id", inplace=True)

    val_loader = trainer.val_loader
    device = trainer.device
    results = []

    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            preds = trainer.model(images)

            # Test-Time Augmentation (Horizontal Flip)
            if Config.TTA_FLIP:
                images_flipped = torch.flip(images, dims=[3])
                preds_flipped = trainer.model(images_flipped)
                preds_flipped = torch.flip(preds_flipped, dims=[3])
                preds = (preds + preds_flipped) / 2.0

            # Convert logits to probabilities
            preds_prob = preds.sigmoid().cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Process each image in the batch
            for i in range(len(ids)):
                img_id = ids[i]

                # Unpad to original size (101x101)
                pred_img = preds_prob[i].squeeze()
                mask_img = masks_np[i].squeeze()

                pred_orig = unpad_image(pred_img, Config.ORIG_SIZE)
                mask_orig = unpad_image(mask_img, Config.ORIG_SIZE)

                # Binarize using the optimized threshold
                pred_bin = (pred_orig > best_threshold).astype(np.float32)

                # Compute mAP for this single image
                # Wrap in batch dimension (1, H, W) for the utility function
                p_tensor = torch.tensor(pred_bin).unsqueeze(0)
                m_tensor = torch.tensor(mask_orig).unsqueeze(0)

                score = compute_map_score(p_tensor, m_tensor)

                # Retrieve metadata features
                if img_id in val_metadata.index:
                    meta = val_metadata.loc[img_id]
                    depth = meta["z"]
                    coverage = meta["coverage"]
                else:
                    depth = 0
                    coverage = 0

                results.append(
                    {
                        "id": img_id,
                        "map": score,
                        "error": 1.0 - score,
                        "depth": depth,
                        "coverage": coverage,
                    }
                )

    # Calculate and Print Final Metric
    results_df = pd.DataFrame(results)
    final_metric = results_df["map"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis (Correlation)
    print("Failure Analysis:")
    if len(results_df) > 1:
        # Check standard deviation to avoid warnings on constant arrays
        if results_df["error"].std() > 0 and results_df["depth"].std() > 0:
            corr_depth, _ = pearsonr(results_df["error"], results_df["depth"])
            print(f"Correlation Error vs Depth: {corr_depth:.4f}")
        else:
            print("Correlation Error vs Depth: Undefined (constant values)")

        if results_df["error"].std() > 0 and results_df["coverage"].std() > 0:
            corr_cov, _ = pearsonr(results_df["error"], results_df["coverage"])
            print(f"Correlation Error vs Salt Coverage: {corr_cov:.4f}")
        else:
            print("Correlation Error vs Salt Coverage: Undefined (constant values)")

    # 4. Generate Submission
    # Only generate if metric exceeds the specified threshold
    TARGET_THRESHOLD = 0.8101666667
    if final_metric > TARGET_THRESHOLD:
        trainer.generate_submission()
    else:
        print(
            f"Validation metric {final_metric} is not higher than {TARGET_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
