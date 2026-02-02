import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add library path if needed
sys.path.append(".")

from library.config import Config, seed_everything
from library.pipeline import run_cv_training
from library.models import TeacherLinkNet
from library.dataset import get_loaders, process_data
from library.utils import unpad_image, calc_map, rle_encode
from library.training import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("--- Starting 5-Fold CV Ensemble Pipeline (Image + Depth) ---")

    # 2. Run Training
    # Returns list of paths to saved models
    model_paths = run_cv_training(debug=False)

    if not model_paths:
        print("No models trained. Exiting.")
        return

    # 3. Ensemble Validation
    print("\n--- Performing Ensemble Validation ---")

    # Load Validation Data (Full Set for final check, though OOF is better, we use Val set from split)
    # Note: get_loaders returns fixed split. For CV ensemble, we should ideally use OOF predictions.
    # But to match the "Final Validation" requirement of the prompt which implies a hold-out set check:
    # We will load the fixed validation set defined in metadata/val.csv and predict using the ensemble.

    _, val_loader, _ = get_loaders(load_cached_data=True)

    # Load Models
    models = []
    for path in model_paths:
        m = TeacherLinkNet(num_classes=1).to(Config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        m.eval()
        models.append(m)

    all_preds_prob = []
    all_masks = []
    all_depths = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE).float()
            masks = batch["mask"].numpy()
            depths = batch["depth"].to(Config.DEVICE).float()

            # Ensemble Prediction
            batch_probs = torch.zeros((images.size(0), 128, 128), device=Config.DEVICE)

            for m in models:
                # Standard Forward
                logits = m(images, depths)
                probs = torch.sigmoid(logits).squeeze(1)

                # TTA: Horizontal Flip
                images_flip = torch.flip(images, dims=[3])
                logits_flip = m(images_flip, depths)
                probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3]).squeeze(1)

                batch_probs += (probs + probs_flip) / 2.0

            batch_probs /= len(models)
            batch_probs = batch_probs.cpu().numpy()

            for i in range(len(batch_probs)):
                p = unpad_image(batch_probs[i])
                m_un = unpad_image(masks[i].squeeze(0))

                all_preds_prob.append(p)
                all_masks.append(m_un)
                all_depths.append(depths[i].cpu().item())

    all_preds_prob = np.array(all_preds_prob)
    all_masks = np.array(all_masks)
    all_depths = np.array(all_depths)

    # Optimize Threshold on Ensemble
    best_map = 0.0
    best_thresh = 0.5
    thresholds = np.arange(0.3, 0.72, 0.02)

    for t in thresholds:
        binary_preds = (all_preds_prob > t).astype(np.uint8)
        score = calc_map(binary_preds, all_masks)
        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Optimized Ensemble Threshold: {best_thresh:.4f}")
    print(f"Final Validation Metric (Ensemble): {best_map:.10f}")

    # 4. Failure Analysis
    binary_preds = (all_preds_prob > best_thresh).astype(np.uint8)
    intersection = np.sum(binary_preds & all_masks, axis=(1, 2))
    union = np.sum(binary_preds | all_masks, axis=(1, 2))
    ious = np.ones_like(intersection, dtype=np.float32)
    valid_mask = union > 0
    ious[valid_mask] = intersection[valid_mask] / union[valid_mask]
    errors = 1.0 - ious

    if np.std(errors) > 0 and np.std(all_depths) > 0:
        correlation = np.corrcoef(errors, all_depths)[0, 1]
    else:
        correlation = 0.0
    print(f"Failure Analysis - Correlation (Error vs Depth): {correlation:.10f}")

    # 5. Submission
    if best_map > 0.7985:
        print("Validation metric meets threshold. Generating submission...")

        # Load Test Data
        _, _, test_loader = get_loaders(load_cached_data=True)

        submission_data = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(Config.DEVICE).float()
                depths = batch["depth"].to(Config.DEVICE).float()
                ids = batch["id"]

                batch_probs = torch.zeros(
                    (images.size(0), 128, 128), device=Config.DEVICE
                )

                for m in models:
                    logits = m(images, depths)
                    probs = torch.sigmoid(logits).squeeze(1)

                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = m(images_flip, depths)
                    probs_flip = torch.flip(
                        torch.sigmoid(logits_flip), dims=[3]
                    ).squeeze(1)

                    batch_probs += (probs + probs_flip) / 2.0

                batch_probs /= len(models)
                batch_probs = batch_probs.cpu().numpy()

                for i, img_id in enumerate(ids):
                    p = unpad_image(batch_probs[i])
                    mask = (p > best_thresh).astype(np.uint8)
                    rle = rle_encode(mask)
                    submission_data.append([img_id, rle])

        df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Validation metric {best_map} <= 0.7985. Discarding submission.")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
