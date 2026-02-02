import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.dataset import SaltDataset
from library.utils import seed_everything, unpad_image, rle_encode, calc_map_score


def get_per_image_map(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates Average Precision for each image across thresholds.
    """
    # Ensure inputs are binary
    preds = (preds > 0).astype(np.uint8)
    targets = (targets > 0).astype(np.uint8)

    ious = []
    for pred, target in zip(preds, targets):
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)  # (N,)

    # Matches: (N, n_thresh)
    # Check if IoU > threshold for each threshold
    matches = ious[:, None] > thresholds[None, :]

    # AP per image: mean over thresholds
    ap_per_image = np.mean(matches, axis=1)
    return ap_per_image


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for this run to ensure it fits within time limits while maximizing performance
    Config.EPOCHS = 50

    print("Initializing Trainer...")
    # debug=False to use full dataset for high score
    trainer = Trainer(debug=False)

    # 2. Train
    print("Starting Training...")
    trainer.start()

    # 3. Load Best Model for Inference
    print("Loading best model for validation...")
    model = trainer.model
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    model.eval()

    # 4. Validation & Failure Analysis
    print("Running Validation Inference...")
    val_dataset = SaltDataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds_prob = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, masks, ids in val_loader:
            images = images.to(Config.DEVICE)

            # TTA: Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            # TTA: Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            # To CPU
            avg_probs = avg_probs.cpu().numpy()
            masks = masks.numpy()

            for i in range(len(ids)):
                # Unpad
                # Shape is (C, H, W) -> need (H, W, C) for unpad
                p_t = np.transpose(avg_probs[i], (1, 2, 0))
                m_t = np.transpose(masks[i], (1, 2, 0))

                p_orig = unpad_image(p_t, Config.ORIG_SIZE).squeeze()
                m_orig = unpad_image(m_t, Config.ORIG_SIZE).squeeze()

                all_preds_prob.append(p_orig)
                all_targets.append(m_orig)
                all_ids.append(ids[i])

    all_preds_prob = np.array(all_preds_prob)
    all_targets = np.array(all_targets)

    # Threshold Optimization
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_map = 0.0
    best_thresh = 0.5

    for t in thresholds:
        binary_preds = (all_preds_prob > t).astype(np.uint8)
        score = calc_map_score(binary_preds, all_targets)
        if score > best_map:
            best_map = score
            best_thresh = t

    print(f"Final Validation Metric: {best_map:.10f}")
    print(f"Optimal Threshold: {best_thresh:.2f}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate per-image mAP at best threshold
    binary_preds_best = (all_preds_prob > best_thresh).astype(np.uint8)
    per_image_scores = get_per_image_map(binary_preds_best, all_targets)
    errors = 1.0 - per_image_scores

    # Load Metadata
    meta_df = pd.read_csv(Config.VAL_METADATA)
    # Map ID to features
    id_map = meta_df.set_index("id").to_dict("index")

    depths = []
    coverages = []
    valid_errors = []

    for i, img_id in enumerate(all_ids):
        if img_id in id_map:
            info = id_map[img_id]
            depths.append(info["z"])
            coverages.append(info["coverage"])
            valid_errors.append(errors[i])

    if len(valid_errors) > 1:
        corr_depth, _ = pearsonr(valid_errors, depths)
        corr_cov, _ = pearsonr(valid_errors, coverages)
        print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
        print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 5. Submission
    if best_map > 0.806:
        print("Metric > 0.806. Generating Submission...")

        test_dataset = SaltDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(Config.DEVICE)

                # TTA
                probs = torch.sigmoid(model(images))

                images_flip = torch.flip(images, dims=[3])
                probs_flip = torch.sigmoid(model(images_flip))
                probs_flip = torch.flip(probs_flip, dims=[3])

                avg_probs = (probs + probs_flip) / 2.0
                avg_probs = avg_probs.cpu().numpy()

                for i in range(len(ids)):
                    p_t = np.transpose(avg_probs[i], (1, 2, 0))
                    p_orig = unpad_image(p_t, Config.ORIG_SIZE).squeeze()

                    # Binarize
                    binary = (p_orig > best_thresh).astype(np.uint8)

                    # RLE
                    rle = rle_encode(binary)
                    submission_rows.append([ids[i], rle])

        sub_df = pd.DataFrame(submission_rows, columns=["id", "rle_mask"])
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Validation metric {best_map:.4f} <= 0.806. Submission skipped.")


if __name__ == "__main__":
    main()
