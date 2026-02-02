import os
import gc
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, do_kaggle_metric, rle_encode
from library.model import UNetPlusPlus
from library.dataset import get_fold_loaders, get_test_loader
from library.train import train_fold


def load_model_for_inference(fold_idx, device):
    """Loads the best checkpoint for a specific fold in evaluation mode."""
    model = UNetPlusPlus()
    model.to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def predict_with_tta(model, inputs):
    """Predicts with Test-Time Augmentation (Horizontal Flip)."""
    # 1. Original
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits)

    # 2. Flipped
    inputs_flipped = torch.flip(inputs, dims=[3])
    with torch.no_grad():
        logits_flipped = model(inputs_flipped)
        probs_flipped = torch.sigmoid(logits_flipped)

    # Flip back
    probs_flipped_back = torch.flip(probs_flipped, dims=[3])

    # Average
    return (probs + probs_flipped_back) / 2.0


def get_oof_predictions(fold_idx, device):
    """Generates OOF predictions for a fold using the best model."""
    # Re-retrieve the validation loader for this fold
    _, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

    model = load_model_for_inference(fold_idx, device)

    preds_list = []
    masks_list = []
    depths_list = []

    # Crop parameters to revert padding (128 -> 101)
    h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    h_end = h_start + Config.ORIG_SIZE
    w_end = w_start + Config.ORIG_SIZE

    # Access metadata directly from dataset to ensure alignment
    # val_loader is not shuffled, so order matches dataset indices
    dataset_depths = val_loader.dataset.depths

    batch_start_idx = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        batch_size = inputs.size(0)

        # Inference with TTA
        probs = predict_with_tta(model, inputs)

        # Crop to original size
        probs = probs[:, :, h_start:h_end, w_start:w_end]
        targets = targets[:, :, h_start:h_end, w_start:w_end]

        # Move to CPU
        preds_list.append(probs.cpu().numpy().squeeze(1))
        masks_list.append(targets.cpu().numpy().squeeze(1))

        # Collect metadata
        depths_list.append(
            dataset_depths[batch_start_idx : batch_start_idx + batch_size]
        )
        batch_start_idx += batch_size

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return (
        np.concatenate(preds_list),
        np.concatenate(masks_list),
        np.concatenate(depths_list),
    )


def optimize_threshold(preds, masks):
    """Sweeps thresholds to find the best mAP."""
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        score = do_kaggle_metric(preds, masks, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def run_failure_analysis(preds, masks, depths, threshold):
    """Correlates prediction errors with metadata."""
    print("Performing failure analysis...")

    # Calculate per-image mAP
    pred_mask = (preds > threshold).astype(np.uint8)
    truth_mask = (masks > 0.5).astype(np.uint8)

    N = pred_mask.shape[0]
    pred_flat = pred_mask.reshape(N, -1)
    truth_flat = truth_mask.reshape(N, -1)

    intersection = (pred_flat & truth_flat).sum(axis=1)
    union = (pred_flat | truth_flat).sum(axis=1)

    iou = np.ones(N)
    mask_u = union > 0
    iou[mask_u] = intersection[mask_u] / union[mask_u]

    # mAP calculation per image
    thresholds_iou = np.arange(0.5, 0.96, 0.05)
    matches = iou[:, None] > thresholds_iou[None, :]
    image_scores = matches.mean(axis=1)

    # Error is 1 - Score
    errors = 1.0 - image_scores

    # Salt Coverage (ratio of salt pixels)
    coverages = truth_flat.mean(axis=1)

    # Correlations
    corr_depth, _ = pearsonr(depths, errors)
    corr_cov, _ = pearsonr(coverages, errors)

    print(f"Correlation (Depth vs Error): {corr_depth:.4f}")
    print(f"Correlation (Salt Coverage vs Error): {corr_cov:.4f}")


def generate_submission(threshold, device):
    """Generates submission file using 5-fold ensemble and TTA."""
    print("Generating submission for test set...")

    test_loader = get_test_loader(load_cached_data=True)
    n_test = len(test_loader.dataset)
    test_ids = test_loader.dataset.ids

    # Accumulator for ensemble probabilities
    # Shape: (N, 101, 101) - Pre-allocate to avoid list overhead
    ensemble_probs = np.zeros(
        (n_test, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.float32
    )

    # Crop parameters
    h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    h_end = h_start + Config.ORIG_SIZE
    w_end = w_start + Config.ORIG_SIZE

    # Iterate through all 5 folds
    for fold_idx in range(Config.FOLDS):
        print(f"Inference with Fold {fold_idx}...")
        model = load_model_for_inference(fold_idx, device)

        batch_start = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Predict with TTA
            probs = predict_with_tta(model, inputs)

            # Crop and accumulate
            probs = probs[:, :, h_start:h_end, w_start:w_end]
            ensemble_probs[batch_start : batch_start + batch_size] += (
                probs.cpu().numpy().squeeze(1)
            )

            batch_start += batch_size

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average over folds
    ensemble_probs /= Config.FOLDS

    # Binarize
    binary_masks = (ensemble_probs > threshold).astype(np.uint8)

    # Encode
    submission_data = []
    for i in range(n_test):
        rle = rle_encode(binary_masks[i])
        submission_data.append([test_ids[i], rle])

    # Save
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 1. Train Folds
    print("Starting Training...")
    oof_preds_all = []
    oof_masks_all = []
    oof_depths_all = []

    for fold in range(Config.FOLDS):
        # Train the fold
        train_fold(fold, debug=False)

        # Generate OOF predictions
        preds, masks, depths = get_oof_predictions(fold, device)
        oof_preds_all.append(preds)
        oof_masks_all.append(masks)
        oof_depths_all.append(depths)

    # Concatenate OOF data
    oof_preds = np.concatenate(oof_preds_all)
    oof_masks = np.concatenate(oof_masks_all)
    oof_depths = np.concatenate(oof_depths_all)

    # 2. Optimize Threshold
    best_thresh, best_score = optimize_threshold(oof_preds, oof_masks)
    print(f"Final Validation Metric: {best_score}")

    # 3. Failure Analysis
    run_failure_analysis(oof_preds, oof_masks, oof_depths, best_thresh)

    # 4. Submission
    if best_score > 0.827:
        generate_submission(best_thresh, device)
    else:
        print(f"Validation score {best_score} is not above 0.827. Skipping submission.")


if __name__ == "__main__":
    main()
