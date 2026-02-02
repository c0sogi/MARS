import os
import torch
import numpy as np
import pandas as pd

from library import config, dataset, utils, model, losses


def optimize_threshold(net, val_loader, device=config.DEVICE):
    """
    Finds the optimal binarization threshold based on validation set performance.
    Performs a linear search over a range of thresholds to maximize the competition metric.

    Args:
        net (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (str): Device to run computation on.

    Returns:
        float: The optimal threshold value.
    """
    print("Optimizing threshold on validation set...")
    net.eval()
    criterion = losses.SaltNetLoss()

    # Use the validate function from library.model to get predictions
    # validate returns: val_loss, val_map, all_preds, all_masks
    # all_preds are already sigmoid probabilities and cropped to 101x101
    _, _, val_preds, val_masks = model.validate(net, val_loader, criterion, device)

    best_thresh = 0.5
    best_score = 0.0

    # Search range 0.2 to 0.85 with step 0.05
    thresholds = np.arange(0.2, 0.85, 0.05)

    for t in thresholds:
        score = utils.do_kaggle_metric(val_preds, val_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Optimized Threshold: {best_thresh} (Val mAP: {best_score})")

    return best_thresh


def generate_submission(net, test_loader, threshold, device=config.DEVICE):
    """
    Generates predictions for the test set and saves the submission file.
    Applies Test-Time Augmentation (TTA) and center cropping.

    Args:
        net (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        threshold (float): Binarization threshold.
        device (str): Device to run computation on.
    """
    print("Generating predictions on test set...")
    net.eval()
    submission_data = []

    with torch.no_grad():
        for images, ids, depths in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Forward Pass (Original)
            logits = net(images, depths)
            probs = torch.sigmoid(logits)

            # TTA: Horizontal Flip Pass
            images_flip = torch.flip(images, dims=[3])
            logits_flip = net(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average Predictions
            avg_probs = (probs + probs_flip) / 2.0

            # Center Crop to Original Size (101x101)
            # Model output is 128x128, Target is 101x101
            h, w = avg_probs.shape[2], avg_probs.shape[3]
            target_size = config.IMG_ORIG_SIZE
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            end_h = start_h + target_size
            end_w = start_w + target_size

            avg_probs = avg_probs[:, :, start_h:end_h, start_w:end_w]

            # Binarize
            preds_bin = (avg_probs > threshold).cpu().numpy().astype(np.uint8)

            # Encode to RLE
            for i in range(len(ids)):
                # preds_bin shape is (B, 1, H, W), take first channel
                mask = preds_bin[i, 0]
                rle = utils.rle_encode(mask)
                submission_data.append([ids[i], rle])

    # Save Submission
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def predict(depth_mean=None, depth_std=None):
    """
    Main prediction pipeline.

    Args:
        depth_mean (float, optional): Mean depth from training set.
        depth_std (float, optional): Std dev of depth from training set.
    """
    utils.set_seed(config.SEED)
    device = config.DEVICE

    # If depth stats are not provided, calculate them from training data
    # This ensures consistent normalization with training
    if depth_mean is None or depth_std is None:
        print("Calculating depth statistics from training set...")
        _, _, depth_mean, depth_std = dataset.get_train_val_loaders(
            load_cached_data=True
        )

    # Load Model
    net = model.WideLinkNet34().to(device)
    if os.path.exists(config.CHECKPOINT_PATH):
        net.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model from {config.CHECKPOINT_PATH}")
    else:
        print("Warning: No checkpoint found. Inference will use random weights.")

    # 1. Optimize Threshold
    # We need the validation loader for this
    _, val_loader, _, _ = dataset.get_train_val_loaders(load_cached_data=True)
    best_thresh = optimize_threshold(net, val_loader, device)

    # 2. Generate Submission
    test_loader = dataset.get_test_loader(depth_mean, depth_std, load_cached_data=True)
    generate_submission(net, test_loader, best_thresh, device)
