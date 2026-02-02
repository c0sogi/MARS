import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import rle_encode, do_kaggle_metric, get_best_threshold
from library.losses import CompositeLoss


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, optimizer, device, epoch, mode="teacher"):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader instance.
        optimizer: Optimizer instance.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (for logging).
        mode: 'teacher' or 'student'.

    Returns:
        avg_loss: The average loss over the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Loss function container
    criterion = CompositeLoss(Config)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{mode}]", leave=False)

    for batch in pbar:
        # Move data to device
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        depths = batch["depth"].to(device)

        # Determine calculation mode for loss
        # Teacher: Always supervised (Hard targets)
        # Student: Check if masks are soft (Unlabeled) or hard (Supervised)
        if mode == "teacher":
            calc_mode = "supervised"
        else:
            # Heuristic: If any mask value is fractional (not 0 or 1), it's soft.
            # We check if values are strictly between 0 and 1 (allowing for float precision).
            # If a mask is purely 0s or 1s (even if from pseudo-labels), treating it as supervised
            # (BCE+Lovasz) is acceptable/beneficial.
            if torch.any((masks > 0.001) & (masks < 0.999)):
                calc_mode = "unlabeled"
            else:
                calc_mode = "supervised"

        optimizer.zero_grad()

        # Forward pass
        if mode == "teacher":
            # Teacher requires depth injection
            outputs = model(images, depths)
        else:
            # Student ignores depth input (uses aux head internally)
            outputs = model(images)

        # Calculate loss
        # outputs can be tensor (teacher) or dict (student)
        loss, _ = criterion(outputs, {"mask": masks, "depth": depths}, mode=calc_mode)

        # Backward
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=loss_meter.avg, mode=calc_mode)

    return loss_meter.avg


def evaluate(model, dataloader, device, mode="teacher"):
    """
    Evaluates the model on the validation set.

    Returns:
        score: mAP metric.
        avg_loss: Average validation loss.
        preds: Numpy array of predictions (N, H, W).
        targets: Numpy array of targets (N, H, W).
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = CompositeLoss(Config)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            if mode == "teacher":
                outputs = model(images, depths)
                logits = outputs
                # Loss calculation (always supervised for val)
                loss, _ = criterion(
                    outputs, {"mask": masks, "depth": depths}, mode="supervised"
                )
            else:
                outputs = model(images)
                logits = outputs["mask"]
                loss, _ = criterion(
                    outputs, {"mask": masks, "depth": depths}, mode="supervised"
                )

            loss_meter.update(loss.item(), images.size(0))

            # Sigmoid for probabilities
            probs = torch.sigmoid(logits)

            # Center crop back to 101x101 (Model outputs 128x128)
            h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
            w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
            probs_cropped = probs[
                :,
                0,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]
            masks_cropped = masks[
                :,
                0,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]

            preds_list.append(probs_cropped.cpu().numpy())
            targets_list.append(masks_cropped.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    # Calculate mAP
    score = do_kaggle_metric(preds, targets)

    return score, loss_meter.avg, preds, targets


def generate_pseudo_labels(teacher_models, test_loader, device, load_cached_data=True):
    """
    Generates soft pseudo-labels for the test set using Marginalized Distillation.

    Args:
        teacher_models: List of loaded Teacher models (ensemble).
        test_loader: DataLoader for test set.
        device: Device.
        load_cached_data: If True, tries to load from disk.

    Returns:
        pseudo_labels: Numpy array (N, 101, 101) aligned with test metadata.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "pseudo_labels.npy")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading pseudo labels from {cache_path}")
        return np.load(cache_path)

    print("Generating pseudo labels with Depth-Scanning...")

    # 2. Compute
    scan_sigmas = Config.DEPTH_SCAN_SIGMAS
    results = {}  # id -> prob_map

    # Ensure models are in eval mode
    for m in teacher_models:
        m.eval()
        m.to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Pseudo-Labeling"):
            images = batch["image"].to(device)
            ids = batch["id"]
            batch_size = images.size(0)

            # Accumulator: (B, 1, 128, 128)
            accum_probs = torch.zeros(
                (batch_size, 1, Config.IMG_SIZE, Config.IMG_SIZE), device=device
            )

            # Marginalization Loop
            for model in teacher_models:
                for sigma in scan_sigmas:
                    # Inject depth (sigma is standard deviations, which matches normalized depth)
                    d = torch.full((batch_size, 1), sigma, device=device)

                    logits = model(images, d)
                    probs = torch.sigmoid(logits)
                    accum_probs += probs

            # Average
            accum_probs /= len(teacher_models) * len(scan_sigmas)

            # Center Crop to 101x101
            h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
            w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
            probs_cropped = accum_probs[
                :,
                0,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]

            np_probs = probs_cropped.cpu().numpy()

            for i, img_id in enumerate(ids):
                results[img_id] = np_probs[i]

    # 3. Sort and Format
    # Load test metadata to ensure correct order
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    sorted_labels = []
    for img_id in test_df["id"]:
        if img_id in results:
            sorted_labels.append(results[img_id])
        else:
            # Fallback
            sorted_labels.append(np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE)))

    final_labels = np.array(sorted_labels, dtype=np.float32)

    # 4. Save to Cache
    np.save(cache_path, final_labels)
    print(f"Saved pseudo labels to {cache_path}")

    return final_labels


def predict_and_submit(model, test_loader, val_loader, device):
    """
    Performs inference on test set, optimizes threshold on val set, and saves submission.

    Args:
        model: Student model.
        test_loader: Test DataLoader.
        val_loader: Validation DataLoader (for threshold optimization).
        device: Device.
    """
    model.eval()
    model.to(device)

    # 1. Optimize Threshold using Validation Set
    print("Optimizing threshold on validation set...")
    _, _, val_preds, val_targets = evaluate(model, val_loader, device, mode="student")
    best_thresh, best_score = get_best_threshold(
        val_targets,
        val_preds,
        start=Config.THRESH_START,
        end=Config.THRESH_END,
        step=Config.THRESH_STEP,
    )
    print(f"Best Threshold: {best_thresh:.3f}, Val mAP: {best_score:.5f}")

    # 2. Inference on Test Set (with TTA)
    print("Predicting test set with TTA...")
    results = {}

    h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            images = batch["image"].to(device)
            ids = batch["id"]

            # TTA: Original
            out_orig = model(images)["mask"]
            prob_orig = torch.sigmoid(out_orig)

            # TTA: Flip
            if Config.TTA_FLIP:
                images_flip = torch.flip(images, dims=[3])
                out_flip = model(images_flip)["mask"]
                prob_flip = torch.sigmoid(out_flip)
                prob_flip = torch.flip(prob_flip, dims=[3])

                # Average
                probs = (prob_orig + prob_flip) / 2.0
            else:
                probs = prob_orig

            # Crop to 101x101
            probs_cropped = probs[
                :,
                0,
                h_start : h_start + Config.ORIG_SIZE,
                w_start : w_start + Config.ORIG_SIZE,
            ]
            np_probs = probs_cropped.cpu().numpy()

            for i, img_id in enumerate(ids):
                results[img_id] = np_probs[i]

    # 3. Generate Submission
    print("Generating submission file...")
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    submission_rows = []

    for _, row in test_df.iterrows():
        img_id = row["id"]
        prob_map = results.get(img_id, np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE)))

        # Binarize
        mask = (prob_map > best_thresh).astype(np.uint8)

        # RLE
        rle = rle_encode(mask)
        submission_rows.append({"id": img_id, "rle_mask": rle})

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
