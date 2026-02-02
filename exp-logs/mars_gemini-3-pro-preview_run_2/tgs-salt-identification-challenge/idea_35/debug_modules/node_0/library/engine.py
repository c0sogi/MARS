import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import do_kaggle_metric, unpad_image_128, rle_encode


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    criterion,
    is_student=False,
    is_pseudo=False,
):
    """
    Executes one epoch of training.
    Handles both Teacher (Depth-Injected) and Student (Multi-Task) modes.

    Args:
        model: The PyTorch model.
        loader: DataLoader.
        optimizer: Optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number.
        criterion: Loss function.
                   - For Student: Expects StudentLoss signature.
                   - For Teacher: Expects (logits, targets) signature.
        is_student: Boolean, True if training the Student model.
        is_pseudo: Boolean, True if using pseudo-labels (Distillation).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, data in enumerate(loader):
        images = data["image"].to(device)

        # Load Masks (Ground Truth or Pseudo Labels)
        masks = None
        if "mask" in data:
            masks = data["mask"].to(device)

        # Load Depths
        depths = None
        if "depth" in data:
            depths = data["depth"].to(device)

        optimizer.zero_grad()

        loss = 0.0

        if is_student:
            # Student Mode: Forward(image) -> logits, aux
            if Config.STUDENT_AUX_HEAD:
                mask_logits, aux_depth_pred = model(images)
                # StudentLoss requires specific unpacking
                loss = criterion(
                    mask_logits,
                    aux_depth_pred,
                    masks,
                    depth_targets=depths,
                    is_pseudo=is_pseudo,
                )
            else:
                # Fallback if aux head is disabled
                mask_logits = model(images)
                loss = criterion(mask_logits, masks)
        else:
            # Teacher Mode: Forward(image, depth) -> logits
            # Teacher requires depth injection
            mask_logits = model(images, depth=depths)

            # Teacher Loss: Sum of Lovasz and BCE (Handled by passed criterion)
            loss = criterion(mask_logits, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Train Loss: {epoch_loss}")
    return epoch_loss


def validate(model, loader, device, is_student=False):
    """
    Evaluates the model on the validation set using the competition metric (mAP).
    Unpads images to original size (101x101) for accurate metric calculation.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            masks = data["mask"].to(device)

            logits = None

            if is_student:
                if Config.STUDENT_AUX_HEAD:
                    logits, _ = model(images)
                else:
                    logits = model(images)
            else:
                depths = data["depth"].to(device)
                logits = model(images, depth=depths)

            probs = torch.sigmoid(logits)

            # Unpad to original size (101x101) to match ground truth strictly
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(probs_np.shape[0]):
                # Transpose to HWC for unpadding
                p = probs_np[i].transpose(1, 2, 0)
                m = masks_np[i].transpose(1, 2, 0)

                p_orig = unpad_image_128(p)
                m_orig = unpad_image_128(m)

                # Expand dims to (1, H, W, C) for metric function which expects batch
                score = do_kaggle_metric(p_orig[None, ...], m_orig[None, ...])
                scores.append(score)

    val_score = np.mean(scores)
    print(f"Validation mAP: {val_score}")
    return val_score


def optimize_threshold(model, loader, device):
    """
    Performs a linear search on the validation set to find the optimal
    binarization threshold that maximizes mAP.
    """
    model.eval()
    all_probs = []
    all_masks = []

    print("Optimizing threshold...")

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            masks = data["mask"].to(device)

            # Determine model mode based on attributes
            if hasattr(model, "use_depth") and model.use_depth:
                depths = data["depth"].to(device)
                logits = model(images, depth=depths)
            elif hasattr(model, "aux_head") and model.aux_head:
                logits, _ = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits)

            # Unpad and collect
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(probs_np.shape[0]):
                p = probs_np[i].transpose(1, 2, 0)
                m = masks_np[i].transpose(1, 2, 0)
                all_probs.append(unpad_image_128(p))
                all_masks.append(unpad_image_128(m))

    thresholds = np.arange(0.3, 0.8, 0.05)
    best_score = -1
    best_thresh = 0.5

    for t in thresholds:
        current_scores = []
        for p, m in zip(all_probs, all_masks):
            # Pass single item as batch of 1
            s = do_kaggle_metric(p[None, ...], m[None, ...], threshold=t)
            current_scores.append(s)

        avg_score = np.mean(current_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_thresh = t

    print(f"Best Threshold: {best_thresh} with mAP: {best_score}")
    return best_thresh


def generate_submission(model, loader, device, threshold=0.5):
    """
    Generates predictions for the test set and saves to submission.csv.
    Applies TTA (Horizontal Flip) and unpadding.
    """
    model.eval()
    submission_data = []

    print(f"Generating submission with threshold {threshold}...")

    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            ids = data["id"]

            # TTA: Original
            if hasattr(model, "aux_head") and model.aux_head:
                logits, _ = model(images)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits)

            # TTA: Flip
            images_flip = torch.flip(images, dims=[3])
            if hasattr(model, "aux_head") and model.aux_head:
                logits_flip, _ = model(images_flip)
            else:
                logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)
            # Flip back
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            # Process batch
            avg_probs_np = avg_probs.cpu().numpy()

            for i in range(len(ids)):
                img_id = ids[i]
                prob_img = avg_probs_np[i].transpose(1, 2, 0)  # HWC

                # Unpad to 101x101
                prob_orig = unpad_image_128(prob_img)

                # Binarize
                mask = (prob_orig > threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask)

                submission_data.append([img_id, rle])

    # Save to CSV
    df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
