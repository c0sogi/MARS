import os
import random
import numpy as np
import torch
from library.config import Config
from library.utils import do_kaggle_metric, rle_encode, crop_image


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Trains the model for one epoch using FP32 (Cite {solution_lesson_node_00097}).
    """
    model.train()
    running_loss = 0.0

    for i, (images, masks, depths, ids) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        depths = depths.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Teacher: Input (Image, Depth) -> Output (Seg)
        logits = model(images, depths)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, device, loss_fn):
    """
    Validates the model on the validation set using FP32.
    Computes Loss and Mean Average Precision (mAP).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, ids in dataloader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)

            logits = model(images, depths)
            loss = loss_fn(logits, masks)
            preds = torch.sigmoid(logits)

            running_loss += loss.item()

            # Store predictions for metric calculation
            preds_np = preds.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            all_preds.append(preds_np)
            all_masks.append(masks_np)

    epoch_loss = running_loss / len(dataloader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Remove channel dimension if present (N, 1, H, W) -> (N, H, W)
    if all_preds.ndim == 4:
        all_preds = all_preds.squeeze(1)
    if all_masks.ndim == 4:
        all_masks = all_masks.squeeze(1)

    # Calculate mAP using default threshold of 0.5 for reporting
    metric_score = do_kaggle_metric(all_preds, all_masks, threshold=0.5)

    return epoch_loss, metric_score


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    epochs,
    patience=15,
    save_path=None,
):
    """
    Orchestrates the training process with Early Stopping and Scheduler.
    """
    best_metric = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_metric = validate(model, val_loader, device, loss_fn)

        if scheduler:
            scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val mAP: {val_metric:.16f}"
        )

        # Early Stopping check
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_metric


def predict(model, dataloader, device):
    """
    Generates predictions using the model with TTA.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, depths, ids in dataloader:
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)

            # Standard Forward
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            if Config.TTA_ENABLED:
                # Horizontal Flip TTA (Cite {solution_lesson_node_00095})
                images_flip = torch.flip(images, [3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, [3])

                # Average original and flipped predictions
                probs = (probs + probs_flip) / 2.0

            probs = probs.detach().cpu().numpy()

            for idx, img_id in enumerate(ids):
                results[img_id] = probs[idx]

    return results


def generate_submission(model, test_loader, device, threshold=0.5):
    """
    Generates the final submission CSV.
    """
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Run Inference
    preds_dict = predict(model, test_loader, device)

    submission_lines = ["id,rle_mask"]
    sorted_ids = sorted(preds_dict.keys())

    for img_id in sorted_ids:
        prob_map = preds_dict[img_id]  # Shape: (1, 128, 128)

        # Squeeze channel dim
        if prob_map.ndim == 3:
            prob_map = prob_map[0]

        # Binarize
        mask = (prob_map > threshold).astype(np.uint8)

        # Crop back to original 101x101
        mask = crop_image(mask, original_size=Config.ORIG_SIZE)

        # Encode
        rle = rle_encode(mask)
        submission_lines.append(f"{img_id},{rle}")

    with open(Config.SUBMISSION_PATH, "w") as f:
        f.write("\n".join(submission_lines))

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
