import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.losses import SegmentationLoss
from library.utils import unpad_image, calc_map_score, rle_encode, optimize_threshold


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    criterion = SegmentationLoss()

    total_loss_meter = 0.0
    bce_loss_meter = 0.0
    lovasz_loss_meter = 0.0
    count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        depths = batch["depth"].to(device)

        optimizer.zero_grad()

        # Pass both image and depth to model
        preds = model(images, depths)

        loss, loss_dict = criterion(preds, masks)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss_meter += loss.item() * batch_size
        bce_loss_meter += loss_dict["bce"] * batch_size
        lovasz_loss_meter += loss_dict["lovasz"] * batch_size
        count += batch_size

    metrics = {
        "loss": total_loss_meter / count,
        "bce": bce_loss_meter / count,
        "lovasz": lovasz_loss_meter / count,
    }

    print(
        f"Epoch {epoch} Train Loss: {metrics['loss']:.10f} (BCE: {metrics['bce']:.6f}, Lovasz: {metrics['lovasz']:.6f})"
    )
    return metrics["loss"]


def validate(model, dataloader, device, return_probs=False):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    criterion = SegmentationLoss()

    total_loss_meter = 0.0
    map_score_meter = 0.0
    count = 0

    all_probs = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            # Pass both image and depth
            preds = model(images, depths)

            loss, _ = criterion(preds, masks)

            batch_size = images.size(0)
            total_loss_meter += loss.item() * batch_size

            # Get Probabilities
            pred_probs = torch.sigmoid(preds)

            # Move to CPU/Numpy for metric calculation
            pred_probs_np = pred_probs.cpu().numpy()
            gt_masks_np = masks.cpu().numpy()

            for i in range(batch_size):
                p = pred_probs_np[i].squeeze()
                g = gt_masks_np[i].squeeze()

                # Unpad to original dimensions (101x101)
                p_unpadded = unpad_image(
                    p, orig_shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                )
                g_unpadded = unpad_image(
                    g, orig_shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                )

                if return_probs:
                    all_probs.append(p_unpadded)
                    all_masks.append(g_unpadded)

                # Calculate mAP at standard threshold 0.5 for monitoring
                p_bin = (p_unpadded > 0.5).astype(np.uint8)
                g_bin = (g_unpadded > 0.5).astype(np.uint8)
                map_score_meter += calc_map_score(p_bin, g_bin)

            count += batch_size

    avg_loss = total_loss_meter / count
    avg_map = map_score_meter / count

    print(f"Val Loss: {avg_loss:.10f}, Val mAP: {avg_map:.10f}")

    if return_probs:
        return avg_loss, avg_map, all_probs, all_masks
    return avg_loss, avg_map


def train_model(
    model, train_loader, val_loader, optimizer, device, epochs, patience, save_path
):
    """
    Orchestrates the training loop with early stopping and threshold optimization.
    """
    best_map = -1.0
    patience_counter = 0
    best_threshold = 0.5

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_map = validate(model, val_loader, device)

        # Save best model based on mAP
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with mAP: {best_map:.10f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model to optimize threshold
    print("Loading best model for threshold optimization...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Get predictions on validation set
    _, _, probs, gt_masks = validate(model, val_loader, device, return_probs=True)

    # Find optimal threshold
    best_threshold, best_score = optimize_threshold(probs, gt_masks)

    return best_threshold


def predict_and_submit(model, test_loader, device, threshold, submission_path):
    """
    Generates predictions for the test set using TTA and saves to CSV.
    """
    model.eval()
    ids_list = []
    rles_list = []

    print(f"Generating predictions with threshold {threshold:.4f}...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ids = batch["id"]

            # Need depth for inference too
            depths = batch["depth"].to(device)

            # TTA: 1. Predict Original
            preds_orig = model(images, depths)
            probs_orig = torch.sigmoid(preds_orig)

            # TTA: 2. Predict Flipped
            images_flipped = torch.flip(images, dims=[3])  # Flip Width
            preds_flip = model(images_flipped, depths)
            probs_flip = torch.sigmoid(preds_flip)
            probs_flip_back = torch.flip(probs_flip, dims=[3])

            # Average
            probs_avg = (probs_orig + probs_flip_back) / 2.0

            probs_np = probs_avg.cpu().numpy()

            for i in range(len(ids)):
                img_id = ids[i]
                p = probs_np[i].squeeze()

                # Unpad
                p_unpadded = unpad_image(
                    p, orig_shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                )

                # Binarize
                mask_bin = (p_unpadded > threshold).astype(np.uint8)

                # Encode
                rle = rle_encode(mask_bin)

                ids_list.append(img_id)
                rles_list.append(rle)

    # Save Submission
    df = pd.DataFrame({"id": ids_list, "rle_mask": rles_list})
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
