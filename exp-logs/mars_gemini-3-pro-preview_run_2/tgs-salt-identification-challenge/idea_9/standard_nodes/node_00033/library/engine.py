import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import set_seed, do_kaggle_metric, rle_encode
from library.losses import MixedLoss, DistillationLoss
from library.models import SaltLinkNet


def train_epoch(model, loader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        depths = batch["depth"].to(device)

        optimizer.zero_grad()

        # Model requires depth
        logits = model(images, depth=depths)

        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, threshold=0.5):
    """
    Evaluates the model on the validation set.
    Returns average loss (if applicable) and mAP score.
    """
    model.eval()
    total_score = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            logits = model(images, depth=depths)
            probs = torch.sigmoid(logits)

            # Calculate batch metric
            # do_kaggle_metric returns mean score for the batch
            batch_score = do_kaggle_metric(probs, masks, threshold=threshold)

            # Weight by batch size to get true average over dataset
            batch_size = images.size(0)
            total_score += batch_score * batch_size
            count += batch_size

    return total_score / count


def run_training(
    loader_train,
    loader_val,
    device,
    epochs=40,
    lr=1e-4,
    patience=10,
    save_path="./working/best_model.pth",
):
    """
    Orchestrates Training.
    """
    print("Starting Training...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = SaltLinkNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Cite solution_lesson_node_00021: Lovasz-Hinge Loss for IoU optimization
    loss_fn = MixedLoss(bce_weight=1.0, lovasz_weight=1.0)

    best_score = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, loader_train, optimizer, loss_fn, device)
        val_score = validate(model, loader_val, device, threshold=0.5)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val mAP: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training finished. Best mAP: {best_score:.6f}")
    return save_path


def optimize_threshold(model, loader, device):
    """
    Finds the optimal threshold on the validation set.
    """
    model.eval()
    all_probs = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            logits = model(images, depth=depths)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    best_thresh = 0.5
    best_score = -1.0

    # Sweep thresholds
    for t in np.arange(0.3, 0.75, 0.05):
        score = do_kaggle_metric(all_probs, all_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Optimal Threshold: {best_thresh:.2f} with Val mAP: {best_score:.6f}")
    return best_thresh


def generate_submission(
    model_path,
    loader_test,
    loader_val,
    device,
    output_path="./submission/submission.csv",
):
    """
    Generates the submission file using the Trained model.
    Includes Threshold Optimization and Test-Time Augmentation (TTA).
    """
    print("Generating submission...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Model
    model = SaltLinkNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Optimize Threshold on Validation Set
    # Cite solution_lesson_node_00018: Post-Training Threshold Optimization
    best_threshold = optimize_threshold(model, loader_val, device)

    # 2. Predict on Test Set with TTA
    ids = []
    rle_masks = []

    with torch.no_grad():
        for batch in loader_test:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            batch_ids = batch["id"]

            # TTA: Original
            logits_orig = model(images, depth=depths)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA: Horizontal Flip
            # Cite solution_lesson_node_00002: Test-Time Augmentation
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, depth=depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Binarize
            preds = (probs_avg > best_threshold).float()
            preds = preds.cpu().numpy()  # (B, 1, H, W)

            for i in range(len(batch_ids)):
                mask = preds[i, 0]
                rle = rle_encode(mask)
                ids.append(batch_ids[i])
                rle_masks.append(rle)

    # 3. Save to CSV
    df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
