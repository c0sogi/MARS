import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided libraries
from library.dataset import get_loaders, SaltDataset, get_transforms
from library.model import ResNet34WideLinkNet
from library.engine import (
    train_one_epoch,
    evaluate,
    optimize_threshold,
    predict_test,
    set_seed,
)
from library.utils import do_kaggle_metric, unpad_image


def failure_analysis(model, loader, device):
    """
    Analyzes the correlation between error (1 - IoU) and depth.
    """
    model.eval()
    ious = []
    depths_list = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths_gpu = depths.to(device)

            logits = model(images, depths_gpu)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            depths_np = depths.numpy()  # Original depths (normalized)

            for i in range(len(probs_np)):
                # Calculate IoU for this single image
                p = (probs_np[i, 0] > 0.5).astype(np.uint8)
                t = (masks_np[i, 0] > 0.5).astype(np.uint8)

                intersection = np.sum((p * t) > 0)
                union = np.sum((p + t) > 0)
                if union == 0:
                    iou = 1.0 if np.sum(t) == 0 else 0.0
                else:
                    iou = intersection / union

                ious.append(iou)
                depths_list.append(depths_np[i].item())

    ious = np.array(ious)
    depths_arr = np.array(depths_list)
    errors = 1.0 - ious

    correlation = np.corrcoef(errors, depths_arr)[0, 1]
    print(
        f"Failure Analysis - Correlation between Error (1-IoU) and Depth: {correlation:.4f}"
    )


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration for fast baseline
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    STAGE1_EPOCHS = 15
    STAGE2_EPOCHS = 15
    MAX_BATCHES = None  # Set to None for full training, or integer for debugging

    # 2. Data Loading
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )

    # Extract depth stats from the dataset object to reuse later
    depth_stats = (train_loader.dataset.depth_mean, train_loader.dataset.depth_std)

    # 3. Stage 1: Robust Supervised Training
    print("\n--- Stage 1: Robust Supervised Training ---")
    model_s1 = ResNet34WideLinkNet(pretrained=True).to(device)
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=1e-4, weight_decay=1e-2)

    best_s1_score = -1.0
    best_s1_path = "./working/stage1_model.pth"

    for epoch in range(1, STAGE1_EPOCHS + 1):
        loss = train_one_epoch(
            model_s1, train_loader, optimizer_s1, device, epoch, max_batches=MAX_BATCHES
        )
        score = evaluate(
            model_s1, val_loader, device, threshold=0.5, max_batches=MAX_BATCHES
        )

        if score > best_s1_score:
            best_s1_score = score
            torch.save(model_s1.state_dict(), best_s1_path)
            # print(f"Stage 1 Best Model Saved: {best_s1_score:.4f}")

    print(f"Stage 1 Complete. Best mAP: {best_s1_score:.4f}")

    # 4. Pseudo-Labeling
    print("\n--- Generating Pseudo-Labels ---")
    # Load best stage 1 model
    model_s1.load_state_dict(torch.load(best_s1_path))
    model_s1.eval()

    pseudo_masks = []

    with torch.no_grad():
        for images, _, _, _ in test_loader:
            images = images.to(device)
            # Force depth to 0 (mean) for test set prediction
            depths = torch.zeros(
                (images.size(0), 1), device=device, dtype=torch.float32
            )

            # TTA
            logits = model_s1(images, depths)
            probs = torch.sigmoid(logits)

            images_flip = torch.flip(images, dims=[3])
            logits_flip = model_s1(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            avg_probs = (probs + probs_flip) / 2.0

            # Generate binary masks (Hard Pseudo-Labels)
            preds = (avg_probs > 0.5).float().cpu().numpy()
            # Remove channel dim: (B, 1, H, W) -> (B, H, W)
            preds = preds[:, 0, :, :]
            pseudo_masks.append(preds)

    pseudo_masks = np.concatenate(pseudo_masks, axis=0).astype(np.uint8)

    # 5. Stage 2: Self-Training (Combined Dataset)
    print("\n--- Stage 2: Self-Training on Combined Data ---")

    # Extract arrays from original loaders
    train_imgs = train_loader.dataset.images
    train_masks = train_loader.dataset.masks
    train_depths = train_loader.dataset.depths
    train_ids = train_loader.dataset.ids

    test_imgs = test_loader.dataset.images
    test_ids = test_loader.dataset.ids
    # For pseudo-labeled data, we use fixed depth 0 (which is the mean)
    # Since dataset normalizes by (x - mean)/std, passing mean results in 0.
    test_depths_fixed = np.full(len(test_imgs), depth_stats[0], dtype=np.float32)

    # Combine
    combined_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
    # train_masks is (N, 128, 128), pseudo_masks is (M, 128, 128)
    combined_masks = np.concatenate([train_masks, pseudo_masks], axis=0)
    combined_depths = np.concatenate([train_depths, test_depths_fixed], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)

    # Create Combined Dataset and Loader
    combined_ds = SaltDataset(
        combined_imgs,
        combined_masks,
        combined_depths,
        combined_ids,
        depth_stats,
        mode="train",  # Enable augmentations and Bernoulli masking
        transform=get_transforms("train"),
    )

    combined_loader = DataLoader(
        combined_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Train Stage 2 Model from scratch
    model_s2 = ResNet34WideLinkNet(pretrained=True).to(device)
    optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=1e-4, weight_decay=1e-2)

    best_s2_score = -1.0
    best_s2_path = "./working/best_model.pth"

    for epoch in range(1, STAGE2_EPOCHS + 1):
        loss = train_one_epoch(
            model_s2,
            combined_loader,
            optimizer_s2,
            device,
            epoch,
            max_batches=MAX_BATCHES,
        )
        # Validate on original validation set
        score = evaluate(
            model_s2, val_loader, device, threshold=0.5, max_batches=MAX_BATCHES
        )

        if score > best_s2_score:
            best_s2_score = score
            torch.save(model_s2.state_dict(), best_s2_path)

    print(
        f"Stage 2 Complete. Best Validation mAP (default thresh): {best_s2_score:.4f}"
    )

    # 6. Final Evaluation & Optimization
    print("\n--- Final Evaluation & Optimization ---")
    # Load best model
    model_s2.load_state_dict(torch.load(best_s2_path))

    # Optimize Threshold
    optimal_threshold = optimize_threshold(
        model_s2, val_loader, device, max_batches=MAX_BATCHES
    )

    # Calculate Final Metric
    final_score = evaluate(
        model_s2,
        val_loader,
        device,
        threshold=optimal_threshold,
        max_batches=MAX_BATCHES,
    )
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    failure_analysis(model_s2, val_loader, device)

    # 7. Submission
    if final_score > 0.7985:
        print("\n--- Generating Submission ---")
        predict_test(
            model_s2,
            test_loader,
            device,
            threshold=optimal_threshold,
            output_path="./submission/submission.csv",
        )
    else:
        print(
            f"\nValidation score {final_score} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
