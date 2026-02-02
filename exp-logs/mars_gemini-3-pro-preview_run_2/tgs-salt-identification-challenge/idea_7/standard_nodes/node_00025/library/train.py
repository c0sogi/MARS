import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

from library import config, dataset, losses, utils, model


def run_training(
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    debug=False,
    device=config.DEVICE,
):
    """
    Executes the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        debug (bool): If True, runs on a small subset of data for debugging.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        tuple: (depth_mean, depth_std) statistics from the training set.
    """
    # Set reproducible seed
    utils.set_seed(config.SEED)

    # Patch config batch size temporarily as dataset library uses it
    original_bs = config.BATCH_SIZE
    config.BATCH_SIZE = batch_size

    # Get DataLoaders (and calculate depth stats)
    train_loader, val_loader, depth_mean, depth_std = dataset.get_train_val_loaders(
        load_cached_data=True
    )

    # Handle Debug Mode
    if debug:
        print("Debug mode enabled: Using subset of data.")
        limit = 64  # Small subset for quick debugging
        train_subset = Subset(
            train_loader.dataset, range(min(len(train_loader.dataset), limit))
        )
        val_subset = Subset(
            val_loader.dataset, range(min(len(val_loader.dataset), limit))
        )

        # Re-create loaders for subsets
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        epochs = 2  # Force minimal epochs for debug

    # Initialize Model
    net = model.WideLinkNet34().to(device)

    # Loss and Optimizer
    criterion = losses.SaltNetLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Training Loop Variables
    best_map = 0.0
    patience = 10
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train Step
        train_loss = model.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validation Step
        val_loss, val_map, _, _ = model.validate(net, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full precision for mAP as requested)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mAP: {val_map} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint & Early Stopping
        if val_map > best_map:
            best_map = val_map
            torch.save(net.state_dict(), config.CHECKPOINT_PATH)
            print(f"  >>> Model Saved! New Best mAP: {best_map}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Restore original config
    config.BATCH_SIZE = original_bs

    return depth_mean, depth_std


def run_inference(depth_mean, depth_std, device=config.DEVICE):
    """
    Generates submission file using the best trained model.
    """
    print("Starting inference...")

    # Load Model
    net = model.WideLinkNet34().to(device)
    if os.path.exists(config.CHECKPOINT_PATH):
        net.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model from {config.CHECKPOINT_PATH}")
    else:
        print("Warning: No checkpoint found. Inference will use random weights.")

    net.eval()

    # Optimize Threshold on Validation Set
    # We reload validation data to find the best threshold for the trained model
    _, val_loader, _, _ = dataset.get_train_val_loaders(load_cached_data=True)
    criterion = losses.SaltNetLoss()

    print("Optimizing threshold on validation set...")
    _, _, val_preds, val_masks = model.validate(net, val_loader, criterion, device)
    best_thresh, best_score = model.find_best_threshold(val_preds, val_masks)
    print(f"Optimized Threshold: {best_thresh} (Val mAP: {best_score})")

    # Generate Predictions on Test Set
    test_loader = dataset.get_test_loader(depth_mean, depth_std, load_cached_data=True)
    submission_data = []

    print("Predicting on test set...")
    with torch.no_grad():
        for images, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Forward Pass
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
            # Model output is 128x128
            h, w = avg_probs.shape[2], avg_probs.shape[3]
            target_size = config.IMG_ORIG_SIZE
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            end_h = start_h + target_size
            end_w = start_w + target_size

            avg_probs = avg_probs[:, :, start_h:end_h, start_w:end_w]

            # Binarize
            preds_bin = (avg_probs > best_thresh).cpu().numpy().astype(np.uint8)

            # Encode to RLE
            for i in range(len(ids)):
                # preds_bin shape is (B, 1, H, W), take first channel
                mask = preds_bin[i, 0]
                rle = utils.rle_encode(mask)
                submission_data.append([ids[i], rle])

    # Save Submission
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
