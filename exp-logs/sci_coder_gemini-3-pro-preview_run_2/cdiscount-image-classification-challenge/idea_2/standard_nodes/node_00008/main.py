import os
import sys
import torch
import numpy as np
import pandas as pd
import random
import time

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model
import library.engine as engine


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Configuration Setup for Fast Baseline
    # We override config parameters to ensure the code runs within the time limit (2 hours)
    # and serves as a fast baseline.
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 100000  # Process 100k samples per split for speed
    config.EPOCHS = 1  # Train for only 1 epoch

    # Ensure device is set correctly
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting Fast Baseline Run...")
    print(
        f"Configuration: DEBUG={config.DEBUG}, SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}, EPOCHS={config.EPOCHS}, DEVICE={config.DEVICE}"
    )

    set_seed(config.SEED)

    # 2. Data Loading
    # Load cached metadata and create dataloaders
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    net = model.get_model()

    # 4. Optimizer and Scheduler Setup
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    # OneCycleLR is effective for short training durations
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        pct_start=0.3,
    )

    # 5. Training
    print("\n==== Starting Training ====")
    trainer = engine.Trainer(net, optimizer, scheduler, config.DEVICE)

    # fit() returns the best validation accuracy
    best_acc = trainer.fit(train_loader, val_loader, epochs=config.EPOCHS)

    # Required Output
    print(f"Final Validation Metric: {best_acc}")

    # 6. Failure Analysis
    print("\n==== Performing Failure Analysis ====")
    net.eval()

    losses = []
    bson_lengths = []
    num_images_list = []
    pixel_intensities = []

    # We iterate over the validation set to collect error metrics and feature stats
    # We use a separate loop because we need per-sample loss and metadata
    print("Collecting validation error statistics...")

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # images shape from loader: (1, N, C, H, W)
            # target shape: (1)

            # Retrieve metadata
            # Since shuffle=False for val_loader, index i matches dataset index
            # Note: When DEBUG=True, the dataset is a subsampled DataFrame.
            # The dataset.__getitem__ handles the mapping, but here we need the raw metadata row.
            # val_loader.dataset.metadata is the DataFrame used by the dataset.
            meta_row = val_loader.dataset.metadata.iloc[i]
            b_len = meta_row["bson_length"]

            # Prepare data
            images = images.squeeze(0).to(config.DEVICE)  # (N, C, H, W)
            target = target.to(config.DEVICE)

            # Forward pass
            output = net(images)

            # Calculate Loss (Error Magnitude)
            # We use the negative log probability of the true class as the error magnitude
            # Late fusion: average probabilities
            probs = torch.softmax(output, dim=1)
            avg_prob = torch.mean(probs, dim=0, keepdim=True)  # (1, num_classes)

            target_idx = target.item()
            target_prob = avg_prob[0, target_idx].item()

            # Error magnitude: -log(p_target)
            # Clamp to avoid log(0)
            error_mag = -np.log(max(target_prob, 1e-9))

            # Collect stats
            losses.append(error_mag)
            bson_lengths.append(b_len)
            num_images_list.append(images.shape[0])
            pixel_intensities.append(images.mean().item())

            if i % 5000 == 0 and i > 0:
                print(f"Analyzed {i}/{len(val_loader)} samples...")

    # Compute Correlations
    if len(losses) > 1:
        corr_bson = np.corrcoef(losses, bson_lengths)[0, 1]
        corr_nimg = np.corrcoef(losses, num_images_list)[0, 1]
        corr_pixel = np.corrcoef(losses, pixel_intensities)[0, 1]

        print(f"\nCorrelation Analysis (Error Magnitude vs Input Features):")
        print(f"Correlation with BSON Record Length: {corr_bson:.6f}")
        print(f"Correlation with Number of Images:   {corr_nimg:.6f}")
        print(f"Correlation with Mean Pixel Intensity: {corr_pixel:.6f}")

        # Interpretation hint
        print(
            "(Positive correlation implies the feature is associated with higher error/lower accuracy)"
        )
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.6116

    if best_acc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation Metric ({best_acc}) > Threshold ({SUBMISSION_THRESHOLD}). Generating Submission..."
        )
        engine.inference(net, test_loader, config.DEVICE)
    else:
        print(
            f"\nValidation Metric ({best_acc}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
