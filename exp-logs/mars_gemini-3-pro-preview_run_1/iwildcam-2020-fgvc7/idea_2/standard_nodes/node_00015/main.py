import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import dataset
from library import model
from library import engine
from library import loss


def main():
    # 1. Setup & Reproducibility
    utils.seed_everything(config.SEED)

    print("Initializing datasets...")

    # 2. Data Loading
    # Initialize datasets
    train_dataset = dataset.CameraTrapDataset(
        "train", transform=dataset.get_transforms("train"), load_cached_data=True
    )
    val_dataset = dataset.CameraTrapDataset(
        "val", transform=dataset.get_transforms("val"), load_cached_data=True
    )
    test_dataset = dataset.CameraTrapDataset(
        "test", transform=dataset.get_transforms("val"), load_cached_data=True
    )

    # Create DataLoaders
    # We create them manually here to use the modified datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing {config.MODEL_NAME}...")
    net = model.get_model(device=config.DEVICE, pretrained=True)

    # Loss and Optimizer
    criterion = loss.FocalLoss(
        alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA
    )

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # 4. Training
    print(f"Starting training for {config.EPOCHS} epochs...")

    engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=config.EPOCHS,
        patience=config.EPOCHS,
        scheduler=scheduler,
    )

    # 5. Validation & Failure Analysis
    print("Loading best model for validation and analysis...")
    # Load the best model saved during training
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    print("Running inference on full validation set...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # TTA: Average predictions from original and flipped images
            outputs_orig = net(images)
            outputs_flip = net(torch.flip(images, dims=[3]))
            outputs = (outputs_orig + outputs_flip) / 2.0

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("Performing failure analysis...")
    # Calculate error vector (1 for error, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    # Extract MegaDetector confidence from validation metadata
    # Ensure alignment: val_loader was shuffle=False, so order matches val_dataset.data
    # Fill NaNs in 'conf' with 0.0 (implies no detection/empty)
    md_conf = val_dataset.data["conf"].fillna(0.0).values

    # Ensure lengths match (just in case of drop_last or other issues, though val shouldn't drop)
    min_len = min(len(errors), len(md_conf))
    errors = errors[:min_len]
    md_conf = md_conf[:min_len]

    # Calculate correlation
    if len(errors) > 0 and np.std(errors) > 0 and np.std(md_conf) > 0:
        correlation = np.corrcoef(errors, md_conf)[0, 1]
        print(
            f"Correlation between Error and MegaDetector Confidence: {correlation:.4f}"
        )
    else:
        print(
            "Correlation could not be calculated (constant values in error or confidence)."
        )

    # 6. Submission
    THRESHOLD = 0.9667610292948249

    if accuracy > THRESHOLD:
        print(f"Validation metric {accuracy} > {THRESHOLD}. Generating submission...")
        engine.generate_submission(
            model=net,
            test_loader=test_loader,
            device=config.DEVICE,
            output_path=config.SUBMISSION_FILE,
        )
    else:
        print(
            f"Validation metric {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
