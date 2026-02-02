import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_label_encoder
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet
from library.train import train_one_epoch, validate
from library.inference import generate_submission


def perform_failure_analysis(model, val_loader, val_df, train_df, device):
    """
    Analyzes model failure modes by correlating error with class frequency.
    """
    print("Performing Failure Analysis...")
    model.eval()

    all_targets = []
    all_top5_preds = []

    # 1. Get Predictions on Validation Set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            _, top5_indices = torch.topk(outputs, k=5, dim=1)

            all_top5_preds.append(top5_indices.cpu().numpy())
            all_targets.append(labels.numpy())

    all_top5_preds = np.vstack(all_top5_preds)
    all_targets = np.concatenate(all_targets)

    # 2. Calculate Error (1 if target NOT in top 5, else 0)
    errors = []
    for i, target in enumerate(all_targets):
        if target in all_top5_preds[i]:
            errors.append(0)  # Success
        else:
            errors.append(1)  # Failure

    errors = np.array(errors)

    # 3. Get Class Frequencies
    # Map validation samples to their hotel_id to lookup frequency in training set
    # Note: val_loader is not shuffled, so it aligns with val_df
    val_hotel_ids = val_df["hotel_id"].values

    # Calculate frequency of each hotel in the full training set
    train_counts = train_df["hotel_id"].value_counts().to_dict()

    # Map frequencies to the validation samples
    sample_freqs = np.array([train_counts.get(hid, 0) for hid in val_hotel_ids])

    # 4. Calculate Correlation
    if len(errors) > 0 and np.std(errors) > 0 and np.std(sample_freqs) > 0:
        correlation = np.corrcoef(errors, sample_freqs)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error and Class Frequency: {correlation:.10f}")


def main():
    # ---------------------------------------------------------
    # 1. Configuration and Setup
    # ---------------------------------------------------------
    # Override Config for optimized run
    Config.EPOCHS = 12
    Config.BATCH_SIZE = 128  # Adjusted for ResNet50 memory usage
    Config.DEBUG = False

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Execution Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Data...")
    # Load Encoder
    encoder = get_label_encoder(Config.TRAIN_CSV, load_cached_data=True)
    num_classes = len(encoder.id_to_class)

    # Load Datasets
    full_train_dataset = HotelDataset(
        csv_path=Config.TRAIN_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=encoder,
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_dataset = HotelDataset(
        csv_path=Config.VAL_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=encoder,
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    # Use full training dataset
    train_dataset = full_train_dataset

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = HotelResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("Starting Training...")
    best_map5 = 0.0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_loss, val_map5 = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MAP@5: {val_map5:.4f}"
        )

        # Save Best Model
        if val_map5 >= best_map5:
            best_map5 = val_map5
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    # ---------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # ---------------------------------------------------------
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    # Calculate Final Metric on Full Validation Set
    _, final_map5 = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_map5}")

    # Failure Analysis
    # Load raw dataframes to get metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    perform_failure_analysis(model, val_loader, val_df, train_df, device)

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    # Only generate submission if we improved over the baseline
    if final_map5 > 0.14571255006929015:
        print("Generating Submission...")
        generate_submission(
            checkpoint_path=Config.MODEL_CHECKPOINT,
            output_file=Config.SUBMISSION_FILE,
            batch_size=Config.BATCH_SIZE,
            debug=False,  # Must be False to predict on all test images
            load_cached_data=True,
        )
    else:
        print(
            f"Validation MAP@5 ({final_map5}) did not beat baseline. Skipping submission."
        )

    print("Process Complete.")


if __name__ == "__main__":
    main()
