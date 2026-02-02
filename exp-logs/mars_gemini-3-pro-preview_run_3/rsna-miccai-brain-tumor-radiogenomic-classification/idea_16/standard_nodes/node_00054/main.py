import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Import provided library modules
from library import config
from library import utils
from library.model import SHDVNet
from library.data_loader import get_dataset, BraTSDataset
from library.train import train_one_epoch, validate
from library.predict import generate_submission


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = config.DEVICE

    # 2. Data Loading
    # Load training and validation data (uses caching)
    X_train, y_train = get_dataset(
        config.TRAIN_META_PATH, "train", load_cached_data=True
    )
    X_val, y_val = get_dataset(config.VAL_META_PATH, "val", load_cached_data=True)

    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Model Initialization
    model = SHDVNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_auc = 0.0

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_PATH)

    # 5. Metric Reporting
    print(f"Final Validation Metric: {best_auc}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")

    # Reload best model
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_preds.extend(probs)
            val_targets.extend(targets.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate Error
    errors = np.abs(val_targets - val_preds)

    # Load Metadata for feature correlation
    val_df = pd.read_parquet(config.VAL_META_PATH)

    # Extract meta-features (slice counts)
    analysis_df = pd.DataFrame()
    analysis_df["error"] = errors

    modalities = ["flair", "t1w", "t1wce", "t2w"]
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate number of slices per modality
        analysis_df[f"{mod}_count"] = val_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )

    # Compute correlations
    correlations = analysis_df.corrwith(analysis_df["error"])
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("error").sort_values(ascending=False))

    # 7. Conditional Submission
    threshold = 0.6978181818181817
    if best_auc > threshold:
        print(
            f"\nValidation metric ({best_auc}) > threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            model_path=config.MODEL_PATH,
            metadata_path=config.TEST_META_PATH,
            output_path=config.SUBMISSION_PATH,
            device=device,
        )
    else:
        print(
            f"\nValidation metric ({best_auc}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
