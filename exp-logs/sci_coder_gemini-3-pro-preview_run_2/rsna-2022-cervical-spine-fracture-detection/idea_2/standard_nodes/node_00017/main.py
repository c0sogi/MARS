import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import warnings

# Import library modules
from library import config
from library import utils
from library import data
from library import model as model_lib
from library import train as train_lib

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    device = config.DEVICE

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load data using cached paths if available.
    # debug=False ensures we use the full training set (161 samples), which is small enough for fast training.
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True, debug=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = model_lib.CervicalSpineSeqModel(pretrained=True)
    model.to(device)

    # ==========================================
    # 4. Optimization Setup
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Use the provided weighted loss (defaults to 1.0 for all classes)
    criterion = utils.WeightedMultiLabelLoss()
    criterion.to(device)

    scaler = GradScaler()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_val_loss = float("inf")

    # Run for the number of epochs specified in config (10)
    for epoch in range(config.EPOCHS):
        # Train one epoch
        train_loss = train_lib.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            config.ACCUMULATION_STEPS,
        )

        # Validate
        val_loss = train_lib.validate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    # ==========================================
    # 6. Final Validation Assessment
    # ==========================================
    # Load best model weights
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect predictions and targets for failure analysis
    val_probs_list = []
    val_targets_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Inference with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            val_probs_list.append(probs.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    val_probs = np.concatenate(val_probs_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Print the required validation metric
    print(f"Final Validation Metric: {best_val_loss}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    # Calculate Log Loss per sample to identify systematic errors
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)

    # Compute binary cross entropy per sample (averaged over the 8 classes)
    bce_per_sample = -(
        val_targets * np.log(val_probs_clipped)
        + (1 - val_targets) * np.log(1 - val_probs_clipped)
    )
    mean_loss_per_sample = np.mean(bce_per_sample, axis=1)

    # Retrieve metadata for correlation analysis
    val_metadata = val_loader.dataset.metadata
    path_map = val_loader.dataset.path_map

    # Feature 1: Slice Count (Z-depth of the scan)
    slice_counts = [
        len(path_map.get(uid, [])) for uid in val_metadata["StudyInstanceUID"]
    ]

    # Feature 2: Patient Overall Label (Fracture vs No Fracture)
    patient_overall = val_metadata["patient_overall"].values

    # Calculate Correlations using Numpy
    # 1. Error vs Slice Count
    if len(set(slice_counts)) > 1:
        corr_slices = np.corrcoef(mean_loss_per_sample, slice_counts)[0, 1]
        print(f"Correlation (Error vs Slice Count): {corr_slices}")
    else:
        print("Correlation (Error vs Slice Count): N/A")

    # 2. Error vs Target Class (Point Biserial equivalent)
    if len(set(patient_overall)) > 1:
        corr_class = np.corrcoef(mean_loss_per_sample, patient_overall)[0, 1]
        print(f"Correlation (Error vs Patient Overall Class): {corr_class}")
    else:
        print("Correlation (Error vs Patient Overall Class): N/A")

    # ==========================================
    # 8. Submission Generation
    # ==========================================
    threshold = 0.8305734395980835

    if best_val_loss < threshold:
        train_lib.generate_submission(
            model, test_loader, config.SUBMISSION_PATH, device
        )


if __name__ == "__main__":
    main()
