import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import log_loss

# Import from library
from library.utils import set_seed, get_device
from library.dataset import create_dataloaders
from library.model import FineTunedResNet18, FineTunedResNet34
from library.engine import train_one_epoch, evaluate, predict
import copy


def main():
    # Configuration
    BATCH_SIZE = 64
    LR = 1e-4
    # Cite solution_lesson_node_00004: Increased epochs for ResNet34 with RandomResizedCrop
    NUM_EPOCHS = 8
    # Cite solution_lesson_node_00002: Use full dataset
    MAX_SAMPLES = None

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"

    # 1. Setup
    set_seed(42)
    device = get_device()

    # 2. Data Loading
    dataloaders = create_dataloaders(
        batch_size=BATCH_SIZE,
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        max_samples=MAX_SAMPLES,
    )

    # 3. Model Initialization
    # Cite solution_lesson_node_00004: Upgrade to ResNet34
    model = FineTunedResNet34().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # Cite solution_lesson_node_00002: Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    best_model_wts = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device
        )
        val_loss = evaluate(model, dataloaders["val"], criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Cite solution_lesson_node_00004: Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model weights with Val Loss: {best_val_loss:.6f}")

    # 5. Final Validation Metric
    # Get predictions and labels for validation set
    # Note: predict() returns (ids, probs). For val set, 'ids' are actually labels.
    val_labels, val_probs = predict(model, dataloaders["val"], device)

    # Calculate Log Loss
    metric = log_loss(val_labels, val_probs)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    # Re-load validation metadata to get paths.
    # create_dataloaders slices the dataframe if max_samples is set.
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    if MAX_SAMPLES is not None:
        val_df = val_df.iloc[:MAX_SAMPLES]

    # Calculate errors: |Label - Prob|
    errors = np.abs(np.array(val_labels) - np.array(val_probs))

    # Extract features
    widths = []
    heights = []
    aspect_ratios = []

    for _, row in val_df.iterrows():
        filepath = os.path.join(INPUT_DIR, row["filepath"])
        img = cv2.imread(filepath)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Drop NaNs if any bad images
    analysis_df = analysis_df.dropna()

    # Calculate correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission
    if metric < 0.029256312873461456:
        test_ids, test_probs = predict(model, dataloaders["test"], device)

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission_df = pd.DataFrame({"id": test_ids, "label": test_probs})

        # Ensure ID is int
        submission_df["id"] = submission_df["id"].astype(int)

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {metric} is not lower than threshold 0.029536589555373065. Skipping submission."
        )


if __name__ == "__main__":
    main()
