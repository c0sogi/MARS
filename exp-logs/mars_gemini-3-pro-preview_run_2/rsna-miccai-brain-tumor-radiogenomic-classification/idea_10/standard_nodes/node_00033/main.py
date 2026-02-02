import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.utils import set_seed, get_device, save_checkpoint
from library.data import get_dataloaders
from library.model import (
    AsymmetricEfficientNet,
    train_one_epoch,
    validate,
    generate_submission,
)


def run():
    # 1. Setup
    set_seed(42)
    device = get_device()

    # 2. Data Loading
    # Using cached data for speed as per instructions
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, load_cached_data=True
    )

    # 3. Model Initialization
    model = AsymmetricEfficientNet(num_classes=1, dropout_rate=0.5)
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. Training Loop
    epochs = 15
    best_auc = 0.0
    best_model_path = "./working/idea_10/best_model.pth"

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler based on Validation AUC
        scheduler.step(val_auc)

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

    # 6. Final Validation Metric
    print(f"Final Validation Metric: {best_auc}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")

    # Load best model for analysis
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get predictions and targets for validation set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_probs.extend(probs)
            val_targets.extend(targets.numpy().flatten())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    errors = np.abs(val_targets - val_probs)

    # Load validation metadata to extract features
    val_df = pd.read_csv("./metadata/val.csv")
    input_dir = "./input"

    # Ensure alignment (DataLoader iterates sequentially)
    if len(val_df) > len(errors):
        val_df = val_df.iloc[: len(errors)]

    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    print("Correlation between Error Magnitude and Slice Counts:")
    for mod in modalities:
        counts = []
        for _, row in val_df.iterrows():
            folder_path = os.path.join(input_dir, row[f"path_{mod}"])
            if os.path.exists(folder_path):
                # Count .dcm files
                try:
                    cnt = len(
                        [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
                    )
                except:
                    cnt = 0
                counts.append(cnt)
            else:
                counts.append(0)

        # Calculate correlation
        if len(counts) > 0 and np.std(counts) > 0:
            corr = np.corrcoef(errors, counts)[0, 1]
            print(f"{mod}_slices: {corr}")
        else:
            print(f"{mod}_slices: NaN (No variance)")

    # 8. Conditional Submission
    submission_threshold = 0.6254545454545455

    if best_auc > submission_threshold:
        print(
            f"Validation metric {best_auc} exceeds threshold {submission_threshold}. Generating submission..."
        )
        generate_submission(
            model, test_loader, device, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"Validation metric {best_auc} does not meet threshold {submission_threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
