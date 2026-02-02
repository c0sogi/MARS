import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device, get_logger
from library.data import get_dataloaders
from library.model import MultiStageConvNeXtMIL
from library.loss import RSNALoss
from library.engine import fit, validate, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = get_logger("runfile")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        logger.error(f"Metadata file not found: {Config.TRAIN_METADATA_PATH}")
        return

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Create DataLoaders
    # We create the test_loader here but do not pass it to fit() to control submission logic
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # 4. Initialize Model and Loss
    model = MultiStageConvNeXtMIL().to(device)
    criterion = RSNALoss().to(device)

    # 5. Training
    # We pass test_loader=None to prevent fit() from running inference automatically.
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        test_df=None,
        criterion=criterion,
        device=device,
    )

    # 6. Evaluation
    # Load the best model checkpoint
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Calculate and print the final validation metric
    val_metric = validate(val_loader, model, criterion, device)
    print(f"Final Validation Metric: {val_metric}")

    # 7. Failure Analysis
    # We need to correlate Error (Loss) with Input Features (e.g., Number of Slices)

    # A. Extract Feature: Number of Slices (Depth)
    val_depths = []
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        count = 0
        try:
            if os.path.exists(full_path):
                # Efficiently count files
                with os.scandir(full_path) as it:
                    for entry in it:
                        if entry.is_file():
                            count += 1
        except Exception:
            pass
        val_depths.append(count)

    val_depths = np.array(val_depths)

    # B. Calculate Per-Sample Error
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute unreduced BCE to get sample-wise loss
    bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Loss = Mean(BCE_C1..C7) + BCE_Patient
    loss_verts = bce_none(all_logits[:, :7], all_targets[:, :7]).mean(dim=1)
    loss_patient = bce_none(all_logits[:, 7], all_targets[:, 7])
    total_error = (loss_verts + loss_patient).numpy()

    # C. Compute Correlation
    if len(total_error) == len(val_depths) and len(total_error) > 1:
        # Check for zero variance to avoid NaN
        if np.std(total_error) > 0 and np.std(val_depths) > 0:
            corr_matrix = np.corrcoef(total_error, val_depths)
            corr = corr_matrix[0, 1]
        else:
            corr = 0.0
        print(f"Correlation between Error and Num_Slices: {corr}")
    else:
        print("Correlation between Error and Num_Slices: NaN (Insufficient data)")

    # 8. Conditional Submission
    THRESHOLD = 0.06429807151236185

    if val_metric < THRESHOLD:
        inference(test_loader, model, device, test_df)


if __name__ == "__main__":
    main()
