import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import ThoracicDataset
from library.model import EfficientNetBiFPN
from library.engine import fit, validate
from library.inference import predict_and_format


def analyze_failures(model, dataloader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and number of findings.
    """
    model.eval()
    errors = []
    num_findings = []

    # We will use the L1 distance between the global classification prediction
    # and the target as a proxy for error magnitude.

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch["target"].items()}

            outputs = model(images)

            # Global Head Error
            # global_logits: (B, 1)
            # global_label: (B)
            preds = torch.sigmoid(outputs["global_logits"])
            labels = targets["global_label"].view(-1, 1)

            # L1 Error per sample
            batch_errors = torch.abs(preds - labels).cpu().numpy().flatten()
            errors.extend(batch_errors)

            # Feature: Number of findings (sum of reg_mask per image)
            # reg_mask is (B, H, W), where 1 indicates an object center
            batch_findings = (
                targets["reg_mask"].view(images.size(0), -1).sum(dim=1).cpu().numpy()
            )
            num_findings.extend(batch_findings)

    # Calculate Correlation
    if len(errors) > 0:
        df = pd.DataFrame({"error": errors, "num_findings": num_findings})
        # Calculate Pearson correlation
        corr = df.corr().iloc[0, 1]
        print(f"Correlation between Error Magnitude and Number of Findings: {corr:.4f}")
    else:
        print("Not enough data for failure analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = Config.DEVICE

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    full_train_dataset = ThoracicDataset(split="train", load_cached_data=True)
    val_dataset = ThoracicDataset(split="val", load_cached_data=True)

    # Subsample Training Data for Fast Baseline
    # Limit to 4000 samples to ensure quick execution
    MAX_TRAIN_SAMPLES = 4000
    if len(full_train_dataset) > MAX_TRAIN_SAMPLES:
        indices = np.random.choice(
            len(full_train_dataset), MAX_TRAIN_SAMPLES, replace=False
        )
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Subsampled training set to {len(train_dataset)} samples.")
    else:
        train_dataset = full_train_dataset
        print(f"Using full training set ({len(full_train_dataset)} samples).")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = EfficientNetBiFPN().to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Short training schedule for baseline
    NUM_EPOCHS = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # Load GT for validation mAP calculation
    gt_df = pd.read_csv(Config.VAL_META_PATH)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        gt_df=gt_df,
    )

    # 5. Final Validation
    print("Performing final validation...")
    # Load best model weights to ensure accurate metric reporting
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model for validation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    val_loss, val_map = validate(model, val_loader, device, gt_df)
    print(f"Final Validation Metric: {val_map}")

    # 6. Failure Analysis
    print("Running failure analysis...")
    analyze_failures(model, val_loader, device)

    # 7. Submission
    SUBMISSION_THRESHOLD = 0.1783551866
    if val_map > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric {val_map} > {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        test_dataset = ThoracicDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        predict_and_format(model, test_loader, device, output_path)
    else:
        print(
            f"Validation metric {val_map} <= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
