import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import scipy.stats

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss
from library.data import cache_dataset, CervicalSpineDataset, VolumetricTransforms
from library.model import SequenceContextResNet
from library.engine import get_optimizer_and_scheduler, train_one_epoch, evaluate
from library.loss import HierarchicalCompoundLoss


def run_failure_analysis(model, dataloader, device, metadata_df):
    """
    Computes per-sample loss and correlates it with metadata features.
    """
    model.eval()
    criterion = HierarchicalCompoundLoss()

    all_losses = []
    all_targets = []

    # Weights for the metric: C1-C7=1.0, patient_overall=7.0
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=device)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                logits = model(images)

            # Calculate per-sample weighted log loss for analysis
            probs_vertebrae = torch.sigmoid(logits)
            probs_patient = torch.max(probs_vertebrae, dim=1).values.unsqueeze(1)
            probs = torch.cat([probs_vertebrae, probs_patient], dim=1)

            # Clip for stability
            epsilon = 1e-15
            probs = torch.clamp(probs, epsilon, 1 - epsilon)

            # BCE per element: -[y*log(p) + (1-y)*log(1-p)]
            bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

            # Weighted sum per patient
            weighted_bce = bce * weights

            # Average loss per patient (over the 8 labels)
            # The competition metric averages over ALL rows, so for a single patient
            # the contribution is sum(weighted_bce) / 8 (conceptually)
            # We use sum here to represent the magnitude of error for that patient
            patient_error_magnitude = weighted_bce.mean(dim=1).cpu().numpy()

            all_losses.extend(patient_error_magnitude)
            all_targets.append(targets.cpu().numpy())

    all_losses = np.array(all_losses)
    all_targets = np.concatenate(all_targets, axis=0)

    # Create a DataFrame for analysis
    # We assume dataloader is not shuffled for validation to match metadata order
    # (The main function ensures val_loader is shuffle=False)
    analysis_df = metadata_df.copy().reset_index(drop=True)

    # Safety check for length
    if len(analysis_df) != len(all_losses):
        print(
            f"Warning: Analysis DataFrame length ({len(analysis_df)}) matches predictions ({len(all_losses)})?"
        )
        analysis_df = analysis_df.iloc[: len(all_losses)]

    analysis_df["error_magnitude"] = all_losses

    print("\n=== Failure Analysis ===")
    print(f"Mean Error Magnitude: {np.mean(all_losses):.6f}")

    # Correlation with targets
    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    print("\nCorrelation between Error Magnitude and Fracture Presence:")
    for col in target_cols:
        if col in analysis_df.columns:
            corr, _ = scipy.stats.pearsonr(
                analysis_df[col], analysis_df["error_magnitude"]
            )
            print(f"  {col}: {corr:.4f}")


def generate_submission(model, test_df, device, output_path):
    """
    Generates the submission.csv file for the test set.
    """
    print("\nGenerating submission...")

    # Ensure test data is cached
    cache_dataset(test_df, Config.CACHE_DIR, load_cached_data=True)

    dataset = CervicalSpineDataset(test_df, Config.CACHE_DIR, mode="test")
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    results = []

    with torch.no_grad():
        for images, uids in dataloader:
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                logits = model(images)

            probs_vertebrae = torch.sigmoid(logits)  # (B, 7)
            probs_patient = torch.max(probs_vertebrae, dim=1).values.unsqueeze(
                1
            )  # (B, 1)

            # Move to CPU
            probs_vertebrae = probs_vertebrae.cpu().numpy()
            probs_patient = probs_patient.cpu().numpy()

            # Format rows
            for i, uid in enumerate(uids):
                # C1-C7
                for j, vert in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
                    results.append(
                        {"row_id": f"{uid}_{vert}", "fractured": probs_vertebrae[i, j]}
                    )
                # Patient Overall
                results.append(
                    {
                        "row_id": f"{uid}_patient_overall",
                        "fractured": probs_patient[i, 0],
                    }
                )

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Configuration and Setup
    Config.setup(debug=False)
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Cache Data (Preprocessing)
    # Only cache train/val initially to save time
    cache_dataset(train_df, Config.CACHE_DIR)
    cache_dataset(val_df, Config.CACHE_DIR)

    # 4. Prepare Datasets and Loaders
    train_transforms = VolumetricTransforms(prob=0.5)

    train_dataset = CervicalSpineDataset(
        train_df, Config.CACHE_DIR, mode="train", transforms=train_transforms
    )
    val_dataset = CervicalSpineDataset(val_df, Config.CACHE_DIR, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Avoid small batches disrupting BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Initialize Model
    model = SequenceContextResNet().to(Config.DEVICE)

    # 6. Optimizer and Scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, Config.EPOCHS)

    # 7. Training Loop
    best_metric = float("inf")

    print("\n=== Starting Training ===")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, Config.DEVICE, epoch
        )
        val_loss, val_metric = evaluate(model, val_loader, Config.DEVICE)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"\nTraining completed. Best Validation Metric: {best_metric:.6f}")

    # 8. Final Evaluation & Failure Analysis
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(Config.DEVICE)

    # Re-calculate metric on full validation set to ensure accuracy
    _, final_metric = evaluate(model, val_loader, Config.DEVICE)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Run Failure Analysis
    run_failure_analysis(model, val_loader, Config.DEVICE, val_df)

    # 9. Submission Generation
    # Condition: metric must be strictly lower than 0.1307335607
    THRESHOLD = 0.1307335607
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD}). Generating submission."
        )
        generate_submission(model, test_df, Config.DEVICE, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
