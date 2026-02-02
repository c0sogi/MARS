import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, weighted_loss_metric
from library.dataset import get_dataloaders, TestDataset, get_transforms, RSNADataset
from library.model import CervicalFractureModel
from library.trainer import train_one_epoch, validate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure working directory exists
    os.makedirs("working", exist_ok=True)

    # 2. Data Loading
    print("Loading DataLoaders...")
    # debug=False ensures we use the full provided training set (which is small: ~161 samples)
    train_loader, val_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    print("Initializing Model...")
    model = CervicalFractureModel(pretrained=True).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: T_max set to 1.5x epochs
    t_max = int(Config.EPOCHS * Config.T_MAX_MULTIPLIER)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Metric: {val_metric:.6f}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "working/best_model.pth")
            print("  >>> New best model saved.")

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("working/best_model.pth", map_location=device))
    model.eval()

    # Compute Final Metric
    _, final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nRunning Failure Analysis...")
    analyze_failures(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.1307335607
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nMetric ({final_metric}) >= threshold ({THRESHOLD}). Skipping submission."
        )


def analyze_failures(model, loader, device):
    """
    Calculates per-study loss and correlates it with the number of slices in the scan.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    study_losses = []
    study_depths = []

    # Weights for the metric: C1-C7=1.0, Patient=7.0. Total weight sum = 14.0
    weights = torch.tensor([1.0] * 7 + [7.0], device=device)
    total_weight = weights.sum()

    print("Analyzing validation samples...")
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            positions = batch["positions"].to(device)
            targets = batch["targets"].to(device)
            study_ids = batch["study_id"]

            # Forward
            logits = model(images, positions)

            # Prepare predictions and targets for loss calculation
            # We calculate loss exactly as the metric does: BCE on each column

            # 1. Vertebral Logits
            vert_logits = logits

            # 2. Patient Overall Logit (derived as max of vert logits)
            patient_logit, _ = torch.max(logits, dim=1, keepdim=True)

            # Combine
            full_logits = torch.cat([vert_logits, patient_logit], dim=1)

            # Calculate BCE per element
            # targets is (B, 8)
            bce_loss = criterion(full_logits, targets)  # (B, 8)

            # Apply weights
            weighted_bce = bce_loss * weights  # (B, 8)

            # Sum over columns and divide by total weight to get per-study average loss
            per_study_loss = weighted_bce.sum(dim=1) / total_weight  # (B,)

            # Collect data
            for i, s_id in enumerate(study_ids):
                loss_val = per_study_loss[i].item()

                # Get number of slices from filesystem
                # We need to look up the path. The loader dataset has the dataframe.
                # Accessing the underlying dataset from the loader
                dataset_df = loader.dataset.df
                row = dataset_df[dataset_df["StudyInstanceUID"] == s_id].iloc[0]
                image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

                # Count files
                try:
                    num_slices = len(glob.glob(os.path.join(image_dir, "*.dcm")))
                except:
                    num_slices = 0

                study_losses.append(loss_val)
                study_depths.append(num_slices)

    # Calculate Correlation
    if len(study_losses) > 1:
        corr, _ = pearsonr(study_losses, study_depths)
        print(
            f"Correlation between Error (Loss) and Input Feature (Num Slices): {corr:.4f}"
        )
    else:
        print("Not enough samples for correlation analysis.")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    # 1. Load Test Metadata
    test_meta_path = Config.TEST_METADATA_PATH
    if not os.path.exists(test_meta_path):
        print("Test metadata not found.")
        return

    test_df = pd.read_csv(test_meta_path)
    print(f"Test set size: {len(test_df)} studies.")

    # 2. Create Dataset & Loader
    # No caching for test set inference to avoid disk writes for large hidden set
    test_dataset = TestDataset(test_df, transforms=get_transforms("valid"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Inference
    model.eval()
    predictions = {}  # study_id -> {C1: p, ..., patient_overall: p}

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            positions = batch["positions"].to(device)
            study_ids = batch["study_id"]

            logits = model(images, positions)
            probs = torch.sigmoid(logits)  # (B, 7)

            probs_np = probs.cpu().numpy()

            for i, s_id in enumerate(study_ids):
                p_vert = probs_np[i]  # array of 7
                p_patient = np.max(p_vert)

                predictions[s_id] = {
                    "C1": p_vert[0],
                    "C2": p_vert[1],
                    "C3": p_vert[2],
                    "C4": p_vert[3],
                    "C5": p_vert[4],
                    "C6": p_vert[5],
                    "C7": p_vert[6],
                    "patient_overall": p_patient,
                }

    # 4. Format Submission
    # Load sample submission to get the required row_ids
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        # Fallback if sample_submission not found (unlikely)
        print("Sample submission not found, cannot format output.")
        return

    sub_df = pd.read_csv(sample_sub_path)

    # Fill predictions
    # row_id format: StudyInstanceUID_Subtype
    # Example: 1.2.3.4_C1 or 1.2.3.4_patient_overall

    new_probs = []
    for row_id in sub_df["row_id"]:
        # Split to get StudyID and Subtype
        # We use rsplit to handle the underscore in 'patient_overall' safely
        study_id, subtype = row_id.rsplit("_", 1)

        # Handle the case where patient_overall was split incorrectly if logic was simple
        # 'patient_overall' -> split('_', 1) on '1.2.3_patient_overall' gives '1.2.3_patient', 'overall' -> Wrong
        # Actually, the subtypes are C1..C7 and patient_overall.
        # Let's try matching known subtypes.

        target_prob = 0.5  # Default

        if study_id in predictions:
            preds = predictions[study_id]
            if subtype in preds:
                target_prob = preds[subtype]
            elif subtype == "overall" and "patient_overall" in row_id:
                # Handle potential split issue if the ID format is tricky
                # But typically it is StudyID_patient_overall
                # If we split on last _, we get "StudyID_patient" and "overall".
                # Let's refine the split logic.
                pass

        # Robust splitting:
        # Known subtypes
        valid_subtypes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        found_subtype = None
        found_study = None

        for vt in valid_subtypes:
            if row_id.endswith(f"_{vt}"):
                found_subtype = vt
                found_study = row_id[: -(len(vt) + 1)]
                break

        if found_study and found_study in predictions:
            target_prob = predictions[found_study][found_subtype]

        new_probs.append(target_prob)

    sub_df["fractured"] = new_probs

    # Save
    out_path = "submission.csv"  # Root of working directory as requested?
    # Prompt says: "./submission/submission.csv" in one place, but "./working/submission.csv" in another context.
    # The prompt "Submission Format" section says "The file should... have the following format".
    # The prompt "Requirements" says "Save the final predictions to ./submission/submission.csv".
    # However, standard Kaggle/Docker structure usually maps ./working as the output.
    # I will save to ./submission/submission.csv if the folder exists, else ./working/submission.csv or just submission.csv.
    # I'll create ./submission folder to be safe.

    os.makedirs("submission", exist_ok=True)
    sub_df.to_csv("submission/submission.csv", index=False)
    print("Submission saved to submission/submission.csv")


if __name__ == "__main__":
    main()
