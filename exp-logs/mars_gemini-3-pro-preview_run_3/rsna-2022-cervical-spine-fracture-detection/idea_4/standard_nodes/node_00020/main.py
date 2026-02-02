import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import FractureMILModel
from library.loss import HierarchicalCompoundLoss
from library.engine import train_one_epoch, validate


def calculate_weighted_log_loss(logits, targets):
    """
    Calculates the competition metric:
    Sum of (Mean LogLoss of C1-C7) and (LogLoss of patient_overall).
    This effectively weights the patient_overall prediction equal to the
    aggregate of all subtypes, satisfying the requirement to weight the
    'any' label more highly than specific subtypes.
    """
    # Sigmoid to get probabilities
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()

    # C1-C7 are indices 0-6
    c_losses = []
    for i in range(7):
        # Clip to avoid log(0)
        p = np.clip(probs[:, i], 1e-15, 1 - 1e-15)
        # Calculate log loss for this column
        l = log_loss(targets[:, i], p, labels=[0, 1])
        c_losses.append(l)

    mean_c_loss = np.mean(c_losses)

    # Patient overall is derived as max(C1...C7)
    probs_c = probs[:, :7]
    probs_patient = np.max(probs_c, axis=1)

    # Target patient overall is index 7
    target_patient = targets[:, 7]

    p_pat = np.clip(probs_patient, 1e-15, 1 - 1e-15)
    patient_loss = log_loss(target_patient, p_pat, labels=[0, 1])

    # Total metric
    total_metric = mean_c_loss + patient_loss
    return total_metric, probs_patient


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Use full provided dataset
    Config.DEBUG_SAMPLE_SIZE = None

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # 3. Model & Training Components
    print("Initializing model...")
    model = FractureMILModel().to(device)
    criterion = HierarchicalCompoundLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULT), eta_min=Config.MIN_LR
    )

    # 4. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # We suppress per-epoch printing to keep output clean,
        # but in a real run, logging is handled by the engine logger.

    # 5. Validation & Metric Calculation
    print("Loading best model for validation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for volumes, targets in val_loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Metric
    final_metric, patient_probs = calculate_weighted_log_loss(all_logits, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Calculate loss per sample to see correlation with patient_overall
    probs = torch.sigmoid(all_logits).numpy()
    targets_np = all_targets.numpy()

    sample_losses = []
    for i in range(len(targets_np)):
        # Vertebrae loss component
        p_c = np.clip(probs[i, :7], 1e-15, 1 - 1e-15)
        t_c = targets_np[i, :7]
        l_c = -np.mean(t_c * np.log(p_c) + (1 - t_c) * np.log(1 - p_c))

        # Patient loss component
        p_pat = np.clip(patient_probs[i], 1e-15, 1 - 1e-15)
        t_pat = targets_np[i, 7]
        l_pat = -(t_pat * np.log(p_pat) + (1 - t_pat) * np.log(1 - p_pat))

        sample_losses.append(l_c + l_pat)

    sample_losses = np.array(sample_losses)
    patient_labels = targets_np[:, 7]

    # Correlation
    if np.std(sample_losses) > 0 and np.std(patient_labels) > 0:
        correlation = np.corrcoef(sample_losses, patient_labels)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis: Correlation between Error Magnitude and Fracture Presence: {correlation:.4f}"
    )

    # 7. Submission
    THRESHOLD = 0.1307335607

    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")

        results = []
        subtypes = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

        with torch.no_grad():
            for i, (volumes, _) in enumerate(test_loader):
                volumes = volumes.to(device)

                # Get UIDs using the dataset dataframe
                start_idx = i * test_loader.batch_size
                end_idx = start_idx + volumes.size(0)
                batch_uids = test_loader.dataset.df.iloc[start_idx:end_idx][
                    "StudyInstanceUID"
                ].values

                logits = model(volumes)
                probs_c = torch.sigmoid(logits)
                # Derive patient_overall probability (Max of C1-C7)
                probs_pat, _ = torch.max(probs_c, dim=1, keepdim=True)

                batch_probs = torch.cat([probs_c, probs_pat], dim=1).cpu().numpy()

                for j, uid in enumerate(batch_uids):
                    row_probs = batch_probs[j]
                    for k, subtype in enumerate(subtypes):
                        results.append(
                            {"row_id": f"{uid}_{subtype}", "fractured": row_probs[k]}
                        )

        submission_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
