import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.data import get_loaders
from library.model import IcebergResNet18
from library.calibration import PlattScaler
from library.engine import train_one_epoch, evaluate, update_bn, predict_tta


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Phase 1: Exploration & Trajectory Extraction
    print("\n=== Phase 1: Exploration & Trajectory Extraction ===")

    model = IcebergResNet18().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Phase 1 uses ReduceLROnPlateau to find the schedule
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    milestones = []
    current_lr = Config.LEARNING_RATE

    # Training Loop
    for epoch in range(1, Config.MAX_EPOCHS_PHASE_1 + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, _, _ = evaluate(model, val_loader, device)

        # Check for LR changes (Trajectory Extraction)
        # Note: ReduceLROnPlateau updates LR after step() based on metric
        scheduler.step(val_loss)
        last_lr = optimizer.param_groups[0]["lr"]
        if last_lr < current_lr:
            print(f"LR dropped at epoch {epoch} from {current_lr} to {last_lr}")
            milestones.append(epoch)
            current_lr = last_lr

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                {"state_dict": model.state_dict()},
                is_best=True,
                filename="phase1_checkpoint.pth",
                best_filename="phase1_best_model.pth",
            )

        # Early Stopping check
        if epoch - best_epoch > Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Phase 1 Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss}")
    print(f"Trajectory Milestones: {milestones}")

    # SWA Phase (Phase 1)
    print("\n--- Starting SWA (Phase 1) ---")
    # Load best model to start SWA
    checkpoint = torch.load(
        os.path.join(Config.CHECKPOINT_DIR, "phase1_best_model.pth")
    )
    model.load_state_dict(checkpoint["state_dict"])

    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    for swa_epoch in range(Config.SWA_DURATION):
        train_one_epoch(model, train_loader, optimizer, device, f"SWA-{swa_epoch}")
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Update BN
    print("Updating SWA Batch Normalization...")
    update_bn(train_loader, swa_model, device)

    # Final Phase 1 Evaluation
    print("Evaluating SWA Model (Phase 1)...")
    final_val_loss, val_logits, val_targets = evaluate(swa_model, val_loader, device)
    print(f"Final Validation Metric: {final_val_loss}")

    # 4. Phase 2: Calibration
    print("\n=== Phase 2: Meta-Calibration ===")
    scaler = PlattScaler()
    scaler.fit(val_logits, val_targets)
    scaler.save()

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Collect angles from val_loader (order is preserved as shuffle=False)
    val_angles = []
    for batch in val_loader:
        val_angles.extend(batch["angle"].numpy())
    val_angles = np.array(val_angles)

    # Calculate per-sample loss/error
    val_probs = scaler.predict_proba(val_logits)  # Use calibrated probs for analysis

    # Log Loss contribution: - (y log(p) + (1-y) log(1-p))
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    sample_losses = -(
        val_targets * np.log(val_probs_clipped)
        + (1 - val_targets) * np.log(1 - val_probs_clipped)
    )

    # Correlation
    if len(val_angles) == len(sample_losses):
        correlation = np.corrcoef(val_angles, sample_losses)[0, 1]
        print(f"Correlation between Incidence Angle and Log Loss: {correlation:.4f}")
    else:
        print("Warning: Mismatch in lengths for correlation analysis.")

    # 6. Submission Logic
    THRESHOLD = 0.16918645240183008
    if final_val_loss < THRESHOLD:
        print(
            f"\nValidation Metric {final_val_loss} < {THRESHOLD}. Proceeding to Phase 3 & Submission."
        )

        # 7. Phase 3: Production (Full-Fit with Trajectory Replay)
        print("\n=== Phase 3: Production (Full-Fit) ===")

        # Combine Datasets
        full_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
        full_loader = DataLoader(
            full_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Re-init Model & Optimizer
        prod_model = IcebergResNet18().to(device)
        prod_optimizer = torch.optim.AdamW(
            prod_model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Trajectory Replay Scheduler
        # If no milestones found, use a dummy one far away
        if not milestones:
            milestones = [Config.MAX_EPOCHS_PHASE_1 + 100]

        prod_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            prod_optimizer, milestones=milestones, gamma=Config.SCHEDULER_FACTOR
        )

        # Train for exactly best_epoch
        print(f"Training for {best_epoch} epochs on full dataset...")
        for epoch in range(1, best_epoch + 1):
            train_one_epoch(prod_model, full_loader, prod_optimizer, device, epoch)
            prod_scheduler.step()

        # SWA Phase (Phase 3)
        print("Starting SWA (Phase 3)...")
        prod_swa_model = AveragedModel(prod_model).to(device)
        prod_swa_scheduler = SWALR(prod_optimizer, swa_lr=Config.SWA_LR)

        for swa_epoch in range(Config.SWA_DURATION):
            train_one_epoch(
                prod_model, full_loader, prod_optimizer, device, f"SWA-{swa_epoch}"
            )
            prod_swa_model.update_parameters(prod_model)
            prod_swa_scheduler.step()

        # Update BN
        print("Updating Production SWA BN...")
        update_bn(full_loader, prod_swa_model, device)

        # 8. Inference
        print("\n=== Inference ===")
        test_logits, test_ids = predict_tta(prod_swa_model, test_loader, device)

        # Calibrate
        test_probs = scaler.predict_proba(test_logits)

        # Save
        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric {final_val_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
