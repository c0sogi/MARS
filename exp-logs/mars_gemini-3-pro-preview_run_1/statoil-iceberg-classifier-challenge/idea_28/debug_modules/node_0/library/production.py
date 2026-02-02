import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.dataset import get_dataset
from library.model import IcebergResNet18
from library.engine import train_one_epoch, cyclic_swa_step, update_bn
from library.utils import seed_everything
from library.augmentation import get_training_transforms, get_test_transforms


def predict_proba(model, dataloader, device):
    """
    Generates raw probabilities for the test set using TTA.
    Internal helper to aggregate predictions in memory.
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for data in dataloader:
            # Test loader yields (img, angle)
            images, angles = data

            images = images.to(device)
            angles = angles.to(device)

            # TTA: Klein Four-Group
            # 1. Original
            out1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (Flip W, dim 3)
            images_h = torch.flip(images, [3])
            out2 = torch.sigmoid(model(images_h, angles))

            # 3. Vertical Flip (Flip H, dim 2)
            images_v = torch.flip(images, [2])
            out3 = torch.sigmoid(model(images_v, angles))

            # 4. Rotate 180 (H + V)
            images_r180 = torch.flip(images, [2, 3])
            out4 = torch.sigmoid(model(images_r180, angles))

            # Average
            avg_out = (out1 + out2 + out3 + out4) / 4.0

            # Flatten and append
            probs.extend(avg_out.view(-1).cpu().numpy())

    return np.array(probs)


def train_production_models(optimal_epochs, milestones, final_lr):
    """
    Executes Phase 2: Production (Trajectory Replay + Cyclic SWA).
    Trains 5 independent models on the full dataset and generates the ensemble submission.

    Args:
        optimal_epochs (int): The fixed number of epochs for the main training phase.
        milestones (list): Epoch indices where LR should decay.
        final_lr (float): The learning rate at the end of the main phase, used as SWA max LR.
    """
    print("Starting Phase 2: Production (Trajectory Replay + Cyclic SWA)...")

    # Update Config with dynamic parameters from Phase 1
    Config.update_production_params(optimal_epochs, milestones, final_lr)

    device = torch.device(Config.DEVICE)

    # 1. Prepare Full Dataset (Train + Val)
    # We load both subsets and concatenate them
    train_ds_part = get_dataset(
        "train", transform=get_training_transforms(), load_cached_data=True
    )
    val_ds_part = get_dataset(
        "val", transform=get_training_transforms(), load_cached_data=True
    )

    full_dataset = ConcatDataset([train_ds_part, val_ds_part])

    full_loader = DataLoader(
        full_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Prepare Test Loader for inference
    test_ds = get_dataset(
        "test", transform=get_test_transforms(), load_cached_data=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    ensemble_probs = []

    # 2. Train 5 Independent Models
    num_models = 5
    for i in range(num_models):
        print(f"\n--- Training Production Model {i+1}/{num_models} ---")

        # Unique seed for each model to ensure diversity
        current_seed = Config.SEED + i
        seed_everything(current_seed)

        # Initialize Model
        model = IcebergResNet18().to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.OPTIMIZER_LR,
            weight_decay=Config.OPTIMIZER_WEIGHT_DECAY,
        )

        # Scheduler: Trajectory Replay
        # We use MultiStepLR to enforce the exact decay points found in calibration
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=Config.PHASE1_FACTOR
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        # --- Part A: Trajectory Replay (Main Training) ---
        print(f"Phase 2a: Replaying Trajectory for {optimal_epochs} epochs...")
        for epoch in range(1, optimal_epochs + 1):
            loss = train_one_epoch(
                model, full_loader, optimizer, criterion, device, epoch
            )
            scheduler.step()

        # --- Part B: Low-Energy Cyclic SWA ---
        print("Phase 2b: Entering Low-Energy Cyclic SWA...")

        # Initialize SWA Model
        swa_model = AveragedModel(model).to(device)

        # SWA Hyperparameters
        swa_start = optimal_epochs
        swa_cycles = Config.SWA_CYCLES
        cycle_len = Config.SWA_CYCLE_LEN
        total_swa_epochs = swa_cycles * cycle_len

        # SWA LR bounds based on final_lr from calibration
        swa_lr_max = final_lr
        swa_lr_min = final_lr / 2.0

        for swa_epoch_idx in range(1, total_swa_epochs + 1):
            current_epoch = swa_start + swa_epoch_idx

            # Train one epoch (gradients update the base model)
            train_one_epoch(
                model, full_loader, optimizer, criterion, device, current_epoch
            )

            # Cyclic Step (updates LR and averages parameters at cycle ends)
            cyclic_swa_step(
                model,
                swa_model,
                optimizer,
                current_epoch,
                swa_start,
                swa_lr_max,
                swa_lr_min,
                cycle_len,
            )

        # --- Finalize Model ---
        # Update BN statistics for the SWA model using the full dataset
        update_bn(full_loader, swa_model, device)

        # Save Checkpoint
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
        torch.save(swa_model.state_dict(), ckpt_path)
        print(f"Model {i+1} saved to {ckpt_path}")

        # --- Inference for Ensemble ---
        print(f"Generating predictions for Model {i+1}...")
        probs = predict_proba(swa_model, test_loader, device)
        ensemble_probs.append(probs)

    # 3. Aggregate Predictions
    print("\nAggregating Ensemble Predictions...")
    ensemble_probs = np.array(ensemble_probs)
    avg_probs = np.mean(ensemble_probs, axis=0)

    # 4. Save Submission
    # Load test IDs
    df_test = pd.read_csv(Config.TEST_META)
    ids = df_test["id"].values

    df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_probs})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False, float_format="%.15f")
    print(f"Final Ensemble Submission saved to {Config.SUBMISSION_PATH}")
