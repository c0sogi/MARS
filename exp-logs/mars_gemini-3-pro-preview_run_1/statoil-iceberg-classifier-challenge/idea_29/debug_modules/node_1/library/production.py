import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn

from library.config import Config
from library.utils import set_seed, AverageMeter, predict_with_klein_tta
from library.data import load_dataset, IcebergDataset, get_transforms
from library.model import IcebergResNet18


def apply_label_smoothing(labels, epsilon=0.05):
    """
    Applies label smoothing to binary labels.
    Formula: y_smooth = y * (1 - epsilon) + 0.5 * epsilon
    """
    return labels * (1 - epsilon) + 0.5 * epsilon


def train_one_epoch(model, loader, optimizer, device, epsilon):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels, epsilon)

        optimizer.zero_grad()
        logits = model(images, angles)
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def train_full_fit(model_idx, e_conv, full_loader, device):
    """
    Trains a single model on the full dataset using the Schedule-Mapped Protocol.

    Args:
        model_idx (int): Index of the model (0-4) for seeding and saving.
        e_conv (int): The optimal convergence epoch found in Phase 1.
        full_loader (DataLoader): DataLoader containing 100% of training data.
        device (torch.device): Compute device.

    Returns:
        str: Path to the saved checkpoint.
    """
    # 1. Seeding for Independence
    # We use base seed + model_idx to ensure each model in the ensemble is initialized differently
    # and sees data in a different order (due to shuffle=True in loader)
    current_seed = Config.SEED + model_idx
    set_seed(current_seed)

    print(f"\n--- Training Full-Fit Model {model_idx + 1} (Seed {current_seed}) ---")

    # 2. Model & Optimizer Setup
    model = IcebergResNet18().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_BASE, weight_decay=Config.WEIGHT_DECAY
    )

    # 3. Phase 2a: Cosine Annealing (Calibration Mapping)
    # Map the trajectory to exactly e_conv epochs, decaying to the SWA LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=e_conv, eta_min=Config.LR_SWA
    )

    print(f"Phase 2a: Training for {e_conv} epochs with Cosine Schedule...")

    for epoch in range(1, e_conv + 1):
        loss = train_one_epoch(
            model, full_loader, optimizer, device, Config.LABEL_SMOOTHING
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Print periodically
        if epoch % 5 == 0 or epoch == e_conv:
            print(f"  Epoch {epoch}/{e_conv} | Loss: {loss:.6f} | LR: {current_lr:.2e}")

    # 4. Phase 2b: Stochastic Weight Averaging (SWA)
    print(
        f"Phase 2b: Starting SWA for {Config.SWA_EPOCHS} epochs at LR {Config.LR_SWA}..."
    )

    swa_model = AveragedModel(model)

    # Ensure optimizer is locked at SWA LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.LR_SWA

    for swa_epoch in range(1, Config.SWA_EPOCHS + 1):
        loss = train_one_epoch(
            model, full_loader, optimizer, device, Config.LABEL_SMOOTHING
        )
        swa_model.update_parameters(model)
        print(f"  SWA Epoch {swa_epoch}/{Config.SWA_EPOCHS} | Loss: {loss:.6f}")

    # 5. Finalize SWA (BatchNorm Update)
    print("Updating SWA BatchNorm statistics...")
    update_bn(full_loader, swa_model, device=device)

    # 6. Save Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{model_idx}.pth")
    torch.save(swa_model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

    return checkpoint_path


def generate_submission(model_paths, device):
    """
    Generates predictions for the test set using the ensemble of SWA models.

    Args:
        model_paths (list): List of paths to saved model checkpoints.
        device (torch.device): Compute device.
    """
    print("\n--- Generating Submission ---")

    # 1. Load Test Data
    ds_test = load_dataset("test", load_cached_data=True)
    test_loader = DataLoader(
        ds_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Test Metadata for IDs
    df_test_meta = pd.read_csv(Config.TEST_META)
    test_ids = df_test_meta["id"].values

    # Array to store accumulated probabilities
    # Shape: (N_test, 1)
    ensemble_probs = np.zeros((len(ds_test), 1), dtype=np.float64)

    # 3. Iterate through models
    for path in model_paths:
        print(f"Inference with model: {os.path.basename(path)}")

        # Load Model
        model = IcebergResNet18().to(device)
        # SWA model state dict keys might differ slightly if not handled carefully,
        # but AveragedModel usually saves standard state_dict if we save swa_model.state_dict()
        # The architecture class matches.
        state_dict = torch.load(path, map_location=device)

        # Handle 'module.' prefix if it exists (though AveragedModel usually adds 'module' only if DataParallel)
        # AveragedModel wraps the module, so keys are usually 'module.layer...'
        # We need to load this into the base IcebergResNet18.
        # If the saved state_dict comes from AveragedModel, it has keys like 'module.backbone.conv1.weight'
        # But our IcebergResNet18 expects 'backbone.conv1.weight'.

        # Fix keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove 'module.'
            elif k.startswith("n_averaged"):
                continue  # Skip the counter buffer from SWA
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        # Predict
        model_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Klein TTA
                probs = predict_with_klein_tta(model, images, angles)
                model_preds.append(probs.cpu().numpy())

        # Accumulate
        ensemble_probs += np.concatenate(model_preds, axis=0)

    # 4. Average
    avg_probs = ensemble_probs / len(model_paths)

    # 5. Save Submission
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs.flatten()})

    # Ensure format matches sample_submission
    # id,is_iceberg
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False, float_format="%.15f")
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(df_sub.head())


def run_production_phase(e_conv):
    """
    Executes Phase 2: Production.
    Trains 5 Full-Fit SWA models and generates the submission.

    Args:
        e_conv (int): The optimal convergence epoch from Phase 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Production Phase on {device} with E_conv={e_conv}...")

    # 1. Prepare Full Dataset (Train + Val)
    print("Aggregating Train and Validation data for Full-Fit...")
    ds_train_part = load_dataset("train", load_cached_data=True)
    ds_val_part = load_dataset("val", load_cached_data=True)

    full_images = np.concatenate([ds_train_part.images, ds_val_part.images], axis=0)
    full_angles = np.concatenate([ds_train_part.angles, ds_val_part.angles], axis=0)
    full_labels = np.concatenate([ds_train_part.labels, ds_val_part.labels], axis=0)

    print(f"Total training samples: {len(full_labels)}")

    # Create Full Dataset with Training Transforms
    full_dataset = IcebergDataset(
        full_images, full_angles, full_labels, transform=get_transforms("train")
    )

    full_loader = DataLoader(
        full_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Train Ensemble
    model_paths = []
    # We train 5 models as per the plan
    for i in range(5):
        path = train_full_fit(i, e_conv, full_loader, device)
        model_paths.append(path)

    # 3. Generate Submission
    generate_submission(model_paths, device)

    print("Production Phase Complete.")
