import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from library.config import Config
from library.dataset import IceCubeDataset
from library.model import SpatiotemporalPointTransformer
from library.utils import set_seed, azimuth_zenith_to_vector, vector_to_azimuth_zenith


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    Returns: average_loss, average_angular_error
    """
    model.train()
    running_loss = 0.0
    running_angle_error = 0.0
    count = 0

    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        # targets: (Batch, 2) -> [azimuth, zenith]
        targets = targets.to(device)

        # Convert targets to 3D unit vectors for cosine similarity loss
        target_vectors = azimuth_zenith_to_vector(targets[:, 0], targets[:, 1]).to(
            device
        )

        optimizer.zero_grad()

        # Forward pass -> (Batch, 3)
        pred_vectors = model(features)

        # Loss: 1 - Cosine Similarity
        # F.cosine_similarity normalizes vectors internally
        cos_sim = torch.nn.functional.cosine_similarity(
            pred_vectors, target_vectors, dim=1
        )
        loss = 1.0 - cos_sim.mean()

        loss.backward()

        # Gradient clipping to stabilize Transformer training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Metrics
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size

        # Angular Error: arccos(clamp(cos_sim))
        # Clamp to avoid NaN at boundaries (-1, 1) due to float precision
        clamped_cos = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angle_error = torch.acos(clamped_cos)
        running_angle_error += angle_error.sum().item()

        count += batch_size

    epoch_loss = running_loss / count
    epoch_angle_error = running_angle_error / count

    return epoch_loss, epoch_angle_error


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns: average_loss, average_angular_error
    """
    model.eval()
    running_loss = 0.0
    running_angle_error = 0.0
    count = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            target_vectors = azimuth_zenith_to_vector(targets[:, 0], targets[:, 1]).to(
                device
            )

            pred_vectors = model(features)

            cos_sim = torch.nn.functional.cosine_similarity(
                pred_vectors, target_vectors, dim=1
            )
            loss = 1.0 - cos_sim.mean()

            clamped_cos = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            angle_error = torch.acos(clamped_cos)

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            running_angle_error += angle_error.sum().item()
            count += batch_size

    return running_loss / count, running_angle_error / count


def generate_submission(model, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    # Load Test Dataset (loads all test data as per Config)
    test_dataset = IceCubeDataset(mode="test", subset_size=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)

            # Predict vectors
            pred_vectors = model(features)

            # Convert vectors back to angles
            azimuth, zenith = vector_to_azimuth_zenith(pred_vectors)

            all_azimuths.append(azimuth.cpu().numpy())
            all_zeniths.append(zenith.cpu().numpy())

    # Concatenate results
    if len(all_azimuths) > 0:
        all_azimuths = np.concatenate(all_azimuths)
        all_zeniths = np.concatenate(all_zeniths)
    else:
        all_azimuths = np.array([])
        all_zeniths = np.array([])

    # Get event IDs from dataset
    # IceCubeDataset loads all requested data into memory arrays, so indices align with loader
    event_ids = test_dataset.event_ids

    # Create DataFrame
    df = pd.DataFrame(
        {"event_id": event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    train_subset_size=None, val_subset_size=None, epochs=None, batch_size=None
):
    """
    Main execution function for training and submission generation.
    """
    # Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Configuration Overrides
    train_size = (
        train_subset_size if train_subset_size is not None else Config.TRAIN_SUBSET_SIZE
    )
    val_size = (
        val_subset_size if val_subset_size is not None else Config.VAL_SUBSET_SIZE
    )
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    b_size = batch_size if batch_size is not None else Config.BATCH_SIZE

    print(f"Initializing Datasets (Train: {train_size}, Val: {val_size})...")
    train_dataset = IceCubeDataset(mode="train", subset_size=train_size)
    val_dataset = IceCubeDataset(mode="val", subset_size=val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print("Initializing Model...")
    model = SpatiotemporalPointTransformer().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler Setup
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(steps_per_epoch * Config.WARMUP_EPOCHS)

    # Warmup: Linear increase from 1% LR to 100% LR
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    # Main: Cosine Decay
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metric = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_metric = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss}")
        print(f"  Train Mean Angular Error: {train_metric}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val Mean Angular Error: {val_metric}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training Complete.")

    # Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, device, Config.SUBMISSION_PATH)
