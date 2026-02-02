import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, calculate_rmse, tiled_inference
from library.model import TSPCResUNet
from library.dataset import DenoisingDataset


class InferenceWrapper(nn.Module):
    """
    Wraps the TSPCResUNet to return only the final stage output (res2)
    during inference, as tiled_inference expects a single tensor return.
    """

    def __init__(self, model):
        super(InferenceWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        _, res2 = self.model(x)
        return res2


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    Computes sum of losses from both stages.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Ground truth noise residual
        noise_gt = noisy - clean

        optimizer.zero_grad()

        # Forward pass: returns (res1, res2)
        res1, res2 = model(noisy)

        # Calculate loss for both stages
        loss1 = criterion(res1, noise_gt)
        loss2 = criterion(res2, noise_gt)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, dataset, device):
    """
    Validation loop.
    Uses tiled_inference to process full images and calculates RMSE.
    """
    # Wrap model to return only the final residual
    infer_model = InferenceWrapper(model)
    infer_model.eval()

    rmse_list = []

    # Iterate directly over the dataset (returns full images)
    # dataset[i] returns (noisy_tensor, clean_tensor, id)
    for i in range(len(dataset)):
        noisy, clean, _ = dataset[i]

        noisy = noisy.to(device)
        # clean is on CPU, keep it there for metric calc or move to CPU later

        # Predict noise residual using tiled inference
        # tiled_inference handles moving patches to device and stitching back to CPU
        # It expects input (C, H, W) or (1, C, H, W)
        with torch.no_grad():
            pred_noise = tiled_inference(
                infer_model,
                noisy,
                patch_size=Config.PATCH_SIZE,
                overlap_ratio=Config.OVERLAP_RATIO,
                batch_size=Config.BATCH_SIZE,
                device=device,
            )

        # Reconstruct clean image: Clean = Noisy - Noise_Pred
        # noisy is on GPU, pred_noise is on CPU (from tiled_inference)
        noisy_cpu = noisy.cpu().squeeze(0)  # (C, H, W)
        pred_clean = noisy_cpu - pred_noise

        # Clip values to [0, 1] as pixel intensities are bounded
        pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

        # Calculate RMSE
        clean_cpu = clean.squeeze(0)  # (C, H, W)
        score = calculate_rmse(pred_clean, clean_cpu)
        rmse_list.append(score)

    return np.mean(rmse_list)


def run_training():
    """
    Main execution function for training the TSP-CResUNet.
    """
    print(f"Starting experiment: {Config.EXPERIMENT_NAME}")
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --- 1. Data Loading ---
    print("Loading datasets...")
    # Train dataset: High density patches
    train_dataset = DenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        root_dir=Config.INPUT_DIR,
        augment=True,
        train_mode=True,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    # Validation dataset: Full images
    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        augment=False,
        train_mode=False,
        load_cached_data=True,
    )

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- 2. Model Initialization ---
    print("Initializing TSP-CResUNet model...")
    model = TSPCResUNet().to(device)

    # --- 3. Optimization ---
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # --- 4. Training Loop ---
    best_rmse = float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_dataset, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val RMSE: {val_rmse}"
        )

        # Checkpointing & Early Stopping
        if val_rmse < best_rmse:
            print(
                f"Validation RMSE improved from {best_rmse} to {val_rmse}. Saving model..."
            )
            best_rmse = val_rmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation RMSE: {best_rmse}")
    print(f"Best model saved to: {Config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    # Ensure working directory exists (handled by Config, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    run_training()
