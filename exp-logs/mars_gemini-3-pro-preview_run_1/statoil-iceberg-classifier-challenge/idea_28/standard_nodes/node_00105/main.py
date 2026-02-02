import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_global_stats
from library.calibration import run_calibration
from library.production import train_production_models
from library.inference import run_inference
from library.dataset import get_dataset
from library.augmentation import get_test_transforms
from library.model import IcebergResNet18


def main():
    # 1. Setup
    print("Initializing Pipeline...")
    Config.create_directories()
    seed_everything(Config.SEED)

    # Compute global stats for normalization
    compute_global_stats(load_cached_data=True)

    # --- Optimize Config for Fast Baseline Execution ---
    # We reduce epochs and patience to ensure the script completes within the time limit
    # while still verifying the architectural logic.
    print("Configuring hyperparameters for fast execution...")
    Config.PHASE1_MAX_EPOCHS = 10
    Config.PHASE1_PATIENCE = 3
    Config.SWA_CYCLES = 2
    Config.SWA_CYCLE_LEN = 2
    Config.BATCH_SIZE = 64

    # 2. Phase 1: Calibration
    # Runs 5-Fold CV to find optimal trajectory
    print("\n" + "=" * 40)
    print("Phase 1: Calibration")
    print("=" * 40)
    optimal_epochs, milestones, final_lr = run_calibration()

    # 3. Phase 2: Production
    # Trains 5 models on Full Data (Train+Val) using discovered trajectory
    print("\n" + "=" * 40)
    print("Phase 2: Production Training")
    print("=" * 40)
    train_production_models(optimal_epochs, milestones, final_lr)

    # 4. Validation & Failure Analysis
    print("\n" + "=" * 40)
    print("Validation & Failure Analysis")
    print("=" * 40)

    device = torch.device(Config.DEVICE)

    # Load Validation Dataset
    val_ds = get_dataset("val", transform=get_test_transforms(), load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Load Ensemble Models
    models = []
    num_models = 5
    for i in range(num_models):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
        if os.path.exists(ckpt_path):
            base_model = IcebergResNet18().to(device)
            swa_model = AveragedModel(base_model).to(device)
            swa_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            swa_model.eval()
            models.append(swa_model)
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found.")

    if not models:
        print("No models found for validation.")
        return

    # Ensemble Inference on Validation Set
    all_probs = []
    all_targets = []
    all_angles = []
    all_images_stats = []  # Store mean/std for failure analysis

    print("Running validation inference...")
    with torch.no_grad():
        for data in val_loader:
            images, angles, labels = data
            images = images.to(device)
            angles_gpu = angles.to(device)

            # Store metadata for analysis
            all_targets.extend(labels.numpy())
            all_angles.extend(angles.numpy())  # These are normalized angles

            # Calculate image stats (on CPU to save GPU mem)
            # images is (B, 3, 224, 224). We want stats from original bands if possible,
            # but here we have processed tensors. We'll use channel 0 (HH) and 1 (HV).
            imgs_np = images.cpu().numpy()
            for img in imgs_np:
                # img shape (3, 224, 224)
                b1 = img[0].flatten()
                b2 = img[1].flatten()
                all_images_stats.append(
                    {
                        "b1_mean": np.mean(b1),
                        "b1_std": np.std(b1),
                        "b2_mean": np.mean(b2),
                        "b2_std": np.std(b2),
                    }
                )

            # TTA Inference
            batch_probs = []
            for model in models:
                # 1. Original
                out1 = torch.sigmoid(model(images, angles_gpu))
                # 2. HFlip
                out2 = torch.sigmoid(model(torch.flip(images, [3]), angles_gpu))
                # 3. VFlip
                out3 = torch.sigmoid(model(torch.flip(images, [2]), angles_gpu))
                # 4. R180
                out4 = torch.sigmoid(model(torch.flip(images, [2, 3]), angles_gpu))

                avg_out = (out1 + out2 + out3 + out4) / 4.0
                batch_probs.append(avg_out.cpu().numpy().flatten())

            # Average across models
            ensemble_batch_prob = np.mean(batch_probs, axis=0)
            all_probs.extend(ensemble_batch_prob)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Compute Metric
    # Clip to avoid log(0)
    clipped_probs = np.clip(all_probs, 1e-15, 1 - 1e-15)
    log_loss = -np.mean(
        all_targets * np.log(clipped_probs)
        + (1 - all_targets) * np.log(1 - clipped_probs)
    )

    print(f"Final Validation Metric: {log_loss:.15f}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(all_targets - all_probs)

    df_analysis = pd.DataFrame(all_images_stats)
    df_analysis["error"] = errors
    df_analysis["inc_angle_norm"] = all_angles

    # Compute correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error and Features:")
    print(correlations)

    # 5. Submission
    print("\n" + "=" * 40)
    print("Submission Generation")
    print("=" * 40)

    threshold = 0.16918645240183008
    if log_loss < threshold:
        print(
            f"Validation metric ({log_loss:.6f}) meets threshold ({threshold:.6f}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"Validation metric ({log_loss:.6f}) did not meet threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
