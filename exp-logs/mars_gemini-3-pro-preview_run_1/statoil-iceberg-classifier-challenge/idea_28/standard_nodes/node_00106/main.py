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
    # Runs 5-Fold CV to find optimal trajectory and calculate unbiased CV score
    print("\n" + "=" * 40)
    print("Phase 1: Calibration")
    print("=" * 40)
    optimal_epochs, milestones, final_lr, cv_score = run_calibration()

    print(f"\nFinal Cross-Validation Score: {cv_score:.6f}")

    # 3. Decision & Production
    print("\n" + "=" * 40)
    print("Decision & Production")
    print("=" * 40)

    # Threshold derived from previous successful runs or competition baseline
    threshold = 0.30

    if cv_score < threshold:
        print(
            f"CV Score ({cv_score:.6f}) meets threshold ({threshold}). Proceeding to Production Training..."
        )

        # Trains 5 models on Full Data (Train+Val) using discovered trajectory
        # This function also generates the final submission.csv
        train_production_models(optimal_epochs, milestones, final_lr)

        print("\nProduction Training Complete.")
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"CV Score ({cv_score:.6f}) did not meet threshold ({threshold}). Skipping Production Training."
        )
        print("No submission generated.")


if __name__ == "__main__":
    main()
