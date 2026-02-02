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

    # --- Optimize Config for Performance ---
    print("Configuring hyperparameters for optimal convergence...")
    # Cite solution_lesson_node_00065: Prioritize prolonged convergence.
    Config.PHASE1_MAX_EPOCHS = 75
    # Cite solution_lesson_node_00070: Increase patience to handle noise in small datasets.
    Config.PHASE1_PATIENCE = 10
    # Cite solution_lesson_node_00055: Ensure sufficient SWA duration after calibration.
    Config.SWA_CYCLES = 4
    Config.SWA_CYCLE_LEN = 3
    # Cite solution_lesson_node_00038: Smaller batch size increases gradient step volume.
    Config.BATCH_SIZE = 32

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

    # Threshold derived from requirements
    threshold = 0.16918645240183008

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
