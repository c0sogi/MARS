import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add library path if needed
sys.path.append(".")

from library.config import Config, seed_everything
from library.pipeline import run_cv_training, generate_ensemble_submission
from library.dataset import process_data

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("--- Starting Optimized 5-Fold CV Pipeline (Image + Depth) ---")

    # 2. Run Training Pipeline (5-Fold CV)
    # This trains models and returns the paths to the best checkpoints and the optimal global threshold
    valid_model_paths, best_threshold, final_map = run_cv_training(debug=False)

    # 3. Final Validation Reporting
    print("\n--- Final Validation Report ---")
    print(f"Final Global OOF mAP: {final_map:.10f}")

    # 4. Submission Logic
    if final_map > 0.7985:
        print("Validation metric meets threshold. Generating Ensemble Submission.")
        generate_ensemble_submission(valid_model_paths, best_threshold)
    else:
        print(f"Validation metric {final_map} <= 0.7985. No submission generated.")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
