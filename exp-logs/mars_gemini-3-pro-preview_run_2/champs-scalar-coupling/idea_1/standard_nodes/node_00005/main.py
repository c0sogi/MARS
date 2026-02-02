import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.engine import Engine
from library.data import get_dataloaders
from library.utils import set_seed, calculate_log_mae


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup for Fast Baseline
    # ---------------------------------------------------------
    print("Setting up configuration for fast baseline...")

    # Override Config for speed and specific output requirements
    Config.DEBUG_SAMPLE_SIZE = None  # Use full dataset

    # Clear old cache files to ensure full dataset is processed
    # Cite debug_lesson_4: Verify data consistency (ensure we don't load subsampled cache)
    if os.path.exists(Config.IDEA_WORK_DIR):
        for filename in os.listdir(Config.IDEA_WORK_DIR):
            if filename.startswith("cached_") and filename.endswith("_v2.npz"):
                file_path = os.path.join(Config.IDEA_WORK_DIR, filename)
                os.remove(file_path)
                print(f"Removed old cache file: {file_path}")

    # Update submission path to match requirements: ./submission/submission.csv
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Training and Submission Generation
    # ---------------------------------------------------------
    print("Initializing Engine...")
    engine = Engine()

    print("Starting Training and Submission Pipeline...")
    # engine.run() trains the model, saves the best version, and generates submission.csv
    engine.run()

    # ---------------------------------------------------------
    # 3. Final Validation Assessment & Failure Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("Post-Training Analysis")
    print("=" * 40)

    # Load the best model for analysis
    best_model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(best_model_path):
        print("Error: Best model not found.")
        return

    print(f"Loading best model from {best_model_path} for analysis...")
    model = engine.model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # Get DataLoaders (using cached data if available)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_types = []
    all_distances = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(Config.DEVICE)

            # Forward pass
            preds = model(batch)
            targets = batch.y

            # Calculate Distances for the couples in the batch
            # Logic mirrors model.py forward pass to ensure alignment
            if hasattr(batch, "ptr"):
                # Batch mode
                node_offsets = batch.ptr[:-1]
                couple_counts = batch.num_couples
                shifts = torch.repeat_interleave(node_offsets, couple_counts)
                idx0 = batch.couple_idx[:, 0] + shifts
                idx1 = batch.couple_idx[:, 1] + shifts
            else:
                idx0 = batch.couple_idx[:, 0]
                idx1 = batch.couple_idx[:, 1]

            pos0 = batch.pos[idx0]
            pos1 = batch.pos[idx1]
            dist = torch.norm(pos0 - pos1, p=2, dim=-1)

            # Collect data
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_types.append(batch.couple_type.cpu().numpy().flatten())
            all_distances.append(dist.cpu().numpy().flatten())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_types = np.concatenate(all_types)
    all_distances = np.concatenate(all_distances)

    # Denormalize predictions and targets
    preds_raw = all_preds * Config.TARGET_STD + Config.TARGET_MEAN
    targets_raw = all_targets * Config.TARGET_STD + Config.TARGET_MEAN

    # ---------------------------------------------------------
    # 4. Metric Calculation
    # ---------------------------------------------------------
    final_metric = calculate_log_mae(preds_raw, targets_raw, all_types)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\nFailure Analysis:")

    # Calculate Absolute Error
    abs_error = np.abs(preds_raw - targets_raw)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "abs_error": abs_error,
            "distance": all_distances,
            "coupling_type_idx": all_types,
            "target_magnitude": np.abs(targets_raw),
        }
    )

    # Correlation with Distance
    corr_dist, _ = pearsonr(df_analysis["abs_error"], df_analysis["distance"])
    print(f"Correlation (Error vs Distance): {corr_dist:.4f}")

    # Correlation with Target Magnitude
    corr_mag, _ = pearsonr(df_analysis["abs_error"], df_analysis["target_magnitude"])
    print(f"Correlation (Error vs Target Magnitude): {corr_mag:.4f}")

    # Correlation with Coupling Type (Categorical Index - rough proxy)
    corr_type, _ = pearsonr(df_analysis["abs_error"], df_analysis["coupling_type_idx"])
    print(f"Correlation (Error vs Coupling Type Index): {corr_type:.4f}")

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()
