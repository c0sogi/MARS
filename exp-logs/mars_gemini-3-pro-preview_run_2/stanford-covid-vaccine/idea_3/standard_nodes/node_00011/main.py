import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_mcrmse_numpy
from library.train import run_training
from library.inference import run_inference
from library.data import get_dataloaders
from library.model import RNA_Net

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing Fast Baseline Pipeline...")
    seed_everything(Config.SEED)

    # Adjust Config for optimized execution
    # Increasing epochs to ensure convergence with the larger model (Cite solution_lesson_node_00010)
    Config.EPOCHS = 25

    # 2. Training
    print(f"Starting training for {Config.EPOCHS} epochs...")
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        patience=5,  # Slightly stricter patience for fast baseline
    )

    # 3. Validation Assessment
    print("\nPerforming Validation Assessment...")
    device = Config.DEVICE

    # Load the best model
    model = RNA_Net()
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get validation data
    # We use the same loader function but only need the val_loader
    _, val_loader = get_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE)

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    val_preds = np.concatenate(all_preds, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

    # Compute Metric
    # The scored columns in the competition are indices 0, 1, 3 corresponding to:
    # reactivity, deg_Mg_pH10, deg_Mg_50C.
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Indices: 0, 1, 2, 3, 4
    scored_indices = [0, 1, 3]

    mcrmse = compute_mcrmse_numpy(val_preds, val_targets, scored_indices=scored_indices)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mcrmse}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaged over scored positions and scored columns)
    # val_preds: (N, 107, 5)
    seq_scored = Config.PRED_LEN
    p = val_preds[:, :seq_scored, :][:, :, scored_indices]
    t = val_targets[:, :seq_scored, :][:, :, scored_indices]

    # MSE per sample: mean over length and channels
    sample_mse = np.mean((p - t) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # Load validation metadata to correlate with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (loader preserves order if shuffle=False, which is true for val_loader)
    if len(val_df) != len(sample_rmse):
        print("Warning: Mismatch in validation set size for analysis.")
    else:
        val_df["error_rmse"] = sample_rmse

        # Features to analyze
        features = ["signal_to_noise", "seq_length"]

        # Add derived features
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda x: (x.count("G") + x.count("C")) / len(x)
        )
        val_df["A_content"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))

        analysis_features = ["signal_to_noise", "gc_content", "A_content"]

        print("Correlation between Error (RMSE) and Input Features:")
        for feat in analysis_features:
            if feat in val_df.columns:
                corr, _ = stats.pearsonr(val_df[feat], val_df["error_rmse"])
                print(f"  {feat}: {corr:.4f}")

    # 5. Submission
    threshold = 0.6795554161071777
    if mcrmse < threshold:
        print(
            f"\nValidation metric ({mcrmse}) meets threshold ({threshold}). Generating submission..."
        )
        run_inference(
            model_path=Config.MODEL_PATH,
            output_path=Config.SUBMISSION_FILE,
            batch_size=Config.BATCH_SIZE,
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation metric ({mcrmse}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
