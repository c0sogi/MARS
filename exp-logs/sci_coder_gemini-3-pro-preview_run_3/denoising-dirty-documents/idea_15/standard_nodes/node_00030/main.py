import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.trainer import ModelTrainer
from library.architecture import ResDnCNN
from library.inference import generate_submission


def run_pipeline():
    """
    Main orchestration function for the High-Capacity Zero-Initialized Deep Residual Ensemble.
    Adapts configuration for a fast baseline execution within the time limit.
    """

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override default Config settings to ensure execution completes within ~29 minutes.
    # We scale down the model and data density while preserving the architectural logic.
    print("Configuring parameters for fast baseline execution...")

    Config.EPOCHS = 5  # Limit epochs to ensure convergence in time
    Config.NUM_ENSEMBLE_MODELS = 1  # Train single model instead of full ensemble
    Config.MODEL_DEPTH = 10  # Reduce depth (from 30) for speed
    Config.MODEL_FILTERS = 64  # Reduce width (from 96) for speed
    Config.STRIDE = 20  # Increase stride to reduce total patch count
    Config.BATCH_SIZE = 256  # Increase batch size for throughput
    Config.PATIENCE = 3  # Aggressive early stopping

    # Use unique cache filenames to prevent conflicts with other runs
    Config.CACHE_TRAIN_PATCHES = "train_patches_fast_run.npy"
    Config.CACHE_TRAIN_TARGETS = "train_targets_fast_run.npy"
    Config.CACHE_VAL_PATCHES = "val_patches_fast_run.npy"
    Config.CACHE_VAL_TARGETS = "val_targets_fast_run.npy"

    seed_everything(Config.SEED)

    print(
        f"Settings: Depth={Config.MODEL_DEPTH}, Filters={Config.MODEL_FILTERS}, Stride={Config.STRIDE}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    # load_cached_data=True allows reusing processed arrays if they exist
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Starting training phase...")

    # Train the specified number of models (1 in this fast baseline)
    for i in range(Config.NUM_ENSEMBLE_MODELS):
        print(f"--- Training Model {i} ---")

        # Instantiate model with overridden config
        model = ResDnCNN(
            depth=Config.MODEL_DEPTH,
            filters=Config.MODEL_FILTERS,
            input_channels=Config.INPUT_CHANNELS,
            output_channels=Config.OUTPUT_CHANNELS,
        )

        # Initialize trainer
        trainer = ModelTrainer(model, device=Config.DEVICE)

        # Execute training loop
        best_loss = trainer.train(train_loader, val_loader, model_id=i)
        print(f"Model {i} training complete. Best Val Loss: {best_loss:.6f}")

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("Starting validation and failure analysis...")

    # Load the best checkpoint of the trained model
    model = ResDnCNN(
        depth=Config.MODEL_DEPTH,
        filters=Config.MODEL_FILTERS,
        input_channels=Config.INPUT_CHANNELS,
        output_channels=Config.OUTPUT_CHANNELS,
    )

    model_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    if not os.path.exists(model_path):
        print("Error: Model checkpoint not found. Aborting.")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # Metrics containers
    total_sse = 0.0
    total_pixels = 0

    # Analysis containers (subsampling for memory efficiency)
    sampled_errors = []
    sampled_inputs = []
    MAX_SAMPLES = 20000

    with torch.no_grad():
        for noisy_imgs, residual_targets in val_loader:
            noisy_imgs = noisy_imgs.to(Config.DEVICE)
            residual_targets = residual_targets.to(Config.DEVICE)

            # Forward pass
            preds = model(noisy_imgs)

            # Calculate residuals
            diff = preds - residual_targets

            # Accumulate SSE for RMSE calculation
            total_sse += torch.sum(diff**2).item()
            total_pixels += diff.numel()

            # Collect samples for failure analysis
            if len(sampled_errors) < MAX_SAMPLES:
                # Flatten and move to CPU
                batch_errors = torch.abs(diff).cpu().numpy().flatten()
                batch_inputs = noisy_imgs.cpu().numpy().flatten()

                # Take slice to fit buffer
                limit = min(MAX_SAMPLES - len(sampled_errors), len(batch_errors))
                sampled_errors.extend(batch_errors[:limit])
                sampled_inputs.extend(batch_inputs[:limit])

    # Calculate Final RMSE
    mse = total_sse / total_pixels if total_pixels > 0 else 0.0
    rmse = np.sqrt(mse)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {rmse}")

    # Failure Analysis: Correlation
    if len(sampled_errors) > 100:
        corr, _ = pearsonr(sampled_inputs, sampled_errors)
        print(f"Correlation between Input Intensity and Error Magnitude: {corr}")
    else:
        print("Insufficient samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in task requirements
    THRESHOLD = 0.011577641381826402

    if rmse < THRESHOLD:
        print(
            f"Validation metric {rmse} meets threshold {THRESHOLD}. Generating submission..."
        )
        # generate_submission uses Config to load models, so it will respect the overrides
        generate_submission()
    else:
        print(
            f"Validation metric {rmse} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
