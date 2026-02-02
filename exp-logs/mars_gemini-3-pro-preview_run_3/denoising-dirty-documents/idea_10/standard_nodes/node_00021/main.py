import torch
import numpy as np
import os
import sys
from library import config, utils, network, data_loader, train, predict


def main():
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Train Model (Fast Baseline)
    # -------------------------------------------------------------------------
    # We limit epochs to 15 to ensure the run completes quickly (< 2 hours).
    # We use the full dataset (max_samples=None) as the A100 can handle it efficiently.
    print("--- Starting Training Pipeline ---")
    train.train_model(
        num_epochs=15,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        load_cached_data=True,
        max_samples=None,
    )

    # -------------------------------------------------------------------------
    # 2. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device(config.DEVICE)

    # Initialize the model architecture
    model = network.SE_ZI_ResDnCNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_blocks=config.NUM_BLOCKS,
        kernel_size=config.KERNEL_SIZE,
        padding=config.PADDING,
        use_se=config.USE_SE,
        se_reduction=config.SE_REDUCTION,
        zero_init_residual=config.ZERO_INIT_RESIDUAL,
    ).to(device)

    # Load the best checkpoint
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Checkpoint not found at {config.BEST_MODEL_PATH}")
        sys.exit(1)

    utils.load_checkpoint(config.BEST_MODEL_PATH, model, device=device)
    model.eval()

    # Load validation data
    _, _, val_patches, val_targets = data_loader.prepare_data(load_cached_data=True)

    # Create DataLoader
    val_dataset = data_loader.DenoisingDataset(val_patches, val_targets, augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    all_preds = []
    all_targets = []
    all_inputs = []

    # Inference Loop (Optimized with no_grad)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predict noise residual
            outputs = model(inputs)

            # Collect results (move to CPU to avoid OOM on large validation sets)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_inputs = np.concatenate(all_inputs)

    # Calculate RMSE
    # Note: RMSE of (Noise_Pred, Noise_True) is equivalent to RMSE of (Clean_Pred, Clean_True)
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)

    print(f"Final Validation Metric: {rmse}")

    # Failure Analysis
    # Calculate correlation between Input Intensity and Error Magnitude
    errors = np.abs(all_preds - all_targets).flatten()
    input_intensities = all_inputs.flatten()

    # Downsample for correlation calculation if dataset is huge (optional optimization)
    if len(errors) > 1_000_000:
        indices = np.random.choice(len(errors), 1_000_000, replace=False)
        errors_sample = errors[indices]
        inputs_sample = input_intensities[indices]
    else:
        errors_sample = errors
        inputs_sample = input_intensities

    # Compute Pearson Correlation
    corr_matrix = np.corrcoef(inputs_sample, errors_sample)
    corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0

    print(f"Correlation between Input Intensity and Error Magnitude: {corr:.8f}")

    # -------------------------------------------------------------------------
    # 3. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.011577641381826402

    if rmse < threshold:
        print(f"\nMetric {rmse} meets threshold {threshold}. Generating submission...")
        predict.generate_submission(
            model_path=config.BEST_MODEL_PATH, output_path=config.SUBMISSION_FILE
        )
    else:
        print(
            f"\nMetric {rmse} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
