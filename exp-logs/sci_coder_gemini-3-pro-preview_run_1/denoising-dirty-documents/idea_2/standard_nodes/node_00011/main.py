import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ResUNet
from library.train import train_one_epoch, validate, predict_tensor
from library.inference import generate_submission
from library.utils import seed_everything, pad_image, unpad_image, calculate_rmse


def run_pipeline():
    # --- 1. Configuration & Setup ---
    # Override Config for optimized execution
    # Extended training for convergence (Cite solution_lesson_node_00003)
    Config.NUM_EPOCHS = 400
    # Decoupled scheduler horizon (Cite solution_lesson_node_00007)
    Config.T_MAX = 1500

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Setup computation device
    device = torch.device(Config.DEVICE)

    # --- 2. Data Loading ---
    # Load datasets
    train_loader, val_loader, _ = get_dataloaders()

    # --- 3. Model Initialization ---
    model = ResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
    ).to(device)

    # Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )
    criterion = nn.MSELoss()

    # --- 4. Training Loop ---
    best_rmse = float("inf")

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT), exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        # Train one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    # --- 5. Final Validation Assessment ---
    # Load the best model weights
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    # Calculate final validation metric
    final_val_rmse = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_rmse}")

    # --- 6. Failure Analysis ---
    print("Performing failure analysis...")
    model.eval()

    analysis_results = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            # Prepare data
            noisy_np = noisy.squeeze().numpy()
            clean_np = clean.squeeze().numpy()

            # Pad image for inference
            padded_noisy = pad_image(noisy_np, factor=32)
            input_tensor = (
                torch.from_numpy(padded_noisy).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict (disable TTA for consistent analysis)
            output_tensor = predict_tensor(model, input_tensor, tta=False)

            # Post-process
            output_np = output_tensor.squeeze().cpu().numpy()
            output_clean = unpad_image(output_np, noisy_np.shape)
            output_clean = np.clip(output_clean, 0, 1)

            # Calculate metrics and features
            error = calculate_rmse(clean_np, output_clean)
            mean_intensity = np.mean(noisy_np)
            std_intensity = np.std(noisy_np)

            analysis_results.append(
                {
                    "error": error,
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity,
                }
            )

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(analysis_results)

    # Calculate correlations
    corr_mean = df_analysis["error"].corr(df_analysis["mean_intensity"])
    corr_std = df_analysis["error"].corr(df_analysis["std_intensity"])

    print(f"Correlation (Error vs Mean Intensity): {corr_mean}")
    print(f"Correlation (Error vs Std Intensity): {corr_std}")

    # --- 7. Submission Generation ---
    THRESHOLD = 0.015272615302544418

    if final_val_rmse < THRESHOLD:
        generate_submission(
            checkpoint_path=Config.MODEL_CHECKPOINT,
            output_file=Config.SUBMISSION_FILE,
            device=Config.DEVICE,
            use_tta=Config.TTA_ENABLED,
        )


if __name__ == "__main__":
    run_pipeline()
