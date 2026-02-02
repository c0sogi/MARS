import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from library
from library.config import Config
from library.utils import set_seed
from library.data import get_loaders
from library.model import GatedDenseNet
from library.engine import fit, validate, predict, generate_submission_csv


def main():
    # 1. Configuration & Setup
    # Override Config for fast execution within limits
    Config.EPOCHS = 20
    Config.setup_directories()
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = GatedDenseNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 5. Training
    print("Starting training...")
    best_mcrmse = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_PATH,
    )

    # 6. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Calculate final validation metric
    _, final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get predictions and targets for validation set
    val_preds, val_ids = predict(model, val_loader, device)

    # Load validation metadata to get ground truth and features
    val_df = pd.read_csv(Config.VAL_CSV)
    # Ensure order matches the predictions
    val_df = val_df.set_index("id").loc[val_ids].reset_index()

    # Extract targets from loader to ensure alignment with preprocessing
    all_targets = []
    for batch in val_loader:
        all_targets.append(batch[2].numpy())
    val_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE per sample
    # Scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice to pred_len (68) and select scored columns
    preds_sliced = val_preds[:, : Config.PRED_LEN, scored_indices]
    targets_sliced = val_targets[:, : Config.PRED_LEN, scored_indices]

    # MSE per sample: mean over (length, channels)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Add to dataframe
    val_df["error"] = rmse_per_sample

    # Features to correlate
    # Signal to noise
    if "signal_to_noise" in val_df.columns:
        corr_sn = val_df["error"].corr(val_df["signal_to_noise"])
        print(f"Correlation between Error and Signal_to_Noise: {corr_sn}")

    # Base counts
    val_df["count_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
    val_df["count_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
    val_df["count_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
    val_df["count_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

    print(
        f"Correlation between Error and Count A: {val_df['error'].corr(val_df['count_A'])}"
    )
    print(
        f"Correlation between Error and Count G: {val_df['error'].corr(val_df['count_G'])}"
    )
    print(
        f"Correlation between Error and Count C: {val_df['error'].corr(val_df['count_C'])}"
    )
    print(
        f"Correlation between Error and Count U: {val_df['error'].corr(val_df['count_U'])}"
    )

    # 8. Submission
    THRESHOLD = 0.5417620723771521
    if final_val_metric < THRESHOLD:
        print(f"\nMetric {final_val_metric} < {THRESHOLD}. Generating submission...")
        test_preds, test_ids = predict(model, test_loader, device)

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission_csv(test_preds, test_ids, submission_path)
    else:
        print(
            f"\nMetric {final_val_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
