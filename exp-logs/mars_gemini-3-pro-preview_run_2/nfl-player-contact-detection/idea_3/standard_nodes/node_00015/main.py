import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.model import TRGCN
from library.data_processor import DataProcessor, NFLContactDataset
from library.trainer import Trainer
from library.inference import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading & Processing
    dp = DataProcessor()

    # Load Training Data
    # Utilizing cached data if available for speed
    X_train, y_train, _ = dp.process_data("train", load_cached_data=True)

    # Subsample training data for fast baseline execution
    # The full dataset is large (~3.4M rows). We limit to 500k for this run
    # to ensure it completes well within the time limit while still learning.
    MAX_TRAIN_SAMPLES = 500000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        # Random sampling
        indices = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    train_dataset = NFLContactDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Validation Data
    # We use the full validation set to ensure the metric is accurate
    X_val, y_val, val_ids = dp.process_data("validation", load_cached_data=True)
    val_dataset = NFLContactDataset(X_val, y_val, contact_ids=val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = TRGCN(
        input_dim=Config.NUM_FEATURES_PER_TIMESTEP, window_size=Config.WINDOW_SIZE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training
    trainer = Trainer(model, optimizer, device=device)

    # Train for limited epochs for fast baseline
    # Trainer handles saving the best model to Config.CACHE_DIR/best_model.pth
    trainer.fit(train_loader, val_loader, epochs=5, patience=3)

    # 5. Validation Assessment
    # Load the best weights found during training
    inference = InferenceEngine(model, device=device)
    inference.load_weights()

    # Generate predictions on validation set
    model.eval()
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            val_probs.append(out.cpu().numpy())
            val_targets.append(batch_y.numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Apply the optimized threshold found by the Trainer
    best_threshold = trainer.best_threshold
    val_preds = (val_probs >= best_threshold).astype(int)

    # Compute and Print Final Metric
    final_mcc = compute_mcc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error (0 = correct, 1 = incorrect)
    errors = np.abs(val_targets - val_preds)

    # Correlate errors with input features
    # We examine the features at the center of the time window
    center_idx = Config.WINDOW_SIZE // 2
    center_features = X_val[:, center_idx, :]

    # Map feature indices based on DataProcessor structure:
    # [P1(7), P2(7), dist(1), log_dist(1), closing_speed(1), is_ground(1)]
    # Indices:
    # distance: 14 (-4)
    # closing_speed: 16 (-2)
    # is_ground: 17 (-1)
    # speed_1: 2
    # speed_2: 9 (7+2)

    features_to_analyze = {
        "distance": -4,
        "closing_speed": -2,
        "is_ground": -1,
        "speed_p1": 2,
        "speed_p2": 9,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, idx in features_to_analyze.items():
        feat_values = center_features[:, idx]
        # Calculate Point-Biserial Correlation
        if np.std(feat_values) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: NaN (No variance)")

    # 7. Submission Generation
    TARGET_SCORE = 0.62458462731896

    if final_mcc > TARGET_SCORE:
        print(
            f"\nValidation metric {final_mcc} exceeds target {TARGET_SCORE}. Generating submission..."
        )

        # Load Test Data
        X_test, _, test_ids = dp.process_data("test", load_cached_data=True)

        test_dataset = NFLContactDataset(X_test, contact_ids=test_ids)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate Submission using the best threshold
        inference.predict_test(
            test_loader, best_threshold, output_path=Config.SUBMISSION_PATH
        )

    else:
        print(
            f"\nValidation metric {final_mcc} does not exceed target {TARGET_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
