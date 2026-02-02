import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import DataProcessor
from library.train_eval import Trainer


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    seed_everything(Config.SEED)

    # Override Config for fast execution
    Config.NUM_EPOCHS = 5  # Reduce epochs for speed
    TRAIN_SAMPLE_SIZE = 500000  # Limit training samples
    SUBMISSION_THRESHOLD = 0.62458462731896

    print("Initializing EC-PIRN Pipeline...")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading and Processing
    processor = DataProcessor()

    # Load Train/Val data (uses cache if available)
    print("Loading training and validation data...")
    X_train, y_train, X_val, y_val = processor.get_train_val_data(load_cached_data=True)

    # Subsample training data for fast baseline
    if len(X_train) > TRAIN_SAMPLE_SIZE:
        print(
            f"Subsampling training data from {len(X_train)} to {TRAIN_SAMPLE_SIZE}..."
        )
        indices = np.random.choice(len(X_train), TRAIN_SAMPLE_SIZE, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Training
    print("Starting Model Training...")
    trainer = Trainer(device=Config.DEVICE)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 4. Final Validation Assessment
    print("Performing Final Validation...")
    # trainer.fit loads the best model state automatically at the end
    val_loss, val_targets, val_logits = trainer.validate(val_loader)

    # Optimize threshold one last time on full validation set
    best_th, best_mcc = trainer.optimize_threshold(val_targets, val_logits)
    trainer.best_threshold = best_th  # Update trainer with best threshold

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate probabilities
    val_probs = 1 / (1 + np.exp(-val_logits))
    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    # Calculate correlation between features and error
    feature_names = Config.get_feature_names()
    print(f"Analyzing correlations for {len(feature_names)} features...")

    # We do this in chunks or just simple loop if memory allows (X_val fits in memory)
    # Using a simple correlation calculation: Corr(X, Y) = Cov(X, Y) / (Std(X) * Std(Y))
    # We'll use numpy for speed

    correlations = []
    # Normalize errors for correlation
    err_mean = np.mean(errors)
    err_std = np.std(errors)

    # Avoid division by zero
    if err_std > 1e-9:
        normalized_error = (errors - err_mean) / err_std

        # Iterate through features
        for i, feat_name in enumerate(feature_names):
            feat_col = X_val[:, i]
            feat_mean = np.mean(feat_col)
            feat_std = np.std(feat_col)

            if feat_std > 1e-9:
                normalized_feat = (feat_col - feat_mean) / feat_std
                corr = np.mean(normalized_feat * normalized_error)
                correlations.append((feat_name, corr))
            else:
                correlations.append((feat_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features associated with Error:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    if best_mcc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation MCC ({best_mcc}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test, df_test_meta = processor.get_test_data(load_cached_data=True)

        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict
        predictions = trainer.predict(test_loader)

        # Save
        df_submission = pd.DataFrame(
            {"contact_id": df_test_meta["contact_id"], "contact": predictions}
        )

        save_path = Config.SUBMISSION_PATH
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(df_submission.head())

    else:
        print(
            f"\nValidation MCC ({best_mcc}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
