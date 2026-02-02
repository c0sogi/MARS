import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Ensure library is in path
sys.path.append(".")

from library import config, utils, model, dataset, data_processing, train, inference


def main():
    # 1. Setup
    utils.set_seed()
    device = config.get_device()
    print(f"Running on device: {device}")

    # 2. Data Processing
    engineer = data_processing.FeatureEngineer()

    print("Loading Training Data...")
    # Load cached if available, else process
    X_train, y_train, train_ids = engineer.process_dataset(
        split="train", load_cached_data=True
    )

    # Subsampling for speed (Requirement: Limit max samples for fast baseline)
    MAX_TRAIN_SAMPLES = 500000
    if len(y_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(y_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        # Use fixed seed for subsampling
        np.random.seed(config.SEED)
        indices = np.random.choice(len(y_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_ids = train_ids[indices]

    print("Loading Validation Data...")
    X_val, y_val, val_ids = engineer.process_dataset(
        split="validation", load_cached_data=True
    )

    # 3. Datasets & Loaders
    train_dataset = dataset.ContactSequenceDataset(X_train, y_train)
    val_dataset = dataset.ContactSequenceDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 4. Model & Training
    net = model.KCAN().to(device)

    # Initialize Trainer
    trainer = train.Trainer(net, train_loader, val_loader, device)

    # Fit
    print("Starting Training...")
    trainer.fit(epochs=config.EPOCHS, patience=config.PATIENCE)

    # 5. Final Validation & Threshold Optimization
    print("Loading best model for evaluation...")
    net.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    net.eval()

    # Inference on Validation set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            sequence, center_features = inputs
            sequence = sequence.to(device)
            center_features = center_features.to(device)

            logits = net((sequence, center_features))
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_probs = np.concatenate(val_probs).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    # Optimize Threshold
    best_thresh, best_mcc = trainer.optimize_threshold(val_targets, val_probs)

    # Save best threshold
    np.save(
        os.path.join(config.WORKING_DIR, "best_threshold.npy"), np.array([best_thresh])
    )

    # 6. Print Metric (Required Format)
    print(f"Final Validation Metric: {best_mcc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Reshape X_val to get center features (t=0)
    # X_val shape: (N, Window * Features)
    num_features = len(config.INPUT_FEATURES)
    window_size = config.WINDOW_SIZE
    center_idx = window_size // 2

    X_val_reshaped = X_val.reshape(-1, window_size, num_features)
    center_features_val = X_val_reshaped[:, center_idx, :]

    feature_names = config.INPUT_FEATURES
    correlations = {}

    for i, name in enumerate(feature_names):
        feat_vals = center_features_val[:, i]
        # Calculate correlation with error
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        else:
            corr = 0.0
        correlations[name] = corr

    print("Correlation between Error Magnitude and Center Frame Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"{name}: {corr:.4f}")

    # 8. Submission
    TARGET_METRIC = 0.62458462731896
    if best_mcc > TARGET_METRIC:
        print(
            f"\nValidation MCC ({best_mcc}) > Target ({TARGET_METRIC}). Generating submission..."
        )
        pipeline = inference.InferencePipeline()
        pipeline.run(load_cached=True)
    else:
        print(
            f"\nValidation MCC ({best_mcc}) <= Target ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
