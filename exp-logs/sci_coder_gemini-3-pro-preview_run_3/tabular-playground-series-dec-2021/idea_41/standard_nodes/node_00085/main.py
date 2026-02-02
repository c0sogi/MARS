import sys
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.data_utils import load_data, CoverTypeDataset
from library.model import ParallelDCNResNet
from library.train_eval import train_one_epoch, evaluate, predict, set_seed


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for fast baseline execution within 1 hour
    # 8 epochs on the full dataset strikes a balance between speed and performance
    Config.EPOCHS = 8

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device(Config.DEVICE)
    # Enable CuDNN benchmark for faster training on fixed input sizes
    if not Config.CUDNN_DETERMINISTIC:
        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load data (cached if available)
    train_X, train_y, val_X, val_y, test_X, test_ids = load_data(load_cached_data=True)

    # Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X, None)

    # Create DataLoaders
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

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = ParallelDCNResNet(
        input_dim=train_X.shape[1], num_classes=Config.NUM_CLASSES
    )
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_acc)

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, Config.MODEL_SAVE_PATH)

    # --------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # --------------------------------------------------------------------------
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on full validation set
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    # REQUIRED PRINT: Final Validation Metric
    print(f"Final Validation Metric: {final_val_acc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Error Vector (1 for error, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    # Calculate Correlation with Input Features
    correlations = []
    n_features = val_X.shape[1]

    for i in range(n_features):
        feature_col = val_X[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, errors)[0, 1]

        correlations.append((i, corr))

    # Sort by absolute correlation descending
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features Correlated with Error:")
    for idx, corr in correlations[:10]:
        print(f"Feature {idx}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9626291666666666

    if final_val_acc > THRESHOLD:
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        raw_preds = predict(model, test_loader, device)
        final_preds = [Config.INVERSE_CLASS_MAPPING[p] for p in raw_preds]

        submission = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: final_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
