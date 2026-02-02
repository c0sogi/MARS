import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
from library import config, utils, dataset, model, trainer


def main():
    print("Starting Library Demonstration...")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")
    config.SEED = 42
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 16  # Use only 16 samples
    config.BATCH_SIZE = 4
    config.EPOCHS = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.PRETRAINED = False  # Skip downloading weights for speed

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure reproducibility
    utils.set_seed(config.SEED)

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("\n[2] Testing Data Loading Pipeline...")

    # Generate/Load Folds
    # We force regeneration to demonstrate the logic, though it usually caches
    df_folds = dataset.get_iterative_folds(load_cached_data=False)
    assert "fold" in df_folds.columns, "Folds dataframe missing 'fold' column"
    print(f"Folds generated. Total samples: {len(df_folds)}")

    # Create DataLoaders for Fold 0
    train_loader, val_loader = dataset.get_dataloaders(
        fold_id=0, df_folds=df_folds, batch_size=config.BATCH_SIZE, debug=config.DEBUG
    )

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Validation: Check Input Dimensions
    # Expected: (Batch, 3, 224, 512) -> 3 channels due to Delta computation
    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_HEIGHT,
        config.IMG_WIDTH,
    ), f"Incorrect image shape: {images.shape}"

    # Validation: Check Label Dimensions
    # Expected: (Batch, 19)
    assert labels.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Incorrect label shape: {labels.shape}"

    print("Data loading and processing verified.")

    # ==========================================
    # 3. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[3] Testing Model Architecture...")

    net = model.BirdResNet(pretrained=config.PRETRAINED)
    net.to(device)

    # Move batch to device
    images = images.to(device)
    labels = labels.to(device)

    # Test Training Forward Pass (Multi-Sample Dropout enabled)
    net.train()
    outputs_train = net(images)
    print(f"Training Output Shape: {outputs_train.shape}")

    # Expected: (Batch, Num_Drops, Num_Classes)
    expected_train_shape = (
        config.BATCH_SIZE,
        len(config.DROPOUT_RATES),
        config.NUM_CLASSES,
    )
    assert (
        outputs_train.shape == expected_train_shape
    ), f"Incorrect train output shape. Expected {expected_train_shape}, got {outputs_train.shape}"

    # Test Evaluation Forward Pass (Averaged Logits)
    net.eval()
    with torch.no_grad():
        outputs_eval = net(images)
    print(f"Evaluation Output Shape: {outputs_eval.shape}")

    # Expected: (Batch, Num_Classes)
    expected_eval_shape = (config.BATCH_SIZE, config.NUM_CLASSES)
    assert (
        outputs_eval.shape == expected_eval_shape
    ), f"Incorrect eval output shape. Expected {expected_eval_shape}, got {outputs_eval.shape}"

    print("Model architecture verified.")

    # ==========================================
    # 4. Loss Function & Mixup Logic
    # ==========================================
    print("\n[4] Testing Loss Calculation and Mixup...")

    # Calculate positive weights for imbalance handling
    # We use the small debug subset dataframe for this calculation
    df_train_subset = df_folds[df_folds["fold"] != 0].iloc[: config.DEBUG_SUBSET_SIZE]
    label_cols = [c for c in df_train_subset.columns if c.startswith("species_")]
    pos_weights = utils.calculate_pos_weights(df_train_subset, label_cols).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Simulate Mixup
    net.train()
    mixed_images, y_a, y_b, lam = trainer.mixup_data(images, labels, alpha=0.4)

    # Forward pass with mixed images
    outputs_mixed = net(mixed_images)

    # Calculate Mixup Loss
    loss = trainer.mixup_criterion(criterion, outputs_mixed, y_a, y_b, lam)

    print(f"Calculated Mixup Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("Loss and Mixup logic verified.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)

    # Run training for one epoch
    train_loss = trainer.train_one_epoch(
        model=net,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=0,
    )

    print(f"Epoch 0 Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_auc = trainer.validate(
        model=net, loader=val_loader, criterion=criterion, device=device
    )

    print(f"Epoch 0 Val Loss: {val_loss:.4f}")
    print(f"Epoch 0 Val AUC: {val_auc:.4f}")

    # Check that metrics are valid numbers
    assert not np.isnan(train_loss), "Train loss is NaN"
    assert not np.isnan(val_loss), "Val loss is NaN"
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"

    print("Training loop verified.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Testing Inference Pipeline...")

    # Get Test Loader
    test_loader = dataset.get_test_dataloader(batch_size=config.BATCH_SIZE)

    # Run Inference on a few batches (simulated by breaking early)
    net.eval()
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

            # Limit inference for demo speed
            if i >= 2:
                break

    predictions = np.concatenate(all_preds, axis=0)
    print(f"Inference output shape: {predictions.shape}")

    assert (
        predictions.shape[1] == config.NUM_CLASSES
    ), f"Prediction columns mismatch. Expected {config.NUM_CLASSES}, got {predictions.shape[1]}"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions must be probabilities between 0 and 1"

    print("Inference pipeline verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
