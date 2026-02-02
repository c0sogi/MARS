import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, MetricMonitor
from library.dataset import get_dataloaders, WhaleDataset
from library.model import WhaleEfficientNet
from library.train import (
    mixup_data,
    mixup_criterion,
    train_one_epoch,
    validate_one_epoch,
    run_training,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring Environment for Fast Execution...")

    # Modify Config attributes to run a fast debug session
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 60  # Small subset for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Validate Utilities
    # --------------------------------------------------------------------------
    print("\n[2] Validating Utilities...")

    # Test MetricMonitor
    monitor = MetricMonitor()
    monitor.update("loss", 0.5, count=1)
    monitor.update("loss", 1.5, count=1)
    avg_loss = monitor.get_avg("loss")
    assert avg_loss == 1.0, f"MetricMonitor failed: expected 1.0, got {avg_loss}"
    print("MetricMonitor: OK")

    # Test AUC Calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_pred)
    assert 0.0 <= auc <= 1.0, "AUC calculation returned invalid range"
    print(f"AUC Calculation: OK (Score: {auc:.4f})")

    # --------------------------------------------------------------------------
    # 3. Validate Dataset & DataLoaders
    # --------------------------------------------------------------------------
    print("\n[3] Validating Dataset and DataLoaders...")

    # Generate DataLoaders (this will trigger data processing and caching)
    # forcing load_cached_data=False to demonstrate processing logic at least once,
    # though subsequent runs in the pipeline might use cache.
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=False
    )

    # Fetch a single batch from training loader
    data_batch, label_batch, clip_batch = next(iter(train_loader))

    # Verify Shapes
    # Expected Data Shape: (Batch, 1, n_mels, time_steps)
    # Time steps depends on duration (2.0s) and hop length. ~251 frames.
    print(f"Data Batch Shape: {data_batch.shape}")
    print(f"Label Batch Shape: {label_batch.shape}")

    assert data_batch.ndim == 4, "Data batch should be 4-dimensional (B, C, F, T)"
    assert data_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert data_batch.shape[1] == 1, "Expected 1 channel (spectrogram)"
    assert label_batch.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("Data Loading: OK")

    # --------------------------------------------------------------------------
    # 4. Validate Model Architecture
    # --------------------------------------------------------------------------
    print("\n[4] Validating Model Architecture...")

    # Instantiate Model
    # We use pretrained=False here to avoid downloading weights during this quick check,
    # though the main training loop uses the default (True).
    model = WhaleEfficientNet(Config, pretrained=False)
    model.to(Config.DEVICE)

    # Move batch to device
    data_batch = data_batch.to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        output = model(data_batch)

    print(f"Model Output Shape: {output.shape}")

    # Verify Output Shape: (Batch, 1) -> Binary classification logits
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("Model Architecture: OK")

    # --------------------------------------------------------------------------
    # 5. Validate Training Components (Mixup & Step)
    # --------------------------------------------------------------------------
    print("\n[5] Validating Training Logic (Mixup & Step)...")

    # Prepare inputs for Mixup
    targets = label_batch.to(Config.DEVICE).view(-1, 1)

    # Test Mixup Data
    mixed_data, target_a, target_b, lam = mixup_data(
        data_batch, targets, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE
    )

    assert mixed_data.shape == data_batch.shape, "Mixup altered input shape"
    assert target_a.shape == targets.shape, "Mixup target shape mismatch"
    print(f"Mixup Lambda: {lam:.4f}")

    # Test Loss Calculation
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simulate one training step
    model.train()
    optimizer.zero_grad()
    output = model(mixed_data)
    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    loss.backward()
    optimizer.step()

    print(f"Single Step Loss: {loss.item():.4f}")
    assert not np.isnan(loss.item()), "Loss is NaN"
    print("Training Step Logic: OK")

    # --------------------------------------------------------------------------
    # 6. Execute Full Pipeline
    # --------------------------------------------------------------------------
    print("\n[6] Executing Full Training Pipeline (run_training)...")
    print("    This will run 1 epoch on the debug subset and generate a submission.")

    # Run the provided training entry point
    # This uses the global Config we modified at the start
    run_training()

    # --------------------------------------------------------------------------
    # 7. Verify Submission
    # --------------------------------------------------------------------------
    print("\n[7] Verifying Submission...")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {sub_df.shape}")
        print("Head:")
        print(sub_df.head())

        # Basic checks
        assert "clip" in sub_df.columns, "Submission missing 'clip' column"
        assert (
            "probability" in sub_df.columns
        ), "Submission missing 'probability' column"
        assert len(sub_df) > 0, "Submission file is empty"
        assert (
            sub_df["probability"].dtype == float
            or sub_df["probability"].dtype == np.float64
        ), "Probability column is not float"

        print("Submission Verification: OK")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
