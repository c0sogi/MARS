import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library modules
import library.config
import library.utils
import library.data_processing
import library.model
import library.trainer
import library.inference


# ==========================================
# 0. Suppress Progress Bars (Monkey Patch)
# ==========================================
# The requirements state "Do not print progress bars".
# Since we cannot modify the library files, we monkey-patch the tqdm object
# used in the trainer and inference modules to be a silent pass-through.
class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


def silent_tqdm(iterable, *args, **kwargs):
    return SilentTqdm(iterable)


library.trainer.tqdm = silent_tqdm
library.inference.tqdm = silent_tqdm

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Starting Demonstration Script...")

    # 1. Configuration Overrides for Speed/Demo
    # We modify the Config class directly to control the execution flow
    # without changing the source file.
    Config = library.config.Config
    Config.DEBUG = True  # Use small subset (100 breaths)
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.USE_CACHE = False  # Force processing to demonstrate pipeline
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Sync Config with Demo Model Architecture
    Config.LSTM_HIDDEN_DIM = 128
    Config.PROJECTION_DIM = 64
    Config.LSTM_LAYERS = 2

    # Ensure working directory is clean/ready
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    library.utils.seed_everything(Config.SEED)
    device = library.utils.get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Pipeline Demonstration
    # ==========================================
    print("\n--- 1. Data Pipeline Verification ---")

    # Generate dataloaders
    train_loader, val_loader, test_loader = library.data_processing.prepare_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Fetch a batch to verify shapes
    batch_x, batch_y = next(iter(train_loader))

    print(f"Batch X shape: {batch_x.shape}")  # Expected: (Batch, 80, Features)
    print(f"Batch Y shape: {batch_y.shape}")  # Expected: (Batch, 80)

    # Assertions
    assert batch_x.dim() == 3, "Input should be 3-dimensional (Batch, Seq, Feat)"
    assert batch_y.dim() == 2, "Target should be 2-dimensional (Batch, Seq)"
    assert batch_x.shape[1] == 80, "Sequence length should be 80"
    assert batch_x.shape[2] == len(Config.SELECTED_FEATURES), "Feature count mismatch"

    # Check for presence of u_out (needed for loss)
    if "u_out" in Config.SELECTED_FEATURES:
        u_out_idx = Config.SELECTED_FEATURES.index("u_out")
        u_out_sample = batch_x[0, :, u_out_idx]
        assert torch.all(
            torch.isin(u_out_sample, torch.tensor([0.0, 1.0]))
        ), "u_out should be binary (0 or 1)"
        print("Data shapes and feature integrity verified.")
    else:
        raise AssertionError("'u_out' missing from selected features!")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n--- 2. Model Architecture Verification ---")

    model = library.model.SC_GI_BiLSTM(
        input_dim=len(Config.SELECTED_FEATURES),
        lstm_hidden_dim=128,  # Reduced for demo speed
        projection_dim=64,
        lstm_layers=2,
        dropout=0.0,
        bidirectional=True,
    ).to(device)

    # Move batch to device
    batch_x = batch_x.to(device)

    # Forward pass
    output = model(batch_x)

    print(f"Model Output shape: {output.shape}")

    # Assertions
    assert (
        output.shape == batch_y.shape
    ), f"Output shape {output.shape} mismatch with target {batch_y.shape}"
    assert output.requires_grad, "Output should track gradients for training"
    assert not torch.isnan(output).any(), "Model produced NaN values"
    print("Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n--- 3. Training Loop Demonstration ---")

    # Setup components
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = library.trainer.WeightedL1Loss()

    trainer = library.trainer.Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    # Run training
    # This will run for Config.EPOCHS (2)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully saved at: {checkpoint_path}")
    else:
        raise FileNotFoundError("Training did not produce a model checkpoint.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n--- 4. Inference and Submission Generation ---")

    # We use the generate_submission function which handles loading, prediction, and formatting
    # We must pass debug=True because we trained on the debug subset and the model
    # expects data/metadata alignment consistent with that subset.

    library.inference.generate_submission(
        model_path=checkpoint_path, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at: {submission_path}")
        print(f"Submission shape: {df_sub.shape}")
        print(f"First few rows:\n{df_sub.head()}")

        # Validation
        assert list(df_sub.columns) == [
            "id",
            "pressure",
        ], "Incorrect submission columns"
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"
        assert len(df_sub) > 0, "Submission is empty"

        # In debug mode, we expect 100 breaths * 80 steps = 8000 rows (approx)
        # The exact number depends on the 'test_metadata.csv' subset logic in prepare_dataloaders
        # But we just ensure it generated something reasonable.
        print("Submission verification successful.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemonstration completed successfully.")
