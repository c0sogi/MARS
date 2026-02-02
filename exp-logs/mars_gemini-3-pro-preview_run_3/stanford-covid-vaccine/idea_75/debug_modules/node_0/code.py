import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders, RNADataset
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("==== RNA Degradation Prediction Pipeline Demonstration ====")

    # 1. Configuration and Seeding
    # We use debug=True to reduce epochs (2) and use a small data subset (32 samples)
    # for quick demonstration and verification.
    print("\n[Step 1] Initializing Configuration...")
    config = Config(debug=True)
    seed_everything(config.seed)

    print(f"Device: {config.device}")
    print(f"Debug Mode: True (Epochs: {config.epochs})")
    print(f"Working Directory: {config.working_dir}")

    # 2. Data Loading
    print("\n[Step 2] Loading Data (Debug Subset)...")
    # get_dataloaders handles loading from cache or parquet and slicing for debug
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=True
    )

    # Fetch a single batch to verify shapes
    features, pair_indices, targets = next(iter(train_loader))

    # Expected shapes:
    # Features: (Batch, Seq_Len=107, Channels=14)
    # Pair Indices: (Batch, Seq_Len=107)
    # Targets: (Batch, Seq_Len=107, Targets=5)
    print(f"Feature Batch Shape: {features.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    assert features.shape == (config.batch_size, 107, 14), "Incorrect feature shape"
    assert pair_indices.shape == (
        config.batch_size,
        107,
    ), "Incorrect pair indices shape"
    assert targets.shape == (config.batch_size, 107, 5), "Incorrect target shape"
    print("Data shapes verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n[Step 3] Initializing Model and Testing Forward Pass...")
    model = RNAModel(config).to(config.device)

    # Move batch to device
    features = features.to(config.device)
    pair_indices = pair_indices.to(config.device)
    targets = targets.to(config.device)

    # Forward pass
    outputs = model(features, pair_indices)

    print(f"Output Shape: {outputs.shape}")
    assert outputs.shape == (config.batch_size, 107, 5), "Model output shape mismatch"
    print("Model forward pass successful.")

    # 4. Loss Function Verification
    print("\n[Step 4] Verifying MCRMSE Loss Function...")
    criterion = MCRMSELoss(seq_scored=config.pred_len)
    loss = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    # Sanity check: Loss should be zero if prediction matches target exactly
    zero_loss = criterion(targets, targets)
    assert zero_loss.item() < 1e-6, "Loss should be zero for perfect predictions"
    print("Loss function logic verified.")

    # 5. Training Loop Execution
    print("\n[Step 5] Running Training Loop (Trainer)...")
    trainer = Trainer(config)

    # Fit the model (runs for 2 epochs in debug mode)
    trainer.fit(train_loader, val_loader)

    # Verify model artifact creation
    assert os.path.exists(config.model_save_path), "Model file was not saved"
    print(f"Training complete. Best model saved to: {config.model_save_path}")

    # 6. Prediction and Metric Verification
    print("\n[Step 6] Generating Predictions and Verifying Metric...")
    predictions = trainer.predict(test_loader)

    # In debug mode, test_loader only has 32 samples
    expected_samples = 32
    print(f"Prediction Shape: {predictions.shape}")
    assert predictions.shape == (expected_samples, 107, 5), "Prediction shape mismatch"

    # Verify Metric Calculation Logic manually
    # Create synthetic data: True=0, Pred=1. RMSE should be 1.0
    synth_true = np.zeros((10, 107, 5))
    synth_pred = np.ones((10, 107, 5))
    # Scored columns are indices 0, 1, 3.
    # RMSE for each is 1.0. Mean is 1.0.
    score = metric_mcrmse(synth_true, synth_pred, seq_scored=68)
    print(f"Synthetic Metric Score (Expected ~1.0): {score:.6f}")
    assert abs(score - 1.0) < 1e-5, "Metric calculation logic is incorrect"

    # 7. Submission Generation Logic
    print("\n[Step 7] Simulating Submission Generation...")
    # Note: We cannot use trainer.generate_submission() directly in debug mode
    # because the test.parquet has 240 samples but our debug predictions only have 32.
    # We will demonstrate the logic manually for the subset.

    test_df = pd.read_parquet(config.test_file)
    ids = test_df["id"].values[:expected_samples]  # Slice to match debug predictions

    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = predictions[i]  # (107, 5)
        for seq_pos in range(config.seq_len):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos].tolist()
            submission_data.append([row_id] + row_values)

    submission_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + target_cols)

    print(f"Generated Submission Rows: {len(submission_df)}")
    print("Sample rows:")
    print(submission_df.head(3))

    # Verify submission format
    assert (
        len(submission_df) == expected_samples * 107
    ), "Incorrect number of submission rows"
    assert (
        list(submission_df.columns) == ["id_seqpos"] + target_cols
    ), "Incorrect submission columns"

    # Save a demo submission file
    demo_sub_path = os.path.join(config.working_dir, "submission_demo.csv")
    submission_df.to_csv(demo_sub_path, index=False)
    print(f"Demo submission saved to {demo_sub_path}")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
