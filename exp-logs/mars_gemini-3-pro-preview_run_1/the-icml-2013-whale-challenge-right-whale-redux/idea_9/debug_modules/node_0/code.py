import os
import torch
import numpy as np
import pandas as pd
import sys

# Import provided library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model
import library.trainer as trainer


def run_demo():
    print("=== Starting Library Verification and Demo ===\n")

    # 1. Setup & Utils Verification
    print("--- Verifying Utils ---")
    utils.set_seed(42)

    # Test AUC calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = utils.compute_score(y_true, y_pred)
    print(f"Computed AUC Score: {score}")
    assert isinstance(score, float), "Score should be a float"
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    print("Utils verification passed.\n")

    # 2. Dataset & Transforms Verification
    print("--- Verifying Dataset & Transforms ---")
    # Simulate a raw waveform batch: (Batch=2, Samples=4000)
    # 4000 samples @ 2000Hz = 2 seconds
    dummy_waveform = torch.randn(2, 4000)

    # Get transforms
    transforms = dataset.get_transforms(train=True)

    # Apply transforms manually to check shape
    # Transform expects (Channels, Time), so we add channel dim
    dummy_input = dummy_waveform.unsqueeze(1)  # (2, 1, 4000)
    # Since transforms are typically applied per item in Dataset,
    # let's simulate what happens inside __getitem__ for a single item
    single_waveform = dummy_waveform[0].unsqueeze(0)  # (1, 4000)
    spec = transforms(single_waveform)

    print(f"Spectrogram Shape: {spec.shape}")
    # Expected: (1, 128, TimeFrames)
    # TimeFrames = 1 + (4000 - n_fft) // hop_length ... roughly 4000/10 = 400
    assert spec.dim() == 3, "Spectrogram should be 3D (C, F, T)"
    assert spec.shape[0] == 1, "Should have 1 channel"
    assert spec.shape[1] == config.N_MELS, f"Should have {config.N_MELS} mel bins"

    # Test Mixup
    dummy_target = torch.tensor([0.0, 1.0])
    mixed_x, y_a, y_b, lam = dataset.mixup_data(
        dummy_input, dummy_target, alpha=0.4, use_cuda=False
    )
    print(f"Mixed Input Shape: {mixed_x.shape}")
    assert mixed_x.shape == dummy_input.shape, "Mixed input shape mismatch"
    assert y_a.shape == dummy_target.shape, "Target A shape mismatch"
    print("Dataset verification passed.\n")

    # 3. Model Verification
    print("--- Verifying Model Architecture ---")
    net = model.SKResNetCRNN()
    net.eval()

    # Create dummy input matching the spectrogram shape: (Batch, 1, Freq, Time)
    # Using the time dimension from the transform test above
    time_dim = spec.shape[2]
    dummy_spec_batch = torch.randn(4, 1, config.N_MELS, time_dim)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_spec_batch)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Output shape should be (Batch, 1)"
    print("Model verification passed.\n")

    # 4. Trainer Integration (Full Pipeline)
    print("--- Verifying Trainer (Debug Mode) ---")

    # Initialize Trainer in debug mode
    # This will load only config.DEBUG_SIZE (100) samples
    # It creates/loads cache in ./working/idea_9/
    my_trainer = trainer.Trainer(load_cached_data=False, debug=True)

    # Verify DataLoaders are populated
    print(f"Train batches: {len(my_trainer.train_loader)}")
    print(f"Val batches: {len(my_trainer.val_loader)}")
    assert len(my_trainer.train_loader) > 0, "Train loader is empty"

    # Run Training for 1 epoch
    print("Running fit (1 epoch)...")
    my_trainer.fit(epochs=1)

    # Check if best model was saved
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint found at {best_model_path}")
    else:
        # It might not save if validation AUC is 0 or fails to improve,
        # but with random init and 1 epoch it might happen.
        # However, the code saves if val_auc > best_auc (init 0).
        # If val_auc is 0.5 (fallback), it should save.
        print(
            "Note: Best model might not have been saved if AUC didn't improve over 0."
        )

    # Run Prediction
    print("Running prediction...")
    submission_df = my_trainer.predict()

    # Verify Submission
    print("Verifying submission...")
    assert isinstance(submission_df, pd.DataFrame), "Predict should return a DataFrame"
    assert "clip" in submission_df.columns, "Submission missing 'clip' column"
    assert (
        "probability" in submission_df.columns
    ), "Submission missing 'probability' column"

    # Since we ran in debug mode, the submission should have DEBUG_SIZE rows
    expected_len = config.DEBUG_SIZE
    assert (
        len(submission_df) == expected_len
    ), f"Expected {expected_len} predictions, got {len(submission_df)}"

    # Check file output
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found on disk"
    print(f"Submission successfully generated at {config.SUBMISSION_PATH}")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
