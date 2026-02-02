import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import logging

# Import provided library modules
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data as data_lib
import library.train as train_lib


def main():
    # =========================================================================
    # 1. Setup and Reproducibility
    # =========================================================================
    print(">>> [1/6] Setting up environment...")
    utils.seed_everything(config.SEED)

    # Ensure the working directory defined in config exists
    os.makedirs(config.WORK_DIR, exist_ok=True)
    print(f"Working directory: {config.WORK_DIR}")

    # =========================================================================
    # 2. Verify Audio Preprocessing Logic
    # =========================================================================
    print(">>> [2/6] Verifying AudioPreprocessor...")

    # Create a dummy wav file to test the preprocessor
    dummy_filename = "dummy_test.wav"
    dummy_path = os.path.join(config.WORK_DIR, dummy_filename)
    sr = config.SR
    # Generate 2 seconds of white noise
    dummy_audio = np.random.uniform(-0.5, 0.5, int(sr * 2.0))
    sf.write(dummy_path, dummy_audio, sr)

    # Instantiate Preprocessor
    preprocessor = data_lib.AudioPreprocessor()

    # Monkey-patch INPUT_ROOT in data_lib so it looks in WORK_DIR for our dummy file
    # The preprocessor uses 'INPUT_ROOT' imported into data.py
    original_input_root = data_lib.INPUT_ROOT
    data_lib.INPUT_ROOT = config.WORK_DIR

    try:
        # Process the dummy file
        spec = preprocessor.process(dummy_filename)

        # Check Shape: (1, F, T)
        # T depends on length (4000 samples) and hop length (64).
        # Approx T = 4000 // 64 + 1 = 63.
        # torchaudio MelSpectrogram output width calculation can vary slightly, usually (L / hop) + 1
        print(f"Spectrogram Shape: {spec.shape}")

        if spec.shape[0] != 1:
            raise AssertionError(f"Expected channel dim 1, got {spec.shape[0]}")
        if spec.shape[1] != config.N_MELS:
            raise AssertionError(f"Expected {config.N_MELS} mels, got {spec.shape[1]}")

        # Capture the actual time dimension to generate matching mock data
        time_dim = spec.shape[2]

    finally:
        # Restore original INPUT_ROOT
        data_lib.INPUT_ROOT = original_input_root
        # Cleanup dummy file
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

    print("AudioPreprocessor logic verified.")

    # =========================================================================
    # 3. Generate Mock Data for Speed
    # =========================================================================
    print(">>> [3/6] Generating mock data for rapid execution...")

    # We create small numpy arrays and save them where get_data() expects them.
    # This simulates the result of processing the full dataset.

    num_train = 60
    num_test = 20

    # Generate random spectrograms: (N, 1, F, T)
    mock_train_data = np.random.randn(num_train, 1, config.N_MELS, time_dim).astype(
        np.float32
    )
    # Ensure we have both classes represented for StratifiedKFold
    mock_train_labels = np.zeros(num_train, dtype=np.float32)
    mock_train_labels[::2] = 1.0  # Alternate 0 and 1

    mock_test_data = np.random.randn(num_test, 1, config.N_MELS, time_dim).astype(
        np.float32
    )
    mock_test_clips = np.array([f"test_clip_{i}.aif" for i in range(num_test)])

    # Save to WORK_DIR with names expected by get_data(load_cached_data=True)
    # Note: config.DEBUG is False, so no "_debug" suffix is used by default.
    np.save(os.path.join(config.WORK_DIR, "train_data.npy"), mock_train_data)
    np.save(os.path.join(config.WORK_DIR, "train_labels.npy"), mock_train_labels)
    np.save(os.path.join(config.WORK_DIR, "test_data.npy"), mock_test_data)
    np.save(os.path.join(config.WORK_DIR, "test_clips.npy"), mock_test_clips)

    # Verify get_data loads this
    data_loaded, labels_loaded, _, _ = data_lib.get_data(load_cached_data=True)
    if len(data_loaded) != num_train:
        raise AssertionError("Mock data was not loaded correctly.")

    print(f"Mock data generated: {num_train} train samples, {num_test} test samples.")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print(">>> [4/6] Verifying Model Architecture...")

    model = model_lib.get_model()
    model.eval()

    # Create a dummy batch
    dummy_batch = torch.from_numpy(mock_train_data[:4])  # Batch of 4

    with torch.no_grad():
        output = model(dummy_batch)

    print(f"Model Output Shape: {output.shape}")
    if output.shape != (4, config.NUM_CLASSES):
        raise AssertionError(
            f"Expected output shape (4, {config.NUM_CLASSES}), got {output.shape}"
        )

    print("Model architecture verified.")

    # =========================================================================
    # 5. Configure and Run Training Pipeline
    # =========================================================================
    print(">>> [5/6] Running Training Pipeline (Patched for Speed)...")

    # Patch hyperparameters in library modules to run a minimal training loop
    # We reduce folds, epochs, and patience to ensure this finishes in seconds.

    # Patch library.train
    train_lib.N_FOLDS = 2
    train_lib.NUM_EPOCHS = 1
    train_lib.PATIENCE = 1

    # Patch library.data (used for DataLoader creation)
    data_lib.N_FOLDS = 2
    data_lib.BATCH_SIZE = 8  # Small batch size for small mock data

    # Execute the full training and prediction pipeline
    # This will:
    # 1. Train models for 2 folds (1 epoch each)
    # 2. Save checkpoints to WORK_DIR
    # 3. Load checkpoints and run inference on test data
    # 4. Save submission.csv
    train_lib.train_and_predict()

    # =========================================================================
    # 6. Verify Submission
    # =========================================================================
    print(">>> [6/6] Verifying Submission...")

    submission_path = config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df.head())

    # Validations
    if df.shape[0] != num_test:
        raise AssertionError(f"Submission has {df.shape[0]} rows, expected {num_test}")

    if list(df.columns) != ["clip", "probability"]:
        raise AssertionError(f"Invalid columns: {df.columns}")

    if df["probability"].isnull().any():
        raise AssertionError("Submission contains NaN probabilities")

    print("\n>>> SUCCESS: Pipeline executed and submission generated.")


if __name__ == "__main__":
    main()
