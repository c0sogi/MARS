import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_lrap
from library.dataset import AudioDataset, collate_fn
from library.model import AudioCRNN
from library.trainer import Trainer


def run_demonstration():
    print("=== Starting Project Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config attributes directly to speed up the run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples per dataset
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=4")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test calculate_lrap with dummy data
    # 2 samples, 3 classes
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0]])
    y_score_dummy = np.array([[0.9, 0.1, 0.8], [0.2, 0.6, 0.3]])

    lrap_score = calculate_lrap(y_true_dummy, y_score_dummy)
    print(f"Dummy LRAP Score: {lrap_score:.4f}")

    # Assertions
    assert isinstance(lrap_score, float), "LRAP should return a float"
    assert 0.0 <= lrap_score <= 1.0, "LRAP score must be between 0 and 1"
    print("LRAP calculation verified.")

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n[3] Initializing Datasets and DataLoaders...")

    # Initialize Datasets
    train_dataset = AudioDataset(mode="train")
    val_dataset = AudioDataset(mode="val")
    test_dataset = AudioDataset(mode="test")

    print(f"Train dataset size (debug): {len(train_dataset)}")
    print(f"Val dataset size (debug): {len(val_dataset)}")

    # Verify single item retrieval
    spec, label = train_dataset[0]
    print(f"Sample Spectrogram Shape: {spec.shape}")
    print(f"Sample Label Shape: {label.shape}")

    # Assertions for shapes
    # Spec: (1, n_mels, time)
    assert spec.dim() == 3, "Spectrogram must be 3D (channels, freq, time)"
    assert spec.shape[0] == 1, "Spectrogram must have 1 channel"
    assert (
        spec.shape[1] == Config.N_MELS
    ), f"Spectrogram must have {Config.N_MELS} mel bins"
    assert (
        label.shape[0] == Config.NUM_CLASSES
    ), f"Label must have {Config.NUM_CLASSES} classes"

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Verify Batch Loading
    batch_specs, batch_labels = next(iter(train_loader))
    print(f"Batch Spectrogram Shape: {batch_specs.shape}")
    print(f"Batch Labels Shape: {batch_labels.shape}")

    assert batch_specs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("DataLoaders initialized and verified.")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Model...")

    device = torch.device(Config.DEVICE)
    model = AudioCRNN().to(device)

    # Dummy Forward Pass
    dummy_input = torch.randn(2, 1, Config.N_MELS, 200).to(
        device
    )  # (batch, channel, freq, time)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model initialized and forward pass verified.")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Starting Training Loop...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run training
    # Since we set EPOCHS=1, this will run one epoch of training and validation
    trainer.train(patience=1)

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training loop completed successfully.")

    # ---------------------------------------------------------
    # 6. Prediction / Inference
    # ---------------------------------------------------------
    print("\n[6] Generating Predictions...")

    trainer.predict()

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    # Load submission and check format
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Expected columns: fname + 80 classes
    expected_cols = 1 + Config.NUM_CLASSES
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Submission should have {expected_cols} columns"
    assert "fname" in sub_df.columns, "Submission missing 'fname' column"

    # Check if probabilities are valid
    # Select a few probability columns
    prob_cols = [c for c in sub_df.columns if c != "fname"]
    probs = sub_df[prob_cols].values
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1"

    print("Prediction completed and submission verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
