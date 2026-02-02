import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import TimePreservingEfficientNet
from library.engine import train_one_epoch, evaluate, predict

if __name__ == "__main__":
    print("Starting demonstration of Right Whale Detection pipeline...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for a fast demonstration run
    Config.DEBUG = True  # Use a small subset (100 samples)
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Configuration: Debug={Config.DEBUG}, Device={Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Testing Data Loading ---")
    # This will load metadata, process audio to spectrograms (caching them), and return loaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Verify Train Loader
    train_batch = next(iter(train_loader))
    train_imgs, train_labels = train_batch

    print(f"Train Batch Shape: Data={train_imgs.shape}, Labels={train_labels.shape}")

    # Assertions for Data Structure
    # Shape: [Batch, Channels, Freq, Time]
    # Channels should be 1, Freq = 128 (N_MELS), Time = 401 (approx for 2s with hop 20)
    # Note: Time dimension depends on padding/stft details, usually fixed.
    # Based on config: 2.0s * 2000Hz = 4000 samples.
    # MelSpectrogram output width approx: 4000 // 20 + 1 = 201.
    assert train_imgs.dim() == 4, "Train images should be 4-dimensional"
    assert train_imgs.size(1) == 1, "Input should have 1 channel"
    assert train_imgs.size(2) == Config.N_MELS, f"Freq dim should be {Config.N_MELS}"
    assert train_labels.dim() == 1, "Labels should be 1-dimensional"
    assert len(train_loader.dataset) <= 100, "Debug mode should limit dataset size"

    print("Data Loading verification passed.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Testing Model Initialization ---")
    device = torch.device(Config.DEVICE)
    model = TimePreservingEfficientNet()
    model = model.to(device)

    # Verify Forward Pass
    dummy_input = train_imgs.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions for Model Output
    assert (
        output.dim() == 2
    ), "Model output should be 2-dimensional [Batch, Num_Classes]"
    assert output.size(0) == train_imgs.size(0), "Batch size should be preserved"
    assert (
        output.size(1) == 1
    ), "Output dimension should be 1 (Binary Classification Logits)"

    print("Model initialization verification passed.")

    # ==========================================
    # 4. Training & Evaluation Loop
    # ==========================================
    print("\n--- Testing Training Loop ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run for configured epochs
    for epoch in range(Config.EPOCHS):
        print(f"Running Epoch {epoch + 1}...")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Assertions for Metrics
        assert not np.isnan(train_loss), "Training loss should not be NaN"
        assert not np.isnan(val_loss), "Validation loss should not be NaN"
        assert 0.0 <= val_auc <= 1.0, "AUC should be between 0 and 1"

    print("Training loop verification passed.")

    # ==========================================
    # 5. Inference / Prediction
    # ==========================================
    print("\n--- Testing Prediction ---")

    # Define output path
    output_csv = Config.SUBMISSION_PATH

    # Run prediction
    predict(model, test_loader, device, output_csv)

    # Verify Output File
    assert os.path.exists(output_csv), "Submission file was not created"

    df_submission = pd.read_csv(output_csv)
    print(f"Submission Head:\n{df_submission.head()}")

    # Assertions for Submission
    assert "clip" in df_submission.columns, "Submission must have 'clip' column"
    assert (
        "probability" in df_submission.columns
    ), "Submission must have 'probability' column"
    # In debug mode, we expect 100 rows
    assert (
        len(df_submission) == 100
    ), f"Expected 100 rows in debug mode, got {len(df_submission)}"
    assert (
        df_submission["probability"].dtype == float
    ), "Probability column should be float"
    assert (
        df_submission["probability"].between(0, 1).all()
    ), "Probabilities must be between 0 and 1"

    print("Prediction verification passed.")

    print("\nAll demonstrations completed successfully.")
