import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure current directory is in path
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.models import WhaleEfficientNet, WhaleDenseNet
from library.inference import save_submission


def main():
    print("=== Right Whale Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed & Safety
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for a fast demo run
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2
    # Disable pretrained weights to avoid download timeouts/errors in demo environment
    Config.PRETRAINED = False

    # Set random seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")
    # get_dataloaders handles processing audio -> spectrograms and caching .npy files
    # This might take a moment on the first run as it processes the audio files.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Train Data
    print("Verifying Train Loader...")
    train_batch, train_labels = next(iter(train_loader))
    print(f"Train Batch Shape: {train_batch.shape}")  # Expected: (B, 1, 128, T)
    print(f"Train Labels Shape: {train_labels.shape}")  # Expected: (B,)

    # Assertions
    assert train_batch.ndim == 4, "Train batch must be 4D (Batch, Channel, Freq, Time)"
    assert train_batch.shape[1] == 1, "Input channel must be 1 (Spectrogram)"
    assert train_labels.ndim == 1, "Labels must be 1D"

    # Verify Test Data
    print("Verifying Test Loader...")
    test_batch, test_clips = next(iter(test_loader))
    print(f"Test Batch Shape: {test_batch.shape}")

    # Assertions
    assert isinstance(
        test_clips[0], (str, np.str_)
    ), "Test clips must be filenames (strings)"
    assert len(test_clips) == Config.BATCH_SIZE, "Test clip batch size mismatch"

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Initializing Models...")

    # Instantiate models (using pretrained=False for speed/offline safety)
    model_a = WhaleEfficientNet(pretrained=False).to(device)
    model_b = WhaleDenseNet(pretrained=False).to(device)

    print(f"Model A: {type(model_a).__name__}")
    print(f"Model B: {type(model_b).__name__}")

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    dummy_input = train_batch.to(device)

    with torch.no_grad():
        out_a = model_a(dummy_input)
        out_b = model_b(dummy_input)

    print(f"Output Shape: {out_a.shape}")

    # Assertions
    assert out_a.shape == (Config.BATCH_SIZE, 1), "Model A output shape incorrect"
    assert out_b.shape == (Config.BATCH_SIZE, 1), "Model B output shape incorrect"

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (Subset)...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    model_a.train()
    steps = 0
    max_steps = 10  # Limit steps for demo speed

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        output = model_a(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        steps += 1
        if steps >= max_steps:
            break

    print(f"Completed {steps} training steps successfully.")

    # ---------------------------------------------------------
    # 5. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Inference (Subset)...")

    model_a.eval()
    test_steps = 0
    max_test_steps = 5

    all_clips = []
    all_probs = []

    with torch.no_grad():
        for data, clips in test_loader:
            data = data.to(device)

            # Forward pass
            logits = model_a(data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_clips.extend(clips)
            all_probs.extend(probs)

            test_steps += 1
            if test_steps >= max_test_steps:
                break

    print(f"Generated predictions for {len(all_clips)} clips.")

    # Assertions
    assert len(all_clips) == len(all_probs), "Mismatch between clips and probabilities"
    all_probs = np.array(all_probs)
    assert np.all(
        (all_probs >= 0) & (all_probs <= 1)
    ), "Probabilities out of range [0, 1]"

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission File...")

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Use the utility function to save
    save_submission(
        np.array(all_clips), np.array(all_probs), output_path=submission_path
    )

    # Verify file
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print("\nSubmission Head:")
    print(df_sub.head())

    # Validate format
    required_cols = ["clip", "probability"]
    assert list(df_sub.columns) == required_cols, f"Columns must be {required_cols}"
    assert df_sub["probability"].dtype == float, "Probability column should be float"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")
    main()
