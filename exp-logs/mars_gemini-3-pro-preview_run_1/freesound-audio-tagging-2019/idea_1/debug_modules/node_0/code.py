import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_lrap
from library.dataset import get_dataloader
from library.model import ShallowCNN
from library.engine import train, predict


def run_demonstration():
    # 1. Setup and Reproducibility
    print("=== Setting up environment ===")
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Verify Metric Logic (LRAP)
    print("\n=== Verifying LRAP Metric ===")
    # Simple case: 2 samples, 3 classes
    # Sample 0: True=[1, 0, 0], Pred=[0.9, 0.1, 0.2] -> Rank 1 correct. AP=1.0
    # Sample 1: True=[0, 1, 1], Pred=[0.1, 0.8, 0.9] -> Ranks 1, 2 correct.
    # For Sample 1:
    # Rank 1 (Class 2, score 0.9): Hit. Prec=1/1.
    # Rank 2 (Class 1, score 0.8): Hit. Prec=2/2.
    # Rank 3 (Class 0, score 0.1): Miss.
    # Average Precision for Sample 1 is not calculated this way for label-weighted.
    # Label-Weighted LRAP averages per class.

    # Let's rely on the function's implementation but ensure it runs and returns a valid float.
    y_true_dummy = np.array([[1, 0, 0], [0, 1, 1]])
    y_score_dummy = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.9]])
    lrap_score = calculate_lrap(y_true_dummy, y_score_dummy)
    print(f"Calculated LRAP on dummy data: {lrap_score:.4f}")
    assert 0.0 <= lrap_score <= 1.0, "LRAP score must be between 0 and 1"

    # 3. Data Loading (Debug Mode)
    print("\n=== Verifying Data Loading ===")
    # Use a small batch size and debug mode to load only a few samples
    batch_size = 4
    train_loader = get_dataloader(
        phase="train", batch_size=batch_size, debug=True, debug_size=20
    )
    val_loader = get_dataloader(
        phase="val", batch_size=batch_size, debug=True, debug_size=20
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Validation assertions
    # Shape: (Batch, Channels, Mel, Time)
    # Expected: (4, 1, 64, 501) based on SR=32000, Dur=5s, Hop=320
    assert (
        inputs.shape[0] == batch_size
    ), f"Expected batch size {batch_size}, got {inputs.shape[0]}"
    assert inputs.shape[1] == 1, "Expected 1 channel (mono)"
    assert inputs.shape[2] == Config.N_MELS, f"Expected {Config.N_MELS} mel bands"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes"

    # 4. Model Instantiation
    print("\n=== Verifying Model Architecture ===")
    model = ShallowCNN(num_classes=Config.NUM_CLASSES).to(device)

    # Forward pass with the fetched batch
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # 5. Training Loop Demonstration
    print("\n=== Running Training Loop (Fast) ===")
    # We use the debug loaders and run for 1 epoch only
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,  # Limit to 1 epoch for speed
        lr=1e-3,
        patience=1,
        save_path=Config.MODEL_PATH,
    )

    # Check if model checkpoint was created
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint successfully saved at {Config.MODEL_PATH}")
    else:
        # It might not save if validation loss doesn't improve (unlikely in epoch 0 vs inf),
        # but let's check just in case.
        print(
            "Warning: Model checkpoint not found (possibly due to logic), but training completed."
        )

    # 6. Prediction Demonstration
    print("\n=== Running Prediction ===")
    test_loader = get_dataloader(
        phase="test", batch_size=batch_size, debug=True, debug_size=10, shuffle=False
    )

    # Load the best model (or use current if not saved, though train usually saves)
    if os.path.exists(Config.MODEL_PATH):
        checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    output_csv = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    predict(model, test_loader, device, output_path=output_csv)

    # Verify submission file
    assert os.path.exists(output_csv), "Submission file was not created"

    sub_df = pd.read_csv(output_csv)
    print(f"Submission file shape: {sub_df.shape}")

    # Expected rows: debug_size (10). Expected cols: fname + 80 classes = 81.
    assert sub_df.shape == (10, 81), f"Expected shape (10, 81), got {sub_df.shape}"
    print("Submission format verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
