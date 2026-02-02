import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
import library.config
from library.utils import set_seed, levenshtein_distance, decode_sequence
from library.data_loader import get_dataloaders
from library.model import RLSGCN
from library.loss import DeepSupervisionLoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def configure_demo_settings():
    """
    Modifies the global configuration to run a fast demonstration.
    Reduces epochs, batch size, and model complexity.
    """
    print("Configuring demo settings...")

    # Reduce training duration
    library.config.HYPERPARAMS["num_epochs"] = 2
    library.config.HYPERPARAMS["batch_size"] = 4

    # Reduce model complexity for speed
    library.config.HYPERPARAMS["model"]["tcn_num_layers"] = 4  # Default is 10
    library.config.HYPERPARAMS["model"]["tcn_channels"] = 64  # Default is 256
    library.config.HYPERPARAMS["model"]["lstm_hidden_dim"] = 64  # Default is 256

    # Clear cache to ensure data processing logic is tested
    cache_dir = library.config.CACHE_DIR
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def test_utilities():
    """
    Validates helper functions in library/utils.py.
    """
    print("Testing utilities...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    assert (
        levenshtein_distance(seq1, seq2) == 0
    ), "Distance should be 0 for identical sequences"

    seq3 = [1, 2]
    assert levenshtein_distance(seq1, seq3) == 1, "Distance should be 1 (deletion)"

    seq4 = [1, 4, 3]
    assert levenshtein_distance(seq1, seq4) == 1, "Distance should be 1 (substitution)"

    # Test Sequence Decoding
    # Logic: Collapse consecutive duplicates, remove 0 (background)
    raw_frames = [0, 0, 1, 1, 1, 0, 2, 2, 0, 3, 3, 0]
    decoded = decode_sequence(raw_frames)
    expected = [1, 2, 3]
    assert decoded == expected, f"Decoding failed. Got {decoded}, expected {expected}"

    print("Utilities validation passed.")


def test_data_pipeline():
    """
    Tests data loading, augmentation, and batching.
    """
    print("Testing data pipeline...")

    # Load a tiny subset of data (10 samples)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=library.config.HYPERPARAMS["batch_size"],
        num_workers=0,  # Use main process for demo stability
        sample_limit=10,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Keys
    expected_keys = [
        "features",
        "mask",
        "cls_target",
        "bnd_target",
        "lengths",
        "sample_ids",
    ]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Verify Shapes
    # Features: (Batch, Time, InputDim)
    features = batch["features"]
    input_dim = library.config.INPUT_DIM  # 85
    assert features.ndim == 3
    assert (
        features.shape[2] == input_dim
    ), f"Expected input dim {input_dim}, got {features.shape[2]}"

    # Targets: (Batch, Time)
    cls_target = batch["cls_target"]
    assert cls_target.ndim == 2
    assert cls_target.shape == (features.shape[0], features.shape[1])

    print(f"Data pipeline verified. Batch shape: {features.shape}")
    return train_loader, val_loader, test_loader


def test_model_logic(train_loader):
    """
    Tests model instantiation, forward pass, loss computation, and backward pass.
    """
    print("Testing model and loss logic...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = RLSGCN().to(device)

    # Instantiate Loss
    criterion = DeepSupervisionLoss().to(device)

    # Get Batch
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    mask = batch["mask"].to(device)
    cls_target = batch["cls_target"].to(device)
    bnd_target = batch["bnd_target"].to(device)
    targets = {"cls_target": cls_target, "bnd_target": bnd_target}

    # Forward Pass
    outputs = model(features, mask)

    # Verify Multi-Stage Outputs
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    # Verify Output Shapes (Stage 3 Probabilities)
    # Expected: (Batch, Time, NumClasses)
    probs = outputs["stage3"]["cls_probs"]
    num_classes = library.config.HYPERPARAMS["model"]["num_classes"]  # 21
    assert (
        probs.shape[2] == num_classes
    ), f"Expected {num_classes} classes, got {probs.shape[2]}"

    # Compute Loss
    loss, loss_dict = criterion(outputs, targets, mask)

    # Verify Loss
    assert loss.item() > 0, "Loss should be non-zero"
    assert "total_loss" in loss_dict
    assert "stage3_ce" in loss_dict

    # Backward Pass (Check for errors)
    loss.backward()

    print("Model logic verified.")


def run_full_training_cycle(train_loader, val_loader, test_loader):
    """
    Runs the Trainer to simulate a full experiment lifecycle: Fit -> Predict.
    """
    print("Running full training cycle...")

    trainer = Trainer(train_loader, val_loader, test_loader)

    # 1. Train (Fit)
    # This will run for the reduced number of epochs defined in configure_demo_settings
    trainer.fit()

    # 2. Inference (Predict)
    trainer.predict()

    # 3. Validate Submission File
    submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Check content
    df = pd.read_csv(submission_path, header=None)
    print(f"Submission generated with {len(df)} predictions.")

    # Verify format of first line (SessionID, Labels...)
    first_line = df.iloc[0, 0] if not df.empty else ""
    # Assuming CSV read might split by comma, check raw file or dataframe structure
    # The format_submission_line creates "SessionID,L1,L2".
    # Pandas read_csv without header might treat L1, L2 as columns.
    assert len(df.columns) >= 1, "Submission file is empty or malformed"

    print("Training cycle completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    configure_demo_settings()

    # 2. Unit Tests
    test_utilities()

    # 3. Data Loading
    train_dl, val_dl, test_dl = test_data_pipeline()

    # 4. Model Verification
    test_model_logic(train_dl)

    # 5. Integration Test (Trainer)
    run_full_training_cycle(train_dl, val_dl, test_dl)

    print("\nAll demonstrations passed.")
