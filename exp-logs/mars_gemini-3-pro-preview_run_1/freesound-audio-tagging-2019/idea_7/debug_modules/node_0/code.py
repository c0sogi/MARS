import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import CFG
from library.utils import set_seed, calculate_lwlrap
from library.dataset import AudioDataset
from library.model import AudioResNeSt
from library.engine import train_fn, valid_fn, inference_fn, save_submission


def run_demonstration():
    # 1. Setup and Configuration Override for Speed
    print("=== Setting up demonstration ===")
    set_seed(CFG.seed)

    # Override CFG for quick execution
    CFG.debug = True
    CFG.epochs = 1
    CFG.batch_size = 8  # Small batch size for the demo
    CFG.inference_batch_size = 8
    CFG.train_duration = 2.0  # Shorter audio crops for speed

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("\n=== Preparing Data ===")
    # Load metadata
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    val_df = pd.read_csv(CFG.VAL_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)

    # Subset data for speed (use top 32 samples for train/val/test)
    subset_size = 32
    train_df = train_df.head(subset_size).reset_index(drop=True)
    val_df = val_df.head(subset_size).reset_index(drop=True)
    test_df = test_df.head(subset_size).reset_index(drop=True)

    print(
        f"Subset sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Instantiate Datasets
    train_dataset = AudioDataset(train_df, mode="train")
    val_dataset = AudioDataset(val_df, mode="val")
    test_dataset = AudioDataset(test_df, mode="test")

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        drop_last=True,
    )

    # For val/test, we use batch_size=1 because raw audio lengths might differ
    # and the dataset class does not pad/crop in 'val'/'test' mode by default
    # unless we implement a custom collate_fn.
    # However, looking at library/dataset.py, _crop_or_pad is only called if mode="train".
    # To batch val/test data in this specific implementation without a custom collate,
    # we must ensure tensors are same size or use batch_size=1.
    # For this demo, we will use batch_size=1 to be safe.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Verification: Check Data Loader Output
    print("Verifying Train Loader batch...")
    dummy_inputs, dummy_targets = next(iter(train_loader))

    # Expected Input: (Batch, 1, Freq, Time) -> Mel Spectrogram
    # Expected Target: (Batch, Num_Classes)
    print(f"Input shape: {dummy_inputs.shape}")
    print(f"Target shape: {dummy_targets.shape}")

    assert dummy_inputs.ndim == 4, "Input should be 4D tensor (B, C, F, T)"
    assert (
        dummy_targets.shape[1] == CFG.num_classes
    ), f"Target should have {CFG.num_classes} classes"

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = AudioResNeSt(pretrained=True)
    model.to(device)

    # Verification: Forward Pass
    print("Verifying Model Forward Pass...")
    with torch.no_grad():
        dummy_inputs = dummy_inputs.to(device)
        outputs = model(dummy_inputs)

    print(f"Model Output shape: {outputs.shape}")
    assert outputs.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n=== Running Training Loop (1 Epoch) ===")
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Train
    train_loss = train_fn(model, train_loader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # 5. Validation Loop Demonstration
    print("\n=== Running Validation Loop ===")
    val_loss, val_score = valid_fn(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation LWLRAP: {val_score:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_score <= 1.0, "LWLRAP score out of range [0, 1]"

    # 6. Inference and Submission
    print("\n=== Running Inference on Test Set ===")
    predictions = inference_fn(model, test_loader, device)
    print(f"Predictions shape: {predictions.shape}")

    assert predictions.shape == (
        len(test_df),
        CFG.num_classes,
    ), "Prediction shape mismatch"
    assert (
        predictions.min() >= 0.0 and predictions.max() <= 1.0
    ), "Probabilities out of range"

    print("Saving submission...")
    save_path = os.path.join(CFG.WORKING_DIR, "demo_submission.csv")
    save_submission(predictions, test_df, save_path=save_path)

    # Verify saved file
    assert os.path.exists(save_path), "Submission file not created"
    saved_df = pd.read_csv(save_path)
    assert saved_df.shape == (
        len(test_df),
        CFG.num_classes + 1,
    ), "Saved submission shape mismatch"
    assert "fname" in saved_df.columns, "fname column missing in submission"
    print(f"Submission saved successfully to {save_path}")

    # 7. Metric Logic Verification
    print("\n=== Verifying Metric Logic (LWLRAP) ===")
    # Case 1: Perfect prediction
    truth = np.array([[1, 0, 0], [0, 1, 1]])
    scores_perfect = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.7]])
    score_p = calculate_lwlrap(truth, scores_perfect)
    print(f"Perfect Score (Expected ~1.0): {score_p:.4f}")
    assert np.isclose(score_p, 1.0), "Metric calculation failed for perfect case"

    # Case 2: Worst prediction (inverse)
    scores_worst = np.array([[0.1, 0.9, 0.8], [0.9, 0.1, 0.2]])
    score_w = calculate_lwlrap(truth, scores_worst)
    print(f"Worst Score: {score_w:.4f}")
    # Calculation for worst case:
    # Item 0 (True=[1,0,0], Pred=[0.1, 0.9, 0.8] -> Rank 3) -> Prec at rank 3 = 1/3.
    # Item 1 (True=[0,1,1], Pred=[0.9, 0.1, 0.2] -> Ranks 2,3)
    #   Rank 2 (Truth 1 found): 1/2. Rank 3 (Truth 1 found): 2/3. Avg prec = (1/2 + 2/3)/2 = 7/12.
    # Per class:
    # Class 0: Item 0 (1/3) / 1 = 0.333
    # Class 1: Item 1 (1/2) / 1 = 0.5
    # Class 2: Item 1 (2/3) / 1 = 0.666
    # Mean = (0.333 + 0.5 + 0.666) / 3 = 0.5
    assert score_w < 1.0, "Metric calculation failed for worst case"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
