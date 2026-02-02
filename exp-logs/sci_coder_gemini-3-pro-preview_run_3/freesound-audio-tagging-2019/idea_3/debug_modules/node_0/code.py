import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import AudioDataset, mixup_data
from library.model import AudioClassifier
from library.train import train_epoch, validate


def main():
    print("Starting Audio Tagging Library Demonstration...")

    # =========================================================================
    # 1. Configuration Overrides for Speed/Demo
    # =========================================================================
    print("\n[1] Configuring environment for fast demonstration...")
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset
    Config.epochs = 1
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small data
    Config.transformer_layers = 1  # Reduce model complexity for speed
    Config.backbone_name = "efficientnet_b0"  # Lighter backbone if supported, though code uses b2 hardcoded in model.py, so this might strictly be ignored by model.py but good for intent.
    # Note: model.py hardcodes efficientnet_b2, so we stick with that but use fewer transformer layers.

    set_seed(Config.seed)
    device = Config.device
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.debug}")

    # =========================================================================
    # 2. Dataset and Augmentation Verification
    # =========================================================================
    print("\n[2] Verifying Dataset and Mixup...")

    # Initialize Dataset
    train_dataset = AudioDataset(Config.train_csv_path, mode="train")
    print(f"    Train Dataset size (debug): {len(train_dataset)}")

    # Fetch one sample
    spec, label = train_dataset[0]
    print(f"    Spectrogram shape: {spec.shape}")
    print(f"    Label vector shape: {label.shape}")

    # Assertions
    assert spec.dim() == 3, "Spectrogram must be 3D (channels, freq, time)"
    assert spec.shape[0] == 1, "Spectrogram channel should be 1"
    assert spec.shape[1] == Config.n_mels, f"Freq dim should be {Config.n_mels}"
    assert (
        label.shape[0] == Config.num_classes
    ), f"Label dim should be {Config.num_classes}"

    # Test Mixup
    # Create a dummy batch
    batch_size = 4
    dummy_specs = torch.stack([spec] * batch_size).to(device)
    dummy_labels = torch.stack([label] * batch_size).to(device)

    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_specs, dummy_labels, alpha=1.0, device=device
    )

    assert mixed_x.shape == dummy_specs.shape, "Mixup output shape mismatch"
    assert y_a.shape == dummy_labels.shape, "Mixup label A shape mismatch"
    assert y_b.shape == dummy_labels.shape, "Mixup label B shape mismatch"
    print("    Mixup verification passed.")

    # =========================================================================
    # 3. Model Verification
    # =========================================================================
    print("\n[3] Verifying Model Architecture...")

    model = AudioClassifier().to(device)
    model.eval()

    # Forward pass with dummy input
    # Input shape: (Batch, Channels, Freq, Time)
    # Time dimension depends on duration and hop length.
    # Target length is defined in Config.target_length.
    # MelSpectrogram output width approx: target_length // hop_length
    expected_width = Config.target_length // Config.hop_length + 1
    dummy_input = torch.randn(2, 1, Config.n_mels, expected_width).to(device)

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Model Output shape: {logits.shape}")

    assert logits.shape == (
        2,
        Config.num_classes,
    ), f"Expected output shape (2, {Config.num_classes}), got {logits.shape}"
    print("    Model forward pass verification passed.")

    # =========================================================================
    # 4. Metric Verification (LWLRAP)
    # =========================================================================
    print("\n[4] Verifying LWLRAP Metric...")

    # Case: 2 samples, 3 classes
    # Sample 1: True=[1, 0, 0], Pred=[0.8, 0.1, 0.1] -> Rank 1 correct -> AP=1.0
    # Sample 2: True=[0, 1, 0], Pred=[0.2, 0.7, 0.1] -> Rank 1 correct -> AP=1.0
    # Overall LWLRAP should be 1.0
    y_true_simple = np.array([[1, 0, 0], [0, 1, 0]])
    y_score_simple = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])

    score = calculate_lwlrap(y_true_simple, y_score_simple)
    print(f"    Simple Case Score: {score}")
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"

    # Case: Mixed
    # Sample 1: True=[1, 0], Pred=[0.2, 0.8]
    #   Sorted indices: [1, 0] (Class 1, Class 0)
    #   Sorted Truth: [0, 1]
    #   Precisions: k=1 (0/1=0), k=2 (1/2=0.5). Relevant: 0.5. Sum: 0.5
    # Sample 2: True=[1, 0], Pred=[0.9, 0.1]
    #   Sorted indices: [0, 1]
    #   Sorted Truth: [1, 0]
    #   Precisions: k=1 (1/1=1). Relevant: 1. Sum: 1.0
    # Per class sum: Class 0 (from S1=0.5, S2=1.0) -> Wait, logic check:
    # calculate_lwlrap sums relevant precisions per class.
    # S1 contributes to Class 0's score? No, S1 true label is Class 0.
    # Actually, let's rely on the function logic.
    # S1: Target is Class 0. Rank of Class 0 is 2. Precision at 2 is 1/2 = 0.5.
    # S2: Target is Class 0. Rank of Class 0 is 1. Precision at 1 is 1/1 = 1.0.
    # Class 0 LWLRAP = (0.5 + 1.0) / 2 = 0.75.
    # Class 1 has no true samples.
    # Overall score should be 0.75.

    y_true_mixed = np.array([[1, 0], [1, 0]])
    y_score_mixed = np.array([[0.2, 0.8], [0.9, 0.1]])

    score_mixed = calculate_lwlrap(y_true_mixed, y_score_mixed)
    print(f"    Mixed Case Score: {score_mixed}")
    assert np.isclose(score_mixed, 0.75), f"Expected 0.75, got {score_mixed}"

    print("    LWLRAP verification passed.")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n[5] Demonstrating Training Loop...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True
    )

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=1, steps_per_epoch=len(train_loader)
    )

    # Run one epoch
    print("    Running 1 epoch of training...")
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, device
    )
    print(f"    Train Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # Run validation
    print("    Running validation...")
    val_dataset = AudioDataset(Config.val_csv_path, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    val_loss, val_lwlrap = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val LWLRAP: {val_lwlrap:.4f}")

    assert val_lwlrap >= 0.0 and val_lwlrap <= 1.0, "LWLRAP must be between 0 and 1"
    print("    Training loop verification passed.")

    # =========================================================================
    # 6. Inference Demonstration
    # =========================================================================
    print("\n[6] Demonstrating Inference...")

    test_dataset = AudioDataset(Config.test_csv_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    print(f"    Test Dataset size (debug): {len(test_dataset)}")

    model.eval()
    all_preds = []

    # Run inference on a few batches
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 2:
                break  # Limit to 2 batches
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output)
            all_preds.append(preds.cpu().numpy())

    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        print(f"    Predictions shape: {all_preds.shape}")

        # Verify values
        assert (all_preds >= 0).all() and (
            all_preds <= 1
        ).all(), "Predictions must be probabilities [0, 1]"

        # Verify Submission Format construction
        # Just checking logic, not saving file to avoid IO overhead in demo
        fnames = test_dataset.df["fname"].values[: len(all_preds)]
        classes = test_dataset.classes

        sub_df = pd.DataFrame(all_preds, columns=classes)
        sub_df.insert(0, "fname", fnames)

        print(
            f"    Submission DataFrame head:\n{sub_df.iloc[:2, :3]}"
        )  # Show first few cols
        assert sub_df.shape == (
            len(all_preds),
            Config.num_classes + 1,
        ), "Submission DataFrame shape incorrect"

    print("    Inference verification passed.")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
