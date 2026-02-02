import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import set_seed, ModelEMA, map_fine_to_coarse
from library.dataset import get_dataloaders
from library.transforms import AudioTransforms
from library.model import get_model
from library.trainer import train_epoch, validate


def run_demo():
    print("=" * 50)
    print("SPEECH RECOGNITION PIPELINE DEMONSTRATION")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch for demo
    Config.WORKING_DIR = "./working/demo_execution"

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subset Creation
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data and Creating Subsets...")

    # Load full dataloaders (this handles metadata processing and caching)
    # We force load_cached_data=False to verify the processing logic for the demo
    train_loader_full, val_loader_full, test_loader_full = get_dataloaders(
        load_cached_data=False
    )

    # Create small subsets for rapid testing (e.g., 64 samples = 4 batches of 16)
    subset_indices = list(range(64))

    train_subset = Subset(train_loader_full.dataset, subset_indices)
    val_subset = Subset(val_loader_full.dataset, subset_indices)
    test_subset = Subset(test_loader_full.dataset, subset_indices)

    train_loader_small = DataLoader(
        train_subset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader_small = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader_small = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    print(f"    Train Subset Size: {len(train_subset)}")
    print(f"    Val Subset Size:   {len(val_subset)}")
    print(f"    Test Subset Size:  {len(test_subset)}")

    # Verify Data Integrity
    batch_waveforms, batch_labels, batch_fnames = next(iter(train_loader_small))
    print(f"    Batch Waveform Shape: {batch_waveforms.shape}")
    print(f"    Batch Labels Shape:   {batch_labels.shape}")

    # Assertions
    assert batch_waveforms.dim() == 2, "Waveforms should be 2D (Batch, Time)"
    assert (
        batch_waveforms.shape[1] == Config.AUDIO_LEN
    ), f"Audio length mismatch. Expected {Config.AUDIO_LEN}, got {batch_waveforms.shape[1]}"
    assert batch_labels.max() < Config.NUM_CLASSES, "Label index out of bounds"

    # -------------------------------------------------------------------------
    # 3. Transforms Verification (GPU)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying GPU Transforms (Mel Spec + Deltas + Mixup)...")

    transforms = AudioTransforms(device=device)

    # Move batch to device for transform testing
    gpu_waveforms = batch_waveforms.to(device)
    gpu_labels = batch_labels.to(device)

    # A. Test Training Transforms (with Mixup)
    features_train, targets_a, targets_b, lam = transforms(
        gpu_waveforms, gpu_labels, train=True, mixup_alpha=1.0
    )

    print(f"    Train Features Shape: {features_train.shape}")
    # Expected: (Batch, 3, n_mels, time)
    # Time dim depends on n_fft/hop. For 16000sr, 1024fft, 160hop -> 101 frames
    expected_time_dim = (Config.AUDIO_LEN // Config.HOP_LENGTH) + 1

    assert features_train.dim() == 4, "Features should be 4D (B, C, F, T)"
    assert (
        features_train.shape[1] == 3
    ), "Should have 3 channels (Mel, Delta, DeltaDelta)"
    assert (
        features_train.shape[2] == Config.N_MELS
    ), f"Freq dim mismatch. Expected {Config.N_MELS}"
    # Allow slight variation in time dim due to padding/centering
    assert (
        abs(features_train.shape[3] - expected_time_dim) <= 2
    ), f"Time dim mismatch. Got {features_train.shape[3]}, expected ~{expected_time_dim}"
    assert isinstance(lam, float), "Mixup lambda should be a float"

    # B. Test Inference Transforms (No Mixup)
    features_val = transforms(gpu_waveforms, labels=None, train=False)
    assert (
        features_val.shape == features_train.shape
    ), "Inference shape should match training shape"

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = get_model(device=device)

    # Forward pass
    logits = model(features_val)
    print(f"    Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, {Config.NUM_CLASSES})"

    # Check Attentive Pooling internal logic (indirectly via forward pass success)
    # The model uses AttentivePooling which reduces (B, C, T) -> (B, C)
    # If forward pass works and outputs correct shape, pooling is functioning.

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch on Subset)...")

    ema = ModelEMA(model, decay=0.9, device=device)  # Low decay for demo visibility
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_loss, train_acc = train_epoch(
        model, ema, transforms, train_loader_small, criterion, optimizer, device
    )

    print(f"    Train Loss: {train_loss:.4f}")
    print(f"    Train Acc:  {train_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0 <= train_acc <= 1, "Accuracy should be between 0 and 1"

    # -------------------------------------------------------------------------
    # 6. Validation Loop Verification
    # -------------------------------------------------------------------------
    print("\n[6] Running Validation Loop...")

    val_loss, val_fine_acc, val_comp_acc = validate(
        ema.ema_model, transforms, val_loader_small, criterion, device
    )

    print(f"    Val Loss:            {val_loss:.4f}")
    print(f"    Val Fine-Grained Acc: {val_fine_acc:.4f}")
    print(f"    Val Competition Acc:  {val_comp_acc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"

    # -------------------------------------------------------------------------
    # 7. Model Saving & Loading
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Checkpointing...")

    save_path = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
    torch.save(ema.ema_model.state_dict(), save_path)
    assert os.path.exists(save_path), "Model file was not saved"

    # Load back
    loaded_model = get_model(device=device)
    loaded_model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    print("    Model saved and loaded successfully.")

    # -------------------------------------------------------------------------
    # 8. Inference & Submission Formatting
    # -------------------------------------------------------------------------
    print("\n[8] Simulating Test Inference & Submission Generation...")

    loaded_model.eval()
    predictions = []
    fnames = []

    with torch.no_grad():
        for waveforms, _, batch_fnames_list in test_loader_small:
            waveforms = waveforms.to(device)
            features = transforms(waveforms, train=False)
            outputs = loaded_model(features)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            fnames.extend(batch_fnames_list)

    # Map indices to labels
    pred_labels_fine = [Config.get_label_from_index(idx) for idx in predictions]
    pred_labels_coarse = map_fine_to_coarse(pred_labels_fine)

    # Create DataFrame
    submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels_coarse})

    print("    Sample Predictions:")
    print(submission_df.head())

    # Validate Submission Format
    assert "fname" in submission_df.columns
    assert "label" in submission_df.columns
    assert len(submission_df) == len(test_subset)

    # Check if labels are valid competition targets
    valid_targets = Config.TARGET_LABELS.union(
        {Config.SILENCE_LABEL, Config.UNKNOWN_LABEL}
    )
    assert (
        submission_df["label"].isin(valid_targets).all()
    ), "Submission contains invalid labels"

    # Save Submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")

    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
