import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import (
    train_config,
    path_config,
    audio_config,
    label_config,
    model_config,
)
from library.utils import set_seed, LabelManager
from library.dataset import get_train_val_datasets, get_test_dataset
from library.model import DilatedEfficientNet
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing Configuration...")

    # Set seed for reproducibility
    set_seed(42)

    # Modify configs for a fast demo execution
    train_config.epochs = 1
    train_config.batch_size = 8
    train_config.num_workers = 2
    train_config.early_stopping_patience = 1

    # Ensure working directory exists
    os.makedirs(path_config.working_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # -------------------------------------------------------------------------
    # 2. Label Manager
    # -------------------------------------------------------------------------
    print("\nInitializing Label Manager...")
    # We load cached data if available, or scan if not.
    # For demo purposes, we allow it to scan.
    label_manager = LabelManager(load_cached_data=True)

    num_classes = label_manager.get_num_classes()
    print(f"Total fine-grained classes: {num_classes}")

    # Verification: Check a known target label and an auxiliary label
    assert "left" in label_manager.classes
    assert (
        "bird" in label_manager.classes or "bed" in label_manager.classes
    )  # Common aux labels

    # -------------------------------------------------------------------------
    # 3. Dataset Loading & Subsetting
    # -------------------------------------------------------------------------
    print("\nLoading Datasets...")
    # get_train_val_datasets returns SpeechCommandsDataset objects
    train_dataset, val_dataset = get_train_val_datasets(
        label_manager, load_cached_data=True
    )

    print(f"Original Train Size: {len(train_dataset)}")
    print(f"Original Val Size:   {len(val_dataset)}")

    # SUBSET DATASETS FOR SPEED
    # We manually slice the internal dataframe to reduce the number of samples
    # This ensures the epoch finishes in seconds rather than minutes.
    train_dataset.df = train_dataset.df.head(64).reset_index(drop=True)
    val_dataset.df = val_dataset.df.head(32).reset_index(drop=True)

    print(f"Subset Train Size: {len(train_dataset)}")
    print(f"Subset Val Size:   {len(val_dataset)}")

    # Verify Data Loading
    sample_spec, sample_label_idx = train_dataset[0]
    print(f"Sample Spectrogram Shape: {sample_spec.shape}")

    # Expected shape: (1, n_mels, time_steps)
    # Time steps depends on duration (1.0s) and hop_length (160) -> 16000/160 = 100 frames (+1 usually)
    assert sample_spec.dim() == 3
    assert sample_spec.shape[0] == 1
    assert sample_spec.shape[1] == audio_config.n_mels

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = DilatedEfficientNet(num_classes=num_classes)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 1, audio_config.n_mels, 101).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (2, num_classes)
    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop (Demo)...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Run training (1 epoch as configured above)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 6. Inference & Label Mapping Verification
    # -------------------------------------------------------------------------
    print("\nVerifying Inference and Label Mapping...")

    # Simulate a prediction
    model.eval()

    # Let's pretend the model predicted index 5
    fake_pred_idx = 5

    # 1. Convert Index -> Fine-Grained Label
    fine_label = label_manager.convert_idx_to_label(fake_pred_idx)
    print(f"Predicted Index: {fake_pred_idx} -> Fine Label: {fine_label}")

    # 2. Convert Fine-Grained Label -> Submission Label
    submission_label = label_manager.map_to_submission_label(fine_label)
    print(f"Fine Label: {fine_label} -> Submission Label: {submission_label}")

    # Verify logic for specific cases
    # Case A: Target Label (e.g., 'up')
    if "up" in label_manager.classes:
        idx_up = label_manager.convert_label_to_idx("up")
        lbl_up = label_manager.convert_idx_to_label(idx_up)
        sub_up = label_manager.map_to_submission_label(lbl_up)
        assert sub_up == "up", f"Expected 'up', got {sub_up}"

    # Case B: Auxiliary Label (e.g., 'bed' or 'bird') -> Should map to 'unknown'
    # Find a label that is NOT in target_labels and NOT silence
    aux_candidates = [
        c
        for c in label_manager.classes
        if c not in label_config.target_labels and c != label_config.silence_label
    ]

    if aux_candidates:
        aux_lbl = aux_candidates[0]
        sub_aux = label_manager.map_to_submission_label(aux_lbl)
        assert sub_aux == "unknown", f"Expected 'unknown' for {aux_lbl}, got {sub_aux}"
        print(f"Verified mapping: {aux_lbl} -> {sub_aux}")

    # Case C: Silence
    silence_lbl = label_config.silence_label
    sub_silence = label_manager.map_to_submission_label(silence_lbl)
    assert sub_silence == "silence", f"Expected 'silence', got {sub_silence}"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
