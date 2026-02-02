import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
import library.dataset
import library.train
from library.config import AudioConfig, ModelConfig, TrainConfig, LabelConfig
from library.utils import (
    set_seed,
    LabelMapper,
    mixup_data,
    mixup_criterion,
    AverageMeter,
)
from library.dataset import SpeechCommandsDataset
from library.model import ContextAwareEfficientNet
from library.train import Trainer


def run_demonstration():
    print("===========================================================")
    print("       Speech Command Recognition - Library Demo           ")
    print("===========================================================")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("\n[1] Initializing Configuration and Seeds...")
    set_seed(42)

    # Modify TrainConfig for a fast demo run
    TrainConfig.work_dir = "./working/demo_run/"
    TrainConfig.checkpoint_path = os.path.join(
        TrainConfig.work_dir, "demo_best_model.pth"
    )
    TrainConfig.epochs = 1
    TrainConfig.batch_size = 4
    TrainConfig.target_samples = 10  # Minimal upsampling for speed

    # Ensure working directory exists
    os.makedirs(TrainConfig.work_dir, exist_ok=True)
    print("    TrainConfig updated for speed (Epochs: 1, Batch: 4).")

    # -------------------------------------------------------------------------
    # 2. Label Mapper Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying LabelMapper Logic...")
    mapper = LabelMapper()

    # Test Case A: Target Label
    target_label = "yes"
    mapped_target = mapper.map_to_submission(target_label)
    print(f"    Mapping '{target_label}' -> '{mapped_target}'")
    assert mapped_target == "yes", f"Error: '{target_label}' should map to 'yes'"

    # Test Case B: Auxiliary Label (should be unknown)
    aux_label = "bed"
    mapped_aux = mapper.map_to_submission(aux_label)
    print(f"    Mapping '{aux_label}' -> '{mapped_aux}'")
    assert mapped_aux == "unknown", f"Error: '{aux_label}' should map to 'unknown'"

    # Test Case C: Silence
    silence_label = "silence"
    mapped_silence = mapper.map_to_submission(silence_label)
    print(f"    Mapping '{silence_label}' -> '{mapped_silence}'")
    assert (
        mapped_silence == "silence"
    ), f"Error: '{silence_label}' should map to 'silence'"

    print("    LabelMapper verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Data Loading...")

    # Load a tiny subset of metadata manually for testing
    df_train_full = pd.read_csv("./metadata/train.csv")

    # Create a dummy 'fine_label' column as expected by the Dataset class logic
    # The dataset logic extracts it from filepath, but we can pre-process or rely on the extraction
    # The provided dataset code extracts it inside __getitem__ via helper or assumes it's there.
    # Looking at library/dataset.py: get_dataloaders adds 'fine_label'.
    # SpeechCommandsDataset uses row["fine_label"] in __getitem__.
    # So we must add it.
    def extract_label_demo(filepath):
        parts = filepath.split("/")
        if len(parts) >= 2:
            lbl = parts[-2]
            return "silence" if lbl == "_background_noise_" else lbl
        return "unknown"

    df_sample = df_train_full.sample(n=10, random_state=42).copy()
    df_sample["fine_label"] = df_sample["filepath"].apply(extract_label_demo)

    # Instantiate Dataset
    dataset = SpeechCommandsDataset(df_sample, phase="train")
    print(f"    Created dataset with {len(dataset)} samples.")

    # Test __getitem__
    spec, label_idx = dataset[0]
    print(f"    Sample 0 - Spectrogram Shape: {spec.shape}, Label Index: {label_idx}")

    # Assertions
    # Expected shape: (1, n_mels, time_steps) -> (1, 128, 101) for 1s audio @ 16k with hop 160
    # 16000 / 160 = 100 frames + 1 = 101
    assert spec.shape == (1, 128, 101), f"Spectrogram shape mismatch. Got {spec.shape}"
    assert isinstance(label_idx, torch.Tensor), "Label should be a tensor"

    print("    Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying ContextAwareEfficientNet Model...")

    model = ContextAwareEfficientNet()
    model.eval()

    # Create dummy input batch (Batch=2, Channels=1, Freq=128, Time=101)
    dummy_input = torch.randn(2, 1, 128, 101)

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"    Input Shape: {dummy_input.shape}")
    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    # Output should be (Batch, Num_Classes) -> (2, 31)
    assert logits.shape == (
        2,
        31,
    ), f"Model output shape mismatch. Expected (2, 31), got {logits.shape}"

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Trainer Integration (Mocked Data)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Mocked DataLoaders)...")

    # Monkey-patch library.train.get_dataloaders to avoid loading the full dataset
    # This allows us to use the Trainer class without waiting for full data processing.
    def mock_get_dataloaders(load_cached_data=True):
        print("    [Mock] Creating small dataloaders for demo...")

        # Create small DataFrames
        df_t = df_train_full.sample(n=16, random_state=42).copy()
        df_v = df_train_full.sample(
            n=8, random_state=42
        ).copy()  # Use train as val for demo

        df_t["fine_label"] = df_t["filepath"].apply(extract_label_demo)
        df_v["fine_label"] = df_v["filepath"].apply(extract_label_demo)

        # Create Datasets
        ds_train = SpeechCommandsDataset(df_t, phase="train")
        ds_val = SpeechCommandsDataset(df_v, phase="val")

        # Create Loaders
        dl_train = DataLoader(ds_train, batch_size=4, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=4, shuffle=False)

        return dl_train, dl_val, None  # Test loader not needed for this check

    # Apply patch
    original_get_dataloaders = library.train.get_dataloaders
    library.train.get_dataloaders = mock_get_dataloaders

    try:
        # Instantiate Trainer
        trainer = Trainer()

        # Run one epoch of training
        print("    Running train_one_epoch(0)...")
        train_loss = trainer.train_one_epoch(0)
        print(f"    Train Loss: {train_loss:.4f}")
        assert isinstance(train_loss, float), "Train loss should be a float"
        assert train_loss > 0, "Train loss should be positive"

        # Run validation
        print("    Running validate()...")
        val_acc = trainer.validate()
        print(f"    Validation Accuracy: {val_acc:.2f}%")
        assert 0 <= val_acc <= 100, "Accuracy should be between 0 and 100"

        print("    Trainer verification passed.")

    finally:
        # Restore original function just in case
        library.train.get_dataloaders = original_get_dataloaders

    print("\n===========================================================")
    print("       Demonstration Completed Successfully                ")
    print("===========================================================")


if __name__ == "__main__":
    run_demonstration()
