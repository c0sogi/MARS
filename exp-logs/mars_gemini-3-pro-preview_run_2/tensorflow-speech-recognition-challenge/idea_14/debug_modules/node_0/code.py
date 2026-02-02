import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.data_utils import load_background_noise, load_all_data, set_seed
from library.custom_layers import (
    GPUNoiseInjector,
    DifferentiableSpectrogram,
    SpecAugment,
    AttentionPooling,
)
from library.network import EfficientNetV2Audio
from library.trainer import Trainer


def run_demo():
    # ==========================================
    # 0. Setup and Configuration Overrides
    # ==========================================
    print("=== Setting up Demo Configuration ===")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 1. Verify Data Loading
    # ==========================================
    print("\n=== Verifying Data Loading ===")

    # Test Background Noise Loading
    noise_tensor = load_background_noise(load_cached_data=False)
    print(f"Background noise loaded. Shape: {noise_tensor.shape}")
    assert noise_tensor.ndim == 1, "Background noise should be a 1D tensor"
    assert noise_tensor.numel() > 0, "Background noise tensor is empty"

    # Test Training Data Loading (Debug Subset)
    train_wavs, train_lbls = load_all_data("train", load_cached_data=False)

    # Apply debug slicing manually if load_all_data didn't (it usually loads full then trainer slices,
    # but let's verify what we got).
    # Note: load_all_data caches the full set, trainer slices it.
    # For this unit test, we just check if we got data.
    print(f"Train waveforms shape: {train_wavs.shape}")
    print(f"Train labels shape: {train_lbls.shape}")

    assert train_wavs.ndim == 2, "Waveforms should be 2D (N, Samples)"
    assert (
        train_wavs.shape[1] == Config.NUM_SAMPLES
    ), f"Waveforms should have length {Config.NUM_SAMPLES}"
    assert len(train_wavs) == len(
        train_lbls
    ), "Mismatch between waveforms and labels count"

    # ==========================================
    # 2. Verify Custom Layers
    # ==========================================
    print("\n=== Verifying Custom Layers ===")

    # Dummy input: Batch of 4, 1 second of audio
    dummy_wavs = torch.randn(4, Config.NUM_SAMPLES).to(device)

    # A. Noise Injector
    # Create a small dummy noise buffer
    dummy_noise = torch.randn(Config.NUM_SAMPLES * 10).to(device)
    injector = GPUNoiseInjector(dummy_noise, p=1.0).to(device)
    injector.train()  # Enable augmentation

    noisy_wavs = injector(dummy_wavs)
    assert noisy_wavs.shape == dummy_wavs.shape, "Noise injector changed output shape"
    assert not torch.allclose(
        noisy_wavs, dummy_wavs
    ), "Noise injector did not modify the signal (p=1.0)"
    print("GPUNoiseInjector: Passed")

    # B. Differentiable Spectrogram
    spec_layer = DifferentiableSpectrogram().to(device)
    specs = spec_layer(dummy_wavs)
    # Expected shape: (B, 1, F, T)
    # F = n_mels = 128
    # T = (NUM_SAMPLES / HOP_LENGTH) + 1 approx. 16000/160 + 1 = 101
    print(f"Spectrogram shape: {specs.shape}")
    assert specs.ndim == 4, "Spectrogram should be 4D (B, C, F, T)"
    assert specs.shape[1] == 1, "Spectrogram should have 1 channel"
    assert (
        specs.shape[2] == Config.N_MELS
    ), f"Spectrogram F dim should be {Config.N_MELS}"
    print("DifferentiableSpectrogram: Passed")

    # C. SpecAugment
    aug_layer = SpecAugment().to(device)
    aug_layer.train()
    aug_specs = aug_layer(specs)
    assert aug_specs.shape == specs.shape, "SpecAugment changed output shape"
    # It's hard to assert values changed deterministically due to randomness, but shape check is good.
    print("SpecAugment: Passed")

    # D. Attention Pooling
    # Input to pooling is (B, C_backbone, H, W)
    # Let's assume C=32, H=4, W=4
    dummy_features = torch.randn(4, 32, 4, 4).to(device)
    pool_layer = AttentionPooling(in_channels=32).to(device)
    pooled = pool_layer(dummy_features)
    # Expected output: (B, C)
    assert pooled.shape == (
        4,
        32,
    ), f"AttentionPooling output shape mismatch: {pooled.shape}"
    print("AttentionPooling: Passed")

    # ==========================================
    # 3. Verify Full Network
    # ==========================================
    print("\n=== Verifying EfficientNetV2Audio Model ===")

    model = EfficientNetV2Audio(
        background_noise=dummy_noise,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # False for speed
    ).to(device)
    model.train()

    logits = model(dummy_wavs)
    print(f"Model Logits Shape: {logits.shape}")

    assert logits.shape == (
        4,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (4, {Config.NUM_CLASSES}), got {logits.shape}"
    print("EfficientNetV2Audio: Passed")

    # ==========================================
    # 4. Verify Trainer (Fit & Predict)
    # ==========================================
    print("\n=== Verifying Trainer Pipeline ===")

    trainer = Trainer()

    # Run Training
    # This uses the DEBUG subset defined in Config
    print("Running Trainer.fit()...")
    trainer.fit()

    # Check if best model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training finished and checkpoint verified.")

    # Run Inference
    print("Running Trainer.predict_and_submit()...")
    trainer.predict_and_submit()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Verify submission format
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # Verify labels are valid
    valid_labels = set(Config.LABELS)
    predicted_labels = set(df_sub["label"].unique())
    assert predicted_labels.issubset(valid_labels), "Submission contains invalid labels"

    print("Trainer Pipeline: Passed")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
