import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, set_seed
from library.utils import get_metadata, FineGrainedLabelEncoder
from library.dataset import SpeechCommandDataset
from library.model import EnergyGatedEfficientNet
from library.trainer import Trainer


def run_demo():
    # ---------------------------------------------------------
    # 1. Configuration and Setup
    # ---------------------------------------------------------
    print("Initializing Demo Configuration...")

    # Override Config paths for the demo to keep things clean
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")
    Config.BATCH_SIZE = 16  # Small batch size for demo
    Config.NUM_WORKERS = 2

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Preparation (Metadata & Encoder)
    # ---------------------------------------------------------
    print("Loading Metadata...")
    df_train_full = get_metadata("train")
    df_val_full = get_metadata("val")
    df_test_full = get_metadata("test")

    # Fit Encoder on full training data to establish correct class mappings
    print("Fitting Label Encoder...")
    label_encoder = FineGrainedLabelEncoder()
    label_encoder.fit(df_train_full)

    num_classes = len(label_encoder)
    print(f"Total Fine-Grained Classes: {num_classes}")

    # Extract noise files for silence generation
    noise_df = df_train_full[df_train_full["label"] == "silence"]
    noise_files = [
        os.path.join(Config.INPUT_ROOT, f) for f in noise_df["filepath"].tolist()
    ]
    noise_files = [f for f in noise_files if os.path.exists(f)]
    print(f"Found {len(noise_files)} background noise files.")

    # ---------------------------------------------------------
    # 3. Subsampling for Speed (Demo Requirement)
    # ---------------------------------------------------------
    print("Subsampling datasets for rapid execution...")
    # We take a small random sample to ensure the script finishes quickly
    df_train_sample = df_train_full.sample(n=128, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_val_sample = df_val_full.sample(n=32, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_test_sample = df_test_full.sample(n=32, random_state=Config.SEED).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # 4. Dataset and DataLoader Instantiation
    # ---------------------------------------------------------
    print("Creating Datasets and Loaders...")

    train_dataset = SpeechCommandDataset(
        df_train_sample, label_encoder, mode="train", noise_files=noise_files
    )
    val_dataset = SpeechCommandDataset(
        df_val_sample, label_encoder, mode="val", noise_files=noise_files
    )
    test_dataset = SpeechCommandDataset(
        df_test_sample, label_encoder, mode="test", noise_files=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check batch structure
    dummy_batch = next(iter(train_loader))
    specs, energies, labels = dummy_batch

    # Expected shapes:
    # Spec: (B, 1, n_mels, time) -> (16, 1, 128, 101) for 1 sec audio @ 16kHz with hop 160
    # Energy: (B, time) -> (16, 101)
    # Labels: (B) -> (16)

    print(
        f"Batch Shapes - Spec: {specs.shape}, Energy: {energies.shape}, Labels: {labels.shape}"
    )
    assert specs.dim() == 4, "Spectrogram should be 4D (B, C, F, T)"
    assert energies.dim() == 2, "Energy should be 2D (B, T)"
    assert labels.dim() == 1, "Labels should be 1D (B)"
    assert specs.shape[0] == Config.BATCH_SIZE

    # ---------------------------------------------------------
    # 5. Model Initialization & Verification
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = EnergyGatedEfficientNet(num_classes=num_classes)

    # Move to device for verification
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Dummy Forward Pass
    print("Verifying Forward Pass...")
    dummy_spec = specs.to(device)
    # Energy needs to be (B, 1, T) for the model input,
    # but dataset returns (B, T). The Trainer handles this, or the model expects (B, 1, T).
    # Checking model.py: forward(x, energy) -> att_pool(features, energy).
    # att_pool expects energy shape (B, 1, T_orig).
    # Dataset returns energy.squeeze(0) -> (T). So batch is (B, T).
    # We need to unsqueeze dim 1 before passing to model.
    dummy_energy = energies.unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(dummy_spec, dummy_energy)

    assert logits.shape == (
        Config.BATCH_SIZE,
        num_classes,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, {num_classes}), got {logits.shape}"
    print("Model verification successful.")

    # ---------------------------------------------------------
    # 6. Training Loop
    # ---------------------------------------------------------
    print("Starting Training Demo...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, label_encoder)

    # Train for 2 epochs to demonstrate the loop
    trainer.fit(epochs=2)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file 'best_model.pth' was not created."
    print("Training complete. Checkpoint verified.")

    # ---------------------------------------------------------
    # 7. Inference & Submission
    # ---------------------------------------------------------
    print("Generating Predictions...")
    trainer.predict()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df_sub)}")

    # Check format
    assert list(df_sub.columns) == [
        "fname",
        "label",
    ], "Submission columns are incorrect."
    assert len(df_sub) == len(
        df_test_sample
    ), f"Submission length mismatch. Expected {len(df_test_sample)}, got {len(df_sub)}"

    # Check label validity (should be mapped to competition targets)
    valid_labels = set(Config.OUTPUT_LABELS)
    pred_labels = set(df_sub["label"].unique())
    invalid_preds = pred_labels - valid_labels
    assert not invalid_preds, f"Found invalid labels in submission: {invalid_preds}"

    print("Demo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
