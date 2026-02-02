import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import (
    set_seed,
    get_fine_grained_labels,
    get_label_map,
    save_submission,
)
from library.audio_transforms import AudioProcessor
from library.dataset import SpeechDataset
from library.model import DilatedEfficientNet
from library.sam import SAM
from library.engine import train_one_epoch, evaluate


def run_demo():
    print("=== Starting Machine Learning Task Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Setting up environment...")
    set_seed(42)

    # Override Config for the demo to run fast
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # 2. Verify Audio Processing
    print("\n[2] Verifying AudioProcessor...")
    processor = AudioProcessor()

    # Find a real file to test
    train_csv_path = Config.TRAIN_CSV
    if os.path.exists(train_csv_path):
        df_train = pd.read_csv(train_csv_path)
        sample_file = os.path.join(Config.INPUT_ROOT, df_train.iloc[0]["filepath"])

        # Test processing
        spec = processor.process_audio(
            sample_file, is_training=True, should_augment=True
        )
        print(f"    Processed Spectrogram Shape: {spec.shape}")

        # Assertions
        # Expected shape: (1, n_mels, time_steps)
        # time_steps = sample_rate * duration / hop_length + 1 approx
        # 16000 * 1.0 / 160 = 100 frames. Torchaudio might produce 101 depending on center padding.
        assert spec.dim() == 3, "Spectrogram must be 3D (C, F, T)"
        assert spec.shape[0] == 1, "Channel dimension should be 1"
        assert (
            spec.shape[1] == Config.N_MELS
        ), f"Freq dimension should be {Config.N_MELS}"
        print("    AudioProcessor logic verified.")
    else:
        print("    Warning: Train metadata not found, skipping specific file check.")

    # 3. Verify Dataset and DataLoader
    print("\n[3] Verifying Dataset and DataLoader...")

    # Initialize Datasets
    # We load cached data if available, but for the demo we will slice it immediately
    train_dataset = SpeechDataset(split="train", mode="train", load_cached_data=False)
    val_dataset = SpeechDataset(split="val", mode="infer", load_cached_data=False)

    print(f"    Original Train Size: {len(train_dataset)}")

    # OPTIMIZATION: Truncate datasets to 40 samples for speed
    train_dataset.data = train_dataset.data[:40]
    val_dataset.data = val_dataset.data[:20]
    print(f"    Truncated Train Size: {len(train_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    inputs, targets, fnames = next(iter(train_loader))
    print(f"    Batch Input Shape: {inputs.shape}")
    print(f"    Batch Target Shape: {targets.shape}")

    assert inputs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert inputs.shape[1] == 1, "Input channel mismatch"
    print("    Dataset and DataLoader verified.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    fine_labels = get_fine_grained_labels()
    num_classes = len(fine_labels)
    print(f"    Number of classes: {num_classes}")

    model = DilatedEfficientNet(num_classes=num_classes)
    model.to(device)

    # Forward pass with the batch fetched earlier
    inputs = inputs.to(device)
    outputs = model(inputs)
    print(f"    Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        Config.BATCH_SIZE,
        num_classes,
    ), "Model output shape incorrect"
    print("    Model architecture verified.")

    # 5. Verify Optimizer (SAM)
    print("\n[5] Verifying SAM Optimizer...")
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy step
    targets = targets.to(device)

    # Step 1: Ascent
    output1 = model(inputs)
    loss1 = criterion(output1, targets)
    loss1.backward()
    optimizer.first_step(zero_grad=True)

    # Step 2: Descent
    output2 = model(inputs)
    loss2 = criterion(output2, targets)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    print("    SAM Optimizer step executed successfully.")

    # 6. Verify Training Engine (Mini-Loop)
    print("\n[6] Running Training Engine Demo (1 Epoch on truncated data)...")

    # We use the engine functions but with our truncated loaders
    start_time = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )

    val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch=1)

    print(f"    Demo Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"    Demo Val Loss:   {val_loss:.4f}   | Acc: {val_acc:.4f}")
    print(f"    Execution time: {time.time() - start_time:.2f}s")

    assert train_loss > 0, "Training loss should be positive"
    print("    Training engine verified.")

    # 7. Verify Submission Logic
    print("\n[7] Verifying Submission Logic...")

    # Get Label Map
    label_map = get_label_map()
    print(f"    Label Map Sample: {list(label_map.items())[:3]}...")

    # Simulate predictions
    # We'll use the validation batch we have
    with torch.no_grad():
        outputs = model(inputs)  # inputs from step 3
        _, preds = torch.max(outputs, 1)

    # Convert indices to fine-grained labels
    idx_to_label = train_dataset.idx_to_label
    fine_preds = [idx_to_label[p.item()] for p in preds]

    # Map to competition labels
    comp_preds = [label_map.get(fp, "unknown") for fp in fine_preds]

    # Save submission
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(comp_preds, fnames, demo_submission_path)

    assert os.path.exists(demo_submission_path), "Submission file was not created"

    # Check content
    df_sub = pd.read_csv(demo_submission_path)
    print(f"    Submission file created with {len(df_sub)} rows.")
    print(f"    First few rows:\n{df_sub.head()}")

    assert len(df_sub) == Config.BATCH_SIZE, "Submission row count mismatch"
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"
    print("    Submission logic verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
