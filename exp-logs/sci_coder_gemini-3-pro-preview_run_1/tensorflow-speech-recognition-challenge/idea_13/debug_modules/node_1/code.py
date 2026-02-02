import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    SAMPLE_RATE,
    AUDIO_LEN,
    N_MELS,
    WORKING_DIR,
)
from library.utils import set_seed, Mixup
from library.model import get_model
from library.dataset import SpeechDataset
from library.engine import (
    train_model,
    generate_submission,
    generate_pseudo_labels,
    load_noise_files,
)


def run_demo():
    print("Initializing Demo Run...")

    # 1. Setup
    DEMO_DIR = os.path.join(os.path.dirname(WORKING_DIR), "demo_run")
    os.makedirs(DEMO_DIR, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Preparation (Subset for Speed)
    print("\nPreparing Data...")

    # Load full metadata
    df_train_full = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(VAL_METADATA_PATH)

    # Sample subsets for the demo to ensure quick execution
    # We ensure we have at least some samples
    df_train_small = df_train_full.sample(
        n=min(128, len(df_train_full)), random_state=42
    ).reset_index(drop=True)
    df_val_small = df_val_full.sample(
        n=min(64, len(df_val_full)), random_state=42
    ).reset_index(drop=True)

    print(f"Training subset size: {len(df_train_small)}")
    print(f"Validation subset size: {len(df_val_small)}")

    # Load noise files for augmentation
    noise_files = load_noise_files()
    print(f"Loaded {len(noise_files)} background noise files.")

    # Instantiate Datasets
    # Note: We pass noise_files to train dataset for augmentation
    train_ds = SpeechDataset(df_train_small, mode="train", noise_files=noise_files)
    val_ds = SpeechDataset(df_val_small, mode="val")

    # Verify Dataset Output
    sample_spec, sample_label = train_ds[0]
    print(f"Spectrogram Shape: {sample_spec.shape}")
    assert sample_spec.dim() == 3, "Spectrogram must be 3D (Channels, Freq, Time)"
    assert sample_spec.shape[0] == 1, "Input channels should be 1"
    assert sample_spec.shape[1] == N_MELS, f"Frequency dimension should be {N_MELS}"

    # Create DataLoaders
    # Reduced batch size for demo if needed, but 128 fits on A100 easily
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,  # False for demo to ensure we process all data
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = get_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 1, N_MELS, int(AUDIO_LEN / 160) + 1).to(
        device
    )  # approx time dim
    # Actually, let's use the real shape from the dataset
    dummy_input = sample_spec.unsqueeze(0).repeat(2, 1, 1, 1).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, NUM_CLASSES), f"Expected output shape (2, {NUM_CLASSES})"

    # 4. Training Loop
    print("\nStarting Training Demo...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    mixup = Mixup(alpha=1.0)

    # Train for 2 epochs to demonstrate the loop
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=2,
        patience=2,
        mixup_fn=mixup,
    )

    # Save the demo model
    demo_model_path = os.path.join(DEMO_DIR, "demo_best_model.pth")
    torch.save(trained_model.state_dict(), demo_model_path)
    print(f"Demo model saved to {demo_model_path}")

    # 5. Pseudo-Labeling Demonstration
    print("\nGenerating Pseudo Labels (Demo)...")
    # We use the trained model to generate pseudo labels for the test set
    # Note: In a real scenario, this would be done after full training
    df_pseudo = generate_pseudo_labels(trained_model, device, confidence_threshold=0.8)
    print(f"Generated {len(df_pseudo)} pseudo-labels.")
    if not df_pseudo.empty:
        print("Sample Pseudo-labels:")
        print(df_pseudo.head())

    # 6. Submission Generation
    print("\nGenerating Submission...")
    submission_path = os.path.join(DEMO_DIR, "demo_submission.csv")
    generate_submission(trained_model, device, output_path=submission_path)

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not created"
    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created with {len(df_sub)} rows.")
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"

    print("\nDemo Run Completed Successfully.")


if __name__ == "__main__":
    run_demo()
