import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import SpeechCommandDataset, MixUpCollate
from library.model import EfficientNetAudio
from library.trainer import Trainer


def run_demo():
    print("=== Starting Audio Classification Pipeline Demo ===")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly for this run
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set Seed
    set_seed(Config.SEED)
    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Prepare Data Subsets
    print("\n--- 1. Data Loading & Verification ---")

    # Load metadata
    try:
        df_train_full = pd.read_csv(Config.TRAIN_CSV)
        df_val_full = pd.read_csv(Config.VAL_CSV)
        df_test_full = pd.read_csv(Config.TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    # Sample subsets for speed
    df_train_sub = df_train_full.sample(n=32, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_val_sub = df_val_full.sample(n=16, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_test_sub = df_test_full.sample(n=16, random_state=Config.SEED).reset_index(
        drop=True
    )

    print(
        f"Subset sizes -> Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # Instantiate Datasets
    train_dataset = SpeechCommandDataset(df_train_sub, mode="train")
    val_dataset = SpeechCommandDataset(df_val_sub, mode="val")
    test_dataset = SpeechCommandDataset(df_test_sub, mode="test")

    # Verify single item structure
    spec, label_id = train_dataset[0]
    print(f"Single item shape: {spec.shape}, Label ID: {label_id}")

    # Assertions for shape: (1, n_mels, time_steps)
    # Time steps = 1 + (16000 / 160) = 101 for 1 sec audio
    assert spec.dim() == 3, "Spectrogram must be 3D (C, F, T)"
    assert spec.shape[0] == 1, "Channel dimension must be 1"
    assert spec.shape[1] == Config.N_MELS, f"Freq dimension must be {Config.N_MELS}"

    # Create DataLoaders
    # Note: Train loader uses MixUpCollate
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MixUpCollate(
            alpha=Config.MIXUP_ALPHA, num_classes=Config.NUM_CLASSES
        ),
        num_workers=0,  # Use 0 workers for simple demo script stability
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Structure
    batch_x, batch_y = next(iter(train_loader))
    print(f"Train Batch X shape: {batch_x.shape}")
    print(f"Train Batch Y shape: {batch_y.shape}")

    # Assertions
    assert batch_x.shape == (Config.BATCH_SIZE, 1, Config.N_MELS, 101)
    # MixUp produces soft targets: (Batch, NumClasses)
    assert batch_y.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    # 3. Model Initialization
    print("\n--- 2. Model Initialization ---")
    # Use pretrained=False to avoid downloading weights during this demo
    model = EfficientNetAudio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = batch_x.to(Config.DEVICE)
        dummy_output = model(dummy_input)

    print(f"Model Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    # 4. Training Loop
    print("\n--- 3. Training Loop Execution ---")
    trainer = Trainer(model, train_loader, val_loader, device=Config.DEVICE)

    # Run fit (1 epoch as configured)
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Success: Model checkpoint saved at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # 5. Inference
    print("\n--- 4. Inference & Submission ---")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Predict
    fnames, predicted_labels = trainer.predict(test_loader)

    print(f"Predictions generated: {len(predicted_labels)}")
    print(f"Sample prediction: {fnames[0]} -> {predicted_labels[0]}")

    assert len(fnames) == len(df_test_sub)
    assert len(predicted_labels) == len(df_test_sub)

    # Generate Submission CSV
    submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Success: Submission file saved at {Config.SUBMISSION_PATH}")
        print("Head of submission:")
        print(submission_df.head())
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
