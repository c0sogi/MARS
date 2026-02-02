import os
import sys
import pandas as pd
import torch
import numpy as np

# Ensure library modules can be imported
sys.path.append(".")

from library import config
from library import utils
from library import dataset
from library import model
from library import trainer


def run_demonstration():
    print("=== Starting Speech Command Recognition Demo ===\n")

    # 1. Configuration and Seeding
    print("[1] Setting up configuration...")
    config.set_seed(42)
    print(f"    Device: {config.DEVICE}")
    print(f"    Work Directory: {config.WORK_DIR}")

    # 2. Demonstrate Utils (Audio Loading and Featurization)
    print("\n[2] Demonstrating Audio Processing (library.utils)...")

    # Read metadata to get a valid file path
    train_meta_path = os.path.join(config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    df_train_full = pd.read_csv(train_meta_path)
    sample_filepath = df_train_full.iloc[0]["filepath"]
    print(f"    Loading sample: {sample_filepath}")

    # Load and pad audio
    waveform = utils.load_and_pad_audio(sample_filepath)
    print(f"    Waveform shape: {waveform.shape}")

    # Verify Waveform
    assert isinstance(waveform, torch.Tensor), "Waveform must be a torch.Tensor"
    assert waveform.shape == (
        1,
        config.AUDIO_LEN,
    ), f"Waveform shape mismatch. Expected (1, {config.AUDIO_LEN}), got {waveform.shape}"

    # Generate Spectrogram
    featurizer = utils.get_featurizer()
    spec = featurizer(waveform)
    print(f"    Spectrogram shape: {spec.shape}")

    # Verify Spectrogram (1, n_mels, time)
    # Time dimension depends on audio length and hop size. For 16000 samples and hop 160, it's usually 101.
    assert (
        spec.shape[1] == config.N_MELS
    ), f"Spectrogram must have {config.N_MELS} mel bins"

    # 3. Demonstrate Dataset (library.dataset)
    print("\n[3] Demonstrating Dataset (library.dataset)...")

    # Use a small subset for speed
    subset_size = 200
    df_train_subset = df_train_full.head(subset_size).copy()
    print(f"    Creating dataset from top {subset_size} training samples...")

    train_dataset = dataset.SpeechCommandsDataset(df_train_subset, phase="train")

    # Verify Dataset
    print(f"    Dataset length: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty"

    # Fetch one item
    sample_spec, sample_label = train_dataset[0]
    print(f"    Item[0] Spectrogram shape: {sample_spec.shape}")
    print(f"    Item[0] Label index: {sample_label}")

    assert (
        sample_spec.dim() == 3
    ), "Dataset item spectrogram should be 3D (1, n_mels, time)"
    assert isinstance(sample_label, (int, np.integer)), "Label must be an integer"
    assert (
        0 <= sample_label < config.NUM_CLASSES
    ), f"Label {sample_label} out of bounds (0-{config.NUM_CLASSES-1})"

    # 4. Demonstrate Model (library.model)
    print("\n[4] Demonstrating Model (library.model)...")

    net = model.SimpleAudioCNN(num_classes=config.NUM_CLASSES)
    net.to(config.DEVICE)

    # Create dummy input batch (Batch Size, 1, n_mels, time)
    dummy_input = sample_spec.unsqueeze(0).to(config.DEVICE)
    print(f"    Forward pass input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = net(dummy_input)

    print(f"    Output logits shape: {output.shape}")
    assert output.shape == (1, config.NUM_CLASSES), "Model output shape mismatch"

    # 5. Demonstrate Trainer (library.trainer)
    print("\n[5] Demonstrating Training Loop (library.trainer)...")

    # Setup DataLoaders for the subset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    # Use a tiny validation set
    val_meta_path = os.path.join(config.METADATA_DIR, "val.csv")
    df_val_subset = pd.read_csv(val_meta_path).head(50)
    val_dataset = dataset.SpeechCommandsDataset(df_val_subset, phase="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Setup Training Components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    save_path = os.path.join(config.WORK_DIR, "demo_model.pth")

    # Initialize Trainer
    trainer_instance = trainer.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        save_path=save_path,
    )

    # Run 1 Epoch
    print("    Running 1 training epoch...")
    train_loss, train_acc = trainer_instance.train_epoch()
    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    print("    Running validation...")
    val_loss, val_acc = trainer_instance.validate()
    print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= train_acc <= 1.0, "Training accuracy out of range"

    # 6. Demonstrate Submission Generation
    print("\n[6] Demonstrating Submission Generation...")

    # We use the model we just 'trained' (even if for just 1 epoch on a subset)
    # generate_submission loads test.csv automatically
    trainer.generate_submission(net, batch_size=64)

    # Verify Submission File
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"    Submission file generated at: {config.SUBMISSION_PATH}")
        print(f"    Submission shape: {sub_df.shape}")

        # Check against test metadata
        test_meta_path = os.path.join(config.METADATA_DIR, "test.csv")
        df_test = pd.read_csv(test_meta_path)

        assert len(sub_df) == len(
            df_test
        ), "Submission row count does not match test set"
        assert list(sub_df.columns) == ["fname", "label"], "Submission columns mismatch"
        print("    Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
