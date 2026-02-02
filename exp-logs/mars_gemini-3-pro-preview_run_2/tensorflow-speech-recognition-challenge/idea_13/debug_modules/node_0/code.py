import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import TrainConfig, PathConfig, AudioConfig, ModelConfig
from library.dataset import SpeechCommandsDataset, get_dataloaders
from library.model import AudioEfficientNetV2
from library.trainer import Trainer
from library.utils import set_seed


def run_demo():
    print("=== Starting Speech Commands Recognition Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("[1] Overriding TrainConfig for fast demonstration...")
    # We modify the class attributes directly so that when Trainer/Dataset
    # instantiate TrainConfig(), they get these values.
    TrainConfig.debug = True  # Limit dataset to 1000 samples
    TrainConfig.epochs = 2  # Run only 2 epochs
    TrainConfig.batch_size = 16  # Small batch size
    TrainConfig.num_workers = 2  # Reduce overhead
    TrainConfig.early_stopping_patience = 2

    # Ensure working directory exists and is clean for this run
    path_config = PathConfig()
    if os.path.exists(path_config.working_dir):
        shutil.rmtree(path_config.working_dir)
    os.makedirs(path_config.working_dir, exist_ok=True)

    # Set global seed
    set_seed(TrainConfig.seed)

    # ---------------------------------------------------------
    # 2. Verify Dataset and Preprocessing
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and Audio Processing...")
    # Instantiate dataset in debug mode
    ds_train = SpeechCommandsDataset(split="train", transform=True, debug=True)

    # Check length
    assert (
        len(ds_train) <= TrainConfig.debug_sample_size
    ), "Dataset debug limiting failed"
    print(f"    Train dataset size (debug): {len(ds_train)}")

    # Check item structure
    spec, label = ds_train[0]
    print(f"    Spectrogram shape: {spec.shape}")
    print(f"    Label index: {label}")

    # Validation: Spectrogram shape
    # Expected: (1, n_mels, time_frames)
    # 16000 samples / 160 hop_length = 100 frames + 1 = 101
    audio_cfg = AudioConfig()
    expected_freq = audio_cfg.n_mels
    expected_time = (audio_cfg.sample_rate // audio_cfg.hop_length) + 1

    assert spec.dim() == 3, "Spectrogram must be 3D (C, F, T)"
    assert spec.shape[0] == 1, "Input channel should be 1"
    assert spec.shape[1] == expected_freq, f"Expected {expected_freq} mels"
    # Allow small variance in time frames due to padding/cropping logic
    assert (
        abs(spec.shape[2] - expected_time) <= 2
    ), f"Expected approx {expected_time} time frames"

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")
    model = AudioEfficientNetV2(num_classes=audio_cfg.num_classes)
    model.eval()

    # Create dummy input batch: (Batch, 1, Freq, Time)
    dummy_input = torch.randn(4, 1, expected_freq, expected_time)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")

    # Validation: Output shape
    assert output.shape == (
        4,
        audio_cfg.num_classes,
    ), f"Model output shape mismatch. Expected (4, {audio_cfg.num_classes})"

    # ---------------------------------------------------------
    # 4. Run Training Loop
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (Trainer)...")
    trainer = Trainer()
    trainer.train()

    # Verify checkpoints were created
    assert os.path.exists(path_config.last_checkpoint_path), "Last checkpoint not found"
    # Best model might not exist if validation accuracy was 0 (unlikely) or never improved,
    # but with 2 epochs it usually saves at least once.
    if os.path.exists(path_config.model_checkpoint_path):
        print("    Best model checkpoint found.")
    else:
        print("    Best model checkpoint not found (might not have improved).")

    # ---------------------------------------------------------
    # 5. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[5] Generating Submission...")

    # Load Test Metadata
    df_test = pd.read_csv(path_config.test_metadata_path)
    # For demo speed, limit test set
    df_test = df_test.iloc[:100]
    print(f"    Processing {len(df_test)} test samples...")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best model if available, else last
    ckpt_path = path_config.model_checkpoint_path
    if not os.path.exists(ckpt_path):
        ckpt_path = path_config.last_checkpoint_path

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Prepare transforms (same as dataset but manual for custom loop)
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        win_length=audio_cfg.win_length,
        hop_length=audio_cfg.hop_length,
        n_mels=audio_cfg.n_mels,
        f_min=audio_cfg.fmin,
        f_max=audio_cfg.fmax,
        normalized=False,
    ).to(device)

    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=audio_cfg.top_db).to(device)

    # Inference Loop
    predictions = []
    idx_to_label = {v: k for k, v in audio_cfg.label_to_idx.items()}

    with torch.no_grad():
        for _, row in df_test.iterrows():
            # Load Audio
            full_path = os.path.join(path_config.input_dir, row["file_path"])
            wav, sr = torchaudio.load(full_path)

            # Resample/Pad (Simplified version of dataset logic)
            if sr != audio_cfg.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, audio_cfg.sample_rate)
                wav = resampler(wav)

            target_len = audio_cfg.sample_rate * audio_cfg.duration
            if wav.shape[1] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
            else:
                wav = wav[:, :target_len]

            wav = wav.to(device)

            # Feature extraction
            spec = mel_transform(wav)
            spec = amp_to_db(spec)

            # Normalize
            mean = spec.mean()
            std = spec.std()
            spec = (spec - mean) / (std + 1e-6)

            # Add batch dim: (1, 1, F, T)
            spec = spec.unsqueeze(0)

            # Predict
            logits = model(spec)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = idx_to_label[pred_idx]

            predictions.append({"fname": row["fname"], "label": pred_label})

    # Save Submission
    df_sub = pd.DataFrame(predictions)
    output_path = path_config.submission_path
    df_sub.to_csv(output_path, index=False)

    print(f"    Submission saved to {output_path}")
    print(f"    First few rows:\n{df_sub.head()}")

    # Validation: Submission format
    assert os.path.exists(output_path), "Submission file was not created"
    df_check = pd.read_csv(output_path)
    assert list(df_check.columns) == ["fname", "label"], "Submission columns mismatch"
    assert len(df_check) == 100, "Submission row count mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
