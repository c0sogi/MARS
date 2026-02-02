import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.preprocessor import Preprocessor
from library.dataset import CachedSpeechDataset
from library.trainer import Trainer
from library.model import FrequencyPreservingSKResNetCRNN
from library.audio_processor import AudioProcessor


def create_subset_metadata():
    """
    Creates small subsets of the original metadata to speed up the demo.
    """
    print("Creating metadata subsets for demonstration...")

    # Define paths for subset metadata
    subset_train_path = os.path.join(Config.PROJECT_ROOT, "working", "demo_train.csv")
    subset_val_path = os.path.join(Config.PROJECT_ROOT, "working", "demo_val.csv")
    subset_test_path = os.path.join(Config.PROJECT_ROOT, "working", "demo_test.csv")

    # Load original metadata
    if not os.path.exists(Config.TRAIN_META):
        raise FileNotFoundError(f"Original metadata not found at {Config.TRAIN_META}")

    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # Sample subsets (ensure enough for a few batches)
    # We use a fixed random state in sampling for reproducibility
    df_train_sub = df_train.sample(n=100, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_val_sub = df_val.sample(n=50, random_state=Config.SEED).reset_index(drop=True)
    df_test_sub = df_test.sample(n=50, random_state=Config.SEED).reset_index(drop=True)

    # Save subsets
    df_train_sub.to_csv(subset_train_path, index=False)
    df_val_sub.to_csv(subset_val_path, index=False)
    df_test_sub.to_csv(subset_test_path, index=False)

    print(
        f"Subsets created: Train={len(df_train_sub)}, Val={len(df_val_sub)}, Test={len(df_test_sub)}"
    )

    return subset_train_path, subset_val_path, subset_test_path


def patch_config(train_path, val_path, test_path):
    """
    Patches the Config class to use subsets and faster training settings.
    """
    print("Patching configuration for fast execution...")

    # Point to subsets
    Config.TRAIN_META = train_path
    Config.VAL_META = val_path
    Config.TEST_META = test_path

    # Reduce training duration and batch size for the small dataset
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)


def verify_components():
    """
    Verifies that the Dataset and Model components work as expected.
    """
    print("Verifying components...")

    # 1. Verify Dataset
    ds = CachedSpeechDataset(Config.TRAIN_META, mode="train")
    assert len(ds) > 0, "Dataset should not be empty."

    # Fetch one sample
    features, label = ds[0]

    # Check Feature Shape: (3, 80, Time)
    # Time depends on N_SAMPLES (16000) and HOP_LENGTH (160) -> ~100 frames
    assert features.dim() == 3, f"Expected 3 dimensions, got {features.dim()}"
    assert features.shape[0] == 3, f"Expected 3 channels, got {features.shape[0]}"
    assert (
        features.shape[1] == Config.N_MELS
    ), f"Expected {Config.N_MELS} mels, got {features.shape[1]}"

    print(f"Dataset verification passed. Feature shape: {features.shape}")

    # 2. Verify Model
    model = FrequencyPreservingSKResNetCRNN()
    # Create a dummy batch
    dummy_input = features.unsqueeze(0)  # (1, 3, 80, T)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check Output Shape: (1, 12)
    assert output.shape == (
        1,
        Config.NUM_CLASSES,
    ), f"Expected output shape (1, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model verification passed.")


def run_inference():
    """
    Runs inference on the test subset using the trained model.
    """
    print("Running inference on test subset...")

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_META)

    # Initialize Processor and Model
    processor = AudioProcessor()
    device = torch.device(Config.DEVICE)
    model = FrequencyPreservingSKResNetCRNN().to(device)

    # Load Best Weights
    # Note: If training didn't improve over random init (unlikely but possible in 2 epochs with tiny data),
    # the file might not exist or score might be low. We handle this gracefully.
    if os.path.exists(Config.MODEL_SAVE_PATH):
        epoch, score = load_checkpoint(
            model, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE
        )
        print(f"Loaded model from epoch {epoch} with val acc {score:.4f}")
    else:
        print("Warning: No best model checkpoint found. Using random weights.")

    model.eval()

    predictions = []

    # Inference Loop
    # We iterate manually to demonstrate usage of AudioProcessor + Model for inference
    with torch.no_grad():
        for idx, row in df_test.iterrows():
            filepath = row["filepath"]
            fname = os.path.basename(filepath)

            # Process Audio
            # We use load_cached_data=True. Preprocessor should have cached it if we ran it on test meta.
            features_np = processor.process_file(filepath, load_cached_data=True)
            features_tensor = (
                torch.from_numpy(features_np).float().unsqueeze(0).to(device)
            )  # (1, 3, 80, T)

            # Predict
            output = model(features_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = Config.ID2LABEL[pred_idx]

            predictions.append({"fname": fname, "label": pred_label})

    # Create Submission DataFrame
    df_sub = pd.DataFrame(predictions)

    # Save
    submission_path = os.path.join(Config.PROJECT_ROOT, "submission", "submission.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Sample predictions:")
    print(df_sub.head())


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Prepare Data Subsets (Optimization for Speed)
    train_sub, val_sub, test_sub = create_subset_metadata()

    # 3. Patch Config (Optimization for Speed)
    patch_config(train_sub, val_sub, test_sub)

    # 4. Preprocessing (Feature Extraction)
    # This will cache features for our subset files.
    # Since we updated Config.TRAIN_META etc., it only processes the subset.
    preprocessor = Preprocessor()
    preprocessor.cache_dataset(load_cached_data=True)

    # 5. Verify Components (Logic Verification)
    verify_components()

    # 6. Training
    print("Starting training...")
    trainer = Trainer()
    trainer.fit()

    # 7. Inference & Submission
    run_inference()

    print("\n=== Demo Execution Complete ===")


if __name__ == "__main__":
    main()
