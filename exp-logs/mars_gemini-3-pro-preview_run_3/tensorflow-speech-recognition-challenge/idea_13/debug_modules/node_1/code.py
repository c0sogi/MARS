import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import from the provided library
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    N_SAMPLES,
    N_CLASSES,
    LABEL_TO_IDX,
)
from library.utils import set_seed, LabelEncoder
from library.dataset import SpeechDataset
from library.transforms import AudioProcessor
from library.modules import HybridSKCRNN
from library.engine import Trainer


def create_demo_data(n_samples=100):
    """
    Creates small subsets of the metadata files for demonstration purposes.
    """
    print("Creating demo metadata subsets...")

    # Define paths for demo metadata
    demo_train_path = os.path.join(WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(WORKING_DIR, "demo_test.csv")

    # Load original metadata
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Sample subsets
    # Ensure we have at least one of each class in train if possible,
    # but for a simple demo random sampling is usually sufficient given the seed.
    df_train_sub = df_train.sample(n=min(n_samples, len(df_train)), random_state=SEED)
    df_val_sub = df_val.sample(n=min(n_samples, len(df_val)), random_state=SEED)
    df_test_sub = df_test.sample(n=min(n_samples, len(df_test)), random_state=SEED)

    # Save to working dir
    df_train_sub.to_csv(demo_train_path, index=False)
    df_val_sub.to_csv(demo_val_path, index=False)
    df_test_sub.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def verify_components(demo_train_path):
    """
    Instantiates model components and verifies shapes/logic.
    """
    print("Verifying model components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify Dataset
    cache_dir = os.path.join(WORKING_DIR, "demo_cache")
    dataset = SpeechDataset(demo_train_path, mode="train", cache_dir=cache_dir)

    # Check length
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check item structure
    wav, target = dataset[0]
    assert isinstance(wav, torch.Tensor), "Waveform should be a tensor"
    assert (
        wav.shape[0] == N_SAMPLES
    ), f"Waveform length mismatch. Expected {N_SAMPLES}, got {wav.shape[0]}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # 2. Verify AudioProcessor
    processor = AudioProcessor().to(device)
    # Create a dummy batch (Batch Size, Time)
    dummy_wavs = torch.randn(4, N_SAMPLES).to(device)

    processor.train()
    wavs_aug, specs_aug = processor(dummy_wavs)

    # Check output shapes
    # Wavs: (B, T)
    assert wavs_aug.shape == dummy_wavs.shape, "Augmented waveform shape mismatch"
    # Specs: (B, 3, F, T). F is N_MELS=64. T depends on hop length.
    # With 16000 samples and hop=320, T approx 51.
    assert specs_aug.dim() == 4, "Spectrograms should be 4D (B, C, F, T)"
    assert specs_aug.shape[1] == 3, "Should have 3 channels (multi-resolution)"
    assert specs_aug.shape[2] == 64, "Should have 64 Mel bands"

    # 3. Verify HybridSKCRNN
    model = HybridSKCRNN().to(device)
    outputs = model(wavs_aug, specs_aug)

    assert outputs.shape == (
        4,
        N_CLASSES,
    ), f"Model output shape mismatch. Expected (4, {N_CLASSES}), got {outputs.shape}"

    print("Component verification passed.")


def get_demo_dataloaders(train_csv, val_csv, test_csv):
    """
    Creates dataloaders specifically for the demo subsets.
    Replicates logic from library.dataset.get_dataloaders but with custom paths.
    """
    cache_dir = os.path.join(WORKING_DIR, "demo_cache")

    # Datasets
    train_dataset = SpeechDataset(train_csv, mode="train", cache_dir=cache_dir)
    val_dataset = SpeechDataset(val_csv, mode="val", cache_dir=cache_dir)
    test_dataset = SpeechDataset(test_csv, mode="test", cache_dir=cache_dir)

    # Sampler for training
    targets = train_dataset.targets
    class_counts = np.bincount(targets, minlength=len(LabelEncoder().label_to_idx))
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Loaders
    # Using smaller batch size for demo if needed, but 32 is fine
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True if len(train_dataset) >= BATCH_SIZE else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def run_demo():
    set_seed(SEED)

    # 1. Prepare Data
    demo_train_csv, demo_val_csv, demo_test_csv = create_demo_data(n_samples=100)

    # 2. Verify Logic
    verify_components(demo_train_csv)

    # 3. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer()

    # 4. Get DataLoaders
    train_loader, val_loader, test_loader = get_demo_dataloaders(
        demo_train_csv, demo_val_csv, demo_test_csv
    )

    # 5. Train
    print("Starting demo training loop...")
    demo_save_path = os.path.join(WORKING_DIR, "demo_model.pth")

    # Train for 2 epochs to demonstrate the loop and checkpointing
    best_acc = trainer.fit(
        train_loader, val_loader, epochs=2, patience=2, save_path=demo_save_path
    )

    assert os.path.exists(demo_save_path), "Model checkpoint was not saved."
    print(f"Training finished. Best Acc: {best_acc}")

    # 6. Generate Submission
    print("Generating demo submission...")
    demo_sub_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # We temporarily monkey-patch the TEST_CSV in library.engine or handle the mismatch
    # The library.engine.Trainer.generate_submission reads TEST_CSV to get filenames.
    # Since we are using a subset, the lengths won't match if it reads the full TEST_CSV.
    # However, the generate_submission method in Trainer reads TEST_CSV internally.
    # To make this work without modifying the library, we must ensure that the
    # test_loader passed corresponds to the file at TEST_CSV.
    #
    # Since we cannot easily change the import in `library.engine`, we will manually
    # generate the submission using the logic here, or we accept that `Trainer.generate_submission`
    # might print a warning about mismatch.
    #
    # Actually, `Trainer.generate_submission` reads `TEST_CSV` from config.
    # We can't change that constant.
    # So we will invoke the prediction manually here to ensure correctness for the demo subset.

    # Manual Prediction Loop for Demo Subset
    trainer.model.eval()
    trainer.processor.eval()
    device = trainer.device
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            waveforms, specs = trainer.processor(inputs)
            outputs = trainer.model(waveforms, specs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    label_encoder = LabelEncoder()
    pred_labels = label_encoder.decode_batch(all_preds)

    # Load the demo test csv to get filenames
    df_demo_test = pd.read_csv(demo_test_csv)
    fnames = df_demo_test["filepath"].apply(os.path.basename).tolist()

    assert len(fnames) == len(pred_labels), "Prediction count mismatch"

    submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})
    submission_df.to_csv(demo_sub_path, index=False)

    print(f"Demo submission saved to {demo_sub_path}")
    print(submission_df.head())


if __name__ == "__main__":
    try:
        run_demo()
        print("\n=== Demo Completed Successfully ===")
    except Exception as e:
        print(f"\n=== Demo Failed: {e} ===")
        raise e
