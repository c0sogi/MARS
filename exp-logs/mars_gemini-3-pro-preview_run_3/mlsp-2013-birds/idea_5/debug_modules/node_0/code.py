import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import CFG
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import (
    BirdDataset,
    load_or_compute_spectrograms,
    mixup_data,
    mixup_criterion,
)
from library.model import BirdResNet
from library.trainer import run_training, train_one_epoch, valid_one_epoch


def demonstrate_components():
    print(">>> Setting up Configuration for Demo...")
    # Override CFG for speed and demo purposes
    # We use a very small subset and minimal epochs to ensure quick execution
    CFG.debug = True
    CFG.debug_sample_size = 10  # Use only 10 samples
    CFG.epochs = 1  # Train for only 1 epoch
    CFG.batch_size = 2  # Small batch size
    CFG.n_folds = 2  # Minimal folds for CV
    CFG.num_workers = 0  # Avoid multiprocessing overhead for small demo
    CFG.print_freq = 1
    CFG.output_dir = "./working/demo_output"
    CFG.submission_dir = "./working/demo_submission"

    # Ensure directories exist
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    seed_everything(CFG.seed)

    print("\n>>> 1. Demonstrating Dataset and Spectrogram Computation...")
    # Load a small subset of data manually to test dataset class logic
    train_df = pd.read_csv(CFG.train_csv).head(CFG.debug_sample_size)

    # Compute spectrograms for this subset
    # We disable loading from cache initially to verify computation logic
    print("Computing spectrograms for subset...")
    spec_cache = load_or_compute_spectrograms([train_df], load_cached_data=False)

    # Verify cache content
    assert len(spec_cache) > 0, "Spectrogram cache should not be empty"
    sample_id = train_df.iloc[0]["rec_id"]
    assert (
        sample_id in spec_cache
    ), "Spectrogram for the first sample should be in cache"
    print(f"Spectrogram shape: {spec_cache[sample_id].shape}")

    # Instantiate Dataset
    print("Instantiating BirdDataset...")
    dataset = BirdDataset(train_df, spec_cache, phase="train")

    # Fetch one item to verify __getitem__
    image, label, rec_id = dataset[0]
    print(
        f"Dataset item - Image Shape: {image.shape}, Label Shape: {label.shape}, Rec ID: {rec_id}"
    )

    # Assertions for data shapes
    assert image.dim() == 3, "Image should be 3D (Channels, Freq, Time)"
    assert image.shape[0] == 3, "Image should have 3 channels (replicated)"
    assert (
        label.shape[0] == CFG.num_classes
    ), f"Label should have {CFG.num_classes} classes"
    assert isinstance(rec_id, (int, np.integer)), "Rec ID should be an integer"

    print("\n>>> 2. Demonstrating Model Architecture...")
    # Use CPU for simple shape check to avoid GPU initialization overhead if not needed
    device = torch.device("cpu")
    model = BirdResNet(pretrained=False, num_classes=CFG.num_classes)
    model.to(device)
    model.eval()

    # Create a dummy batch matching the dataset output
    dummy_input = image.unsqueeze(0).to(device)  # Add batch dimension -> (1, 3, F, T)
    print(f"Input batch shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (1, CFG.num_classes), "Output shape mismatch"

    print("\n>>> 3. Demonstrating Mixup...")
    # Create a dummy batch of 2 images to test mixup
    img2, lbl2, _ = dataset[1]
    batch_imgs = torch.stack([image, img2])
    batch_lbls = torch.stack([label, lbl2])

    # Apply mixup
    mixed_x, y_a, y_b, lam = mixup_data(
        batch_imgs, batch_lbls, alpha=1.0, use_cuda=False
    )
    print(f"Mixed input shape: {mixed_x.shape}")
    print(f"Lambda: {lam}")

    # Assertions for mixup
    assert mixed_x.shape == batch_imgs.shape, "Mixed batch shape should match original"
    assert y_a.shape == batch_lbls.shape, "Target A shape mismatch"
    assert y_b.shape == batch_lbls.shape, "Target B shape mismatch"

    print("\n>>> 4. Demonstrating Training Pipeline (run_training)...")
    # This function uses the CFG settings we modified earlier.
    # It will:
    # 1. Load metadata (train/val/test csvs).
    # 2. Compute/Load spectrograms (using the cache we built or computing new ones).
    # 3. Run Cross-Validation (2 folds, 1 epoch each).
    # 4. Generate submission.csv.
    run_training()

    # Verify submission file generation
    submission_path = os.path.join(CFG.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check submission format
    assert (
        "Id" in sub_df.columns and "Probability" in sub_df.columns
    ), "Submission columns missing"
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    # Check probability range
    assert (
        sub_df["Probability"].min() >= 0 and sub_df["Probability"].max() <= 1
    ), "Probabilities out of range [0, 1]"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    demonstrate_components()
