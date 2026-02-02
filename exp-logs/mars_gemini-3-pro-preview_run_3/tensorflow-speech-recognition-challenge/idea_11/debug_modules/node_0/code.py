import os
import shutil
import pandas as pd
import torch
import numpy as np
from library.config import (
    PathConfig,
    TrainConfig,
    ModelConfig,
    AudioConfig,
    set_seed,
)
from library.preprocessing import CacheGenerator
from library.dataset import get_dataloaders, SpeechCommandDataset, IDX2LABEL
from library.model import MultiScaleHierarchicalSKResNet
from library.trainer import Trainer
from library.utils import load_checkpoint


def run_demo():
    # --- 1. Setup & Configuration ---
    print("\n=== 1. Setup & Configuration ===")
    set_seed(42)

    # Define paths for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # Create custom configs for the demo
    demo_path_cfg = PathConfig(
        working_dir=demo_dir,
        cache_dir=os.path.join(demo_dir, "cache"),
        model_save_path=os.path.join(demo_dir, "best_model.pth"),
        submission_dir=demo_dir,
        submission_path=os.path.join(demo_dir, "submission.csv"),
        # We will update these after creating subsets
        train_meta=os.path.join(demo_dir, "train.csv"),
        val_meta=os.path.join(demo_dir, "val.csv"),
        test_meta=os.path.join(demo_dir, "test.csv"),
    )

    # Reduce training parameters for speed
    demo_train_cfg = TrainConfig(
        epochs=2,
        batch_size=8,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # --- 2. Data Subsetting ---
    print("\n=== 2. Creating Data Subsets ===")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (ensure we have enough for a batch)
    subset_size = 32
    df_train_sub = orig_train.head(subset_size).copy()
    df_val_sub = orig_val.head(subset_size).copy()
    df_test_sub = orig_test.head(subset_size).copy()

    # Save subsets
    df_train_sub.to_csv(demo_path_cfg.train_meta, index=False)
    df_val_sub.to_csv(demo_path_cfg.val_meta, index=False)
    df_test_sub.to_csv(demo_path_cfg.test_meta, index=False)

    print(f"Created subsets of size {subset_size} in {demo_dir}")

    # --- 3. Preprocessing (Caching) ---
    print("\n=== 3. Running Preprocessing (Caching) ===")

    cache_gen = CacheGenerator(config=demo_path_cfg)

    # Run caching for all splits
    # This generates spectrograms and saves new CSVs with 'cache_path' column
    meta_paths = cache_gen.run(load_cached_data=False)

    # Verify cache files exist
    cached_train_df = pd.read_csv(meta_paths["train"])
    sample_cache_path = cached_train_df.iloc[0]["cache_path"]

    if not os.path.exists(sample_cache_path):
        raise FileNotFoundError(f"Cache file was not created: {sample_cache_path}")

    # Verify content of cache
    spec_data = np.load(sample_cache_path)
    print(f"Cached Spectrogram Shape: {spec_data.shape}")

    # Assertion: Check shape (3 channels, 64 mels, ~101 time steps for 1s)
    # Time steps depend on padding in load_audio and hop_length.
    # 16000 samples / 160 hop = 100 frames + 1 center = 101.
    assert spec_data.shape[0] == 3, f"Expected 3 channels, got {spec_data.shape[0]}"
    assert spec_data.shape[1] == 64, f"Expected 64 mel bins, got {spec_data.shape[1]}"

    # Update paths to point to the cached metadata files
    cached_train_path = meta_paths["train"]
    cached_val_path = meta_paths["val"]
    cached_test_path = meta_paths["test"]

    # --- 4. Dataset & DataLoader ---
    print("\n=== 4. Verifying Dataset & DataLoader ===")

    # Initialize Loaders
    # Note: We temporarily override the global config in the library via the arguments we pass to Trainer,
    # but get_dataloaders uses train_cfg global. We need to patch it or just rely on the fact
    # that we passed batch_size to the loader, but get_dataloaders reads train_cfg.batch_size.
    # To ensure consistency, we will manually modify the global train_cfg in library.config for this session.
    import library.config

    library.config.train_cfg = demo_train_cfg

    train_loader, val_loader, test_loader = get_dataloaders(
        cached_train_path, cached_val_path, cached_test_path
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    assert images.shape[0] == demo_train_cfg.batch_size
    assert images.shape[1] == 3  # Channels
    assert images.shape[2] == 64  # Freq
    assert targets.shape[0] == demo_train_cfg.batch_size

    # --- 5. Model Initialization ---
    print("\n=== 5. Model Initialization & Forward Pass ===")

    model = MultiScaleHierarchicalSKResNet(config=ModelConfig())
    model.to(demo_train_cfg.device)

    # Forward pass
    images = images.to(demo_train_cfg.device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (demo_train_cfg.batch_size, ModelConfig.num_classes)

    # --- 6. Training Loop ---
    print("\n=== 6. Running Training Loop ===")

    trainer = Trainer(
        config=demo_train_cfg, model_config=ModelConfig(), path_config=demo_path_cfg
    )

    # Run fit
    trainer.fit(train_loader, val_loader)

    # Check if model was saved
    if not os.path.exists(demo_path_cfg.model_save_path):
        raise FileNotFoundError("Model checkpoint was not saved.")
    print("Training completed and model saved.")

    # --- 7. Inference ---
    print("\n=== 7. Running Inference ===")

    # Load best model
    best_model = MultiScaleHierarchicalSKResNet(config=ModelConfig())
    best_model.to(demo_train_cfg.device)
    load_checkpoint(
        best_model, demo_path_cfg.model_save_path, device=demo_train_cfg.device
    )
    best_model.eval()

    predictions = []
    fnames = []

    # Iterate test loader
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(demo_train_cfg.device)
            outputs = best_model(images)

            # Get predicted class indices
            _, preds = torch.max(outputs, 1)

            # Map indices to labels
            pred_labels = [IDX2LABEL[idx.item()] for idx in preds]
            predictions.extend(pred_labels)

    # Get filenames from the test dataframe
    df_test_cached = pd.read_csv(cached_test_path)
    # Extract filename from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
    fnames = df_test_cached["filepath"].apply(os.path.basename).tolist()

    # Create submission dataframe
    submission = pd.DataFrame({"fname": fnames, "label": predictions})

    # Save submission
    submission.to_csv(demo_path_cfg.submission_path, index=False)
    print(f"Submission saved to {demo_path_cfg.submission_path}")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
