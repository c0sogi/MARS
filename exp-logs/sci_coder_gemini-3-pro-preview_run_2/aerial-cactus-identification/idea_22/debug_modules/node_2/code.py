import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, dataset, model, engine


def main():
    print("Starting Cactus Classification Demo...")

    # 1. Setup
    device = config.DEVICE
    demo_seed = 42
    utils.set_seed(demo_seed)

    # Override config for the demo to ensure it runs quickly
    # We will only train one seed and for 1 epoch
    config.SEEDS = [demo_seed]
    config.EPOCHS = 1
    # We'll use a small batch size for the demo to ensure we have multiple batches even with small data
    demo_batch_size = 16

    print(f"Device: {device}")

    # 2. Data Loading & Subset Creation (Optimize for Speed)
    print("\n--- Data Loading & Verification ---")

    # Load raw data arrays
    # We use the library function to load data from metadata
    train_imgs, train_lbls, train_ids = dataset.load_data_split(
        config.TRAIN_METADATA_PATH, "train"
    )
    val_imgs, val_lbls, val_ids = dataset.load_data_split(
        config.VAL_METADATA_PATH, "val"
    )
    test_imgs, test_lbls, test_ids = dataset.load_data_split(
        config.TEST_METADATA_PATH, "test"
    )

    # Create tiny subsets (e.g., 100 samples) to make the epoch finish instantly
    subset_size = 100
    print(f"Subsetting data to {subset_size} samples for demonstration speed.")

    train_imgs_sub = train_imgs[:subset_size]
    train_lbls_sub = train_lbls[:subset_size]
    train_ids_sub = train_ids[:subset_size]

    val_imgs_sub = val_imgs[:subset_size]
    val_lbls_sub = val_lbls[:subset_size]
    val_ids_sub = val_ids[:subset_size]

    test_imgs_sub = test_imgs[:subset_size]
    test_ids_sub = test_ids[:subset_size]

    # Instantiate Datasets with the subsets
    train_dataset = dataset.CactusDataset(
        train_imgs_sub,
        train_lbls_sub,
        train_ids_sub,
        transform=dataset.get_transforms("train"),
    )
    val_dataset = dataset.CactusDataset(
        val_imgs_sub, val_lbls_sub, val_ids_sub, transform=dataset.get_transforms("val")
    )
    test_dataset = dataset.CactusDataset(
        test_imgs_sub,
        labels=None,
        ids=test_ids_sub,
        transform=dataset.get_transforms("test"),
    )

    # Create DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=demo_batch_size, shuffle=True, drop_last=True
        ),
        "val": DataLoader(val_dataset, batch_size=demo_batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=demo_batch_size, shuffle=False),
    }

    # Verify DataLoader output
    sample_imgs, sample_lbls, sample_ids = next(iter(dataloaders["train"]))
    print(f"Batch Image Shape: {sample_imgs.shape}")  # Should be (B, 3, 32, 32)
    print(f"Batch Label Shape: {sample_lbls.shape}")  # Should be (B,)

    # Assertions
    assert sample_imgs.shape == (
        demo_batch_size,
        3,
        32,
        32,
    ), "Incorrect image batch shape"
    assert sample_lbls.shape == (demo_batch_size,), "Incorrect label batch shape"
    assert (
        sample_imgs.max() <= 1.0 and sample_imgs.min() >= 0.0
    ), "Images not normalized to [0, 1]"

    # 3. Model Instantiation & Forward Pass Verification
    print("\n--- Model Verification ---")
    net = model.WideSERes2Net().to(device)

    # Pass the sample batch through the model
    with torch.no_grad():
        output = net(sample_imgs.to(device))

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Output should be (Batch_Size, 1) because the model ends with a Linear(..., 1)
    assert output.shape == (
        demo_batch_size,
        1,
    ), f"Expected output shape ({demo_batch_size}, 1), got {output.shape}"

    # 4. Training Loop Demonstration
    print("\n--- Training Loop Demonstration ---")
    # We run for 1 epoch on the subset
    best_auc = engine.train_seed(demo_seed, dataloaders, device, epochs=1)

    # Verify that the model checkpoint was saved
    expected_model_path = os.path.join(
        config.WORKING_DIR, f"model_seed_{demo_seed}.pth"
    )
    if os.path.exists(expected_model_path):
        print(f"Successfully saved checkpoint: {expected_model_path}")
    else:
        raise FileNotFoundError(
            f"Model checkpoint was not created at {expected_model_path}"
        )

    # 5. Inference Demonstration
    print("\n--- Inference Demonstration ---")
    # Run the ensemble inference engine (it will look for model_seed_42.pth)
    # Note: run_inference_ensemble uses config.SEEDS. We set config.SEEDS = [42] earlier.
    engine.run_inference_ensemble(dataloaders, device)

    # 6. Submission Verification
    print("\n--- Submission Verification ---")
    submission_path = config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Assertions
    # We used a subset of 100 test images, so submission should have 100 rows
    assert (
        len(df_sub) == subset_size
    ), f"Expected {subset_size} predictions, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Missing required columns in submission"

    # Check probability range
    probs = df_sub["has_cactus"].values
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\nDemo completed successfully. All components verified.")


if __name__ == "__main__":
    main()
