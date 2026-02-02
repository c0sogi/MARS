import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import model
from library import engine


def run_demonstration():
    print("=== Starting Cdiscount Classification Task Demonstration ===")

    # 1. Setup and Reproducibility
    config.seed_everything(config.SEED)
    print(f"Device selected: {config.DEVICE}")

    # 2. Create Mini-Datasets for Speed
    # We create small subsets of the metadata to demonstrate the pipeline quickly.
    print("\n--- Creating Mini Metadata Subsets ---")

    # Define paths for mini metadata
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # Sample Train
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_train_mini = df_train.sample(n=100, random_state=config.SEED)
    df_train_mini.to_csv(mini_train_path, index=False)
    print(f"Created mini train metadata: {len(df_train_mini)} samples")

    # Sample Val
    df_val = pd.read_csv(config.VAL_META_PATH)
    df_val_mini = df_val.sample(n=20, random_state=config.SEED)
    df_val_mini.to_csv(mini_val_path, index=False)
    print(f"Created mini val metadata: {len(df_val_mini)} samples")

    # Sample Test
    df_test = pd.read_csv(config.TEST_META_PATH)
    df_test_mini = df_test.sample(n=20, random_state=config.SEED)
    df_test_mini.to_csv(mini_test_path, index=False)
    print(f"Created mini test metadata: {len(df_test_mini)} samples")

    # 3. Instantiate Datasets and DataLoaders
    print("\n--- Instantiating Datasets and Loaders ---")

    # Training Dataset
    train_ds = dataset.CdiscountDataset(
        metadata_path=mini_train_path,
        bson_path=config.TRAIN_BSON,
        mode="train",
        transform=dataset.get_transforms("train"),
    )

    # Validation Dataset
    val_ds = dataset.CdiscountDataset(
        metadata_path=mini_val_path,
        bson_path=config.TRAIN_BSON,
        mode="val",
        transform=dataset.get_transforms("val"),
    )

    # Test Dataset
    test_ds = dataset.CdiscountDataset(
        metadata_path=mini_test_path,
        bson_path=config.TEST_BSON,
        mode="test",
        transform=dataset.get_transforms("test"),
    )

    # Verification of Dataset Logic
    print("Verifying dataset output shapes...")
    sample_img, sample_label = train_ds[0]
    assert isinstance(sample_img, torch.Tensor), "Image should be a tensor"
    assert sample_img.shape == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert isinstance(sample_label, int) or isinstance(
        sample_label, pd.Int64Dtype
    ), "Label should be an integer"

    # Test dataset returns a stack of images and a product ID
    sample_stack, sample_pid = test_ds[0]
    assert sample_stack.dim() == 4, "Test output should be 4D (Num_Images, C, H, W)"
    assert sample_stack.shape[1:] == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), "Incorrect stack image dimensions"

    # Create Loaders
    # Using small batch size for demo
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    # Test loader must have batch_size=1 because number of images varies per product
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    # Using pretrained=False for speed/offline safety in this demo
    model_instance = model.MobileNetV2Classifier(
        num_classes=config.NUM_CLASSES, pretrained=False
    )
    model_instance = model_instance.to(config.DEVICE)

    # Verify model forward pass
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    with torch.no_grad():
        dummy_output = model_instance(dummy_input)
    assert dummy_output.shape == (2, config.NUM_CLASSES), "Model output shape mismatch"
    print("Model initialized and verified.")

    # 5. Training Loop
    print("\n--- Executing Training Loop (1 Epoch) ---")
    # We run for 1 epoch to demonstrate the engine
    trained_model = engine.train_model(
        model=model_instance,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        device=config.DEVICE,
        patience=1,
        load_cached_weights=False,  # Force calculation (or load if cache exists)
    )

    # 6. Inference and Submission
    print("\n--- Generating Predictions ---")
    output_csv = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    engine.generate_predictions(
        model=trained_model,
        test_loader=test_loader,
        device=config.DEVICE,
        output_path=output_csv,
    )

    # Verify Submission
    assert os.path.exists(output_csv), "Submission file was not created"
    df_sub = pd.read_csv(output_csv)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    assert list(df_sub.columns) == ["_id", "category_id"], "Submission columns mismatch"
    assert len(df_sub) == len(df_test_mini), "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
