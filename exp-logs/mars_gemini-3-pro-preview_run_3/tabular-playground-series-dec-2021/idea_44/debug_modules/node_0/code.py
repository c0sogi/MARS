import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_accuracy, save_submission
from library.data_loader import get_dataloaders, ForestDataset, engineer_features
from library.model import AsymmetricParallelNet
from library.train import train_model, inference


def main():
    # 1. Setup
    print("=== Setting up Demo Environment ===")
    warnings.filterwarnings("ignore")
    set_seed(42)

    # Define temporary paths for the demo
    demo_dir = "./working/demo_execution"
    demo_data_dir = os.path.join(demo_dir, "data")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_submission_dir = os.path.join(demo_dir, "submission")
    demo_model_path = os.path.join(demo_dir, "best_model.pth")
    demo_sub_path = os.path.join(demo_submission_dir, "submission_demo.csv")

    # Clean up previous runs if they exist
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_data_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # 2. Create a Mini Dataset for Speed
    # We read a small chunk of the actual data to ensure schema consistency
    print("Creating mini-dataset from metadata...")
    subset_size = 2000  # Small subset for fast execution

    # Load and save subsets
    train_subset = pd.read_parquet(Config.TRAIN_META).head(subset_size)
    val_subset = pd.read_parquet(Config.VAL_META).head(subset_size)
    test_subset = pd.read_parquet(Config.TEST_META).head(subset_size)

    train_meta_path = os.path.join(demo_data_dir, "train.parquet")
    val_meta_path = os.path.join(demo_data_dir, "val.parquet")
    test_meta_path = os.path.join(demo_data_dir, "test.parquet")

    train_subset.to_parquet(train_meta_path)
    val_subset.to_parquet(val_meta_path)
    test_subset.to_parquet(test_meta_path)
    print(f"Mini-datasets saved to {demo_data_dir}")

    # 3. Monkey-Patch Config
    # We override the Config class attributes to use our demo settings
    print("Overriding Config for demo...")
    Config.TRAIN_META = train_meta_path
    Config.VAL_META = val_meta_path
    Config.TEST_META = test_meta_path
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = demo_sub_path
    Config.MODEL_PATH = demo_model_path

    # Optimization settings
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny dataset

    # Initialize directories based on new config
    Config.initialize()

    # 4. Component Verification: Feature Engineering & Dataset
    print("\n=== Verifying Data Components ===")

    # Test engineer_features
    print("Testing Feature Engineering...")
    raw_df = train_subset.copy()
    processed_df = engineer_features(raw_df)

    expected_new_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Hydrology_Euclidean",
        "Hydrology_Elevation",
        "Mean_Amenities",
    ]
    for col in expected_new_cols:
        if col not in processed_df.columns:
            raise AssertionError(f"Feature engineering failed: Missing {col}")
    print("Feature Engineering check passed.")

    # Test ForestDataset
    print("Testing ForestDataset...")
    # Create dummy data
    dummy_X = np.random.randn(10, 5)
    dummy_y = np.random.randint(0, 7, size=(10,))
    dataset = ForestDataset(dummy_X, dummy_y)

    x_sample, y_sample = dataset[0]
    if not isinstance(x_sample, torch.Tensor) or not isinstance(y_sample, torch.Tensor):
        raise AssertionError("Dataset __getitem__ should return Tensors")
    if x_sample.shape[0] != 5:
        raise AssertionError("Dataset feature shape mismatch")
    print("ForestDataset check passed.")

    # 5. Component Verification: Model
    print("\n=== Verifying Model Architecture ===")
    input_dim = processed_df.shape[1] - 2  # Minus Id and Target
    # Note: Preprocessing adds binary/continuous handling, so input_dim might differ slightly
    # in the pipeline, but we test the class mechanics here.

    model = AsymmetricParallelNet(input_dim=100, num_classes=7)
    dummy_input = torch.randn(4, 100)  # Batch size 4, 100 features

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (4, 7):
        raise AssertionError(
            f"Model output shape mismatch. Expected (4, 7), got {output.shape}"
        )
    print("AsymmetricParallelNet forward pass check passed.")

    # 6. Full Pipeline Execution
    print("\n=== Executing Full Pipeline (Train/Val/Test) ===")

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # A. Get Dataloaders (Triggers preprocessing and caching)
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing of our new mini-dataset
    )

    # Verify DataLoader
    batch_x, batch_y = next(iter(train_loader))
    real_input_dim = batch_x.shape[1]
    print(f"Data Loaded. Input Dimension: {real_input_dim}")

    # B. Initialize Model
    model = AsymmetricParallelNet(
        input_dim=real_input_dim, num_classes=Config.NUM_CLASSES
    )
    model = model.to(device)

    # C. Train
    print("Starting Training Loop...")
    model = train_model(model, train_loader, val_loader, device)

    # Verify model file was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise AssertionError("Model checkpoint was not saved.")

    # D. Inference
    print("Running Inference...")
    preds = inference(model, test_loader, device)

    if len(preds) != len(test_ids):
        raise AssertionError(
            f"Prediction count mismatch. Expected {len(test_ids)}, got {len(preds)}"
        )

    # E. Submission
    print("Saving Submission...")
    save_submission(preds, test_ids, Config.SUBMISSION_PATH)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    if list(sub_df.columns) != ["Id", "Cover_Type"]:
        raise AssertionError("Submission file has incorrect columns.")
    if len(sub_df) != subset_size:
        raise AssertionError(
            f"Submission file row count mismatch. Expected {subset_size}, got {len(sub_df)}"
        )

    print("\n=== Demo Completed Successfully ===")
    print(f"Output stored in: {Config.WORKING_DIR}")


if __name__ == "__main__":
    main()
