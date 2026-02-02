import os
import shutil
import pandas as pd
import numpy as np
import torch
import logging

# Import library components
from library.config import CFG
from library.utils import seed_everything
from library.features import FeatureEngineer
from library.data import get_loaders, get_test_loader
from library.neural_net import EssayModel
from library.workflow import main as run_workflow


# --- Setup & Configuration ---
def setup_demo_environment():
    print("Setting up demo environment...")
    seed_everything(42)

    # Define paths
    demo_dir = "./working/demo_run"
    data_dir = os.path.join(demo_dir, "data")
    output_dir = os.path.join(demo_dir, "output")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load a small subset of real data for realistic testing
    # We use the metadata/train.csv as the source
    full_train = pd.read_csv("./metadata/train.csv")

    # Select 20 samples for training/validation and 5 for testing
    # Ensure we cover a few score classes
    subset = full_train.sample(n=25, random_state=42).reset_index(drop=True)

    train_subset = subset.iloc[:15]
    val_subset = subset.iloc[15:20]
    test_subset = subset.iloc[20:].drop(columns=["score"])  # Test set has no score

    # Save to demo directory
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(output_dir, "submission.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(
        f"Created demo datasets: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # --- Monkey-patch CFG for Speed ---
    print("Overriding CFG parameters for fast demonstration...")
    CFG.output_dir = output_dir
    CFG.train_path = train_path
    CFG.val_path = val_path
    CFG.test_path = test_path
    CFG.submission_file = sub_path

    # Reduce compute load
    CFG.epochs = 1
    CFG.num_folds = 2  # Run 2 folds to demonstrate CV loop
    CFG.train_batch_size = 2
    CFG.valid_batch_size = 2
    CFG.use_awp = False  # Disable AWP for speed

    # We keep the model as DeBERTa to ensure code compatibility,
    # but with only 15 training samples, it will run fast on A100.

    return train_subset, test_subset


def test_feature_engineering(train_df):
    print("\n--- Testing Feature Engineering ---")
    fe = FeatureEngineer()

    # Test extraction on the small training set
    # Note: split_name='train' triggers vocab building
    features = fe.extract_features(train_df, split_name="train", load_cached_data=False)

    print(f"Extracted features shape: {features.shape}")

    # Assertions
    expected_cols = ["word_count", "sent_count", "flesch_kincaid", "oov_ratio"]
    for col in expected_cols:
        if col not in features.columns:
            raise AssertionError(f"Missing expected feature column: {col}")

    if len(features) != len(train_df):
        raise AssertionError(
            f"Feature count mismatch: {len(features)} vs {len(train_df)}"
        )

    print("Feature Engineering verification passed.")


def test_data_loading():
    print("\n--- Testing Data Loading ---")
    # Test getting loaders for Fold 0
    # This triggers tokenization and dataset creation
    train_loader, val_loader = get_loaders(fold=0, load_cached_data=False)

    # Fetch a batch
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Assertions
    if input_ids.shape[0] != CFG.train_batch_size:
        raise AssertionError(
            f"Batch size mismatch. Expected {CFG.train_batch_size}, got {input_ids.shape[0]}"
        )

    if input_ids.shape[1] != CFG.max_length:
        raise AssertionError(
            f"Sequence length mismatch. Expected {CFG.max_length}, got {input_ids.shape[1]}"
        )

    if not torch.is_floating_point(labels):
        raise AssertionError("Labels should be floating point tensors for regression.")

    print("Data Loading verification passed.")
    return batch


def test_model_architecture(batch):
    print("\n--- Testing Model Architecture ---")
    device = CFG.device

    # Instantiate model (pretrained=False for speed in initialization test,
    # but workflow uses True. We use True here to verify weight loading works)
    model = EssayModel(pretrained=True)
    model.to(device)
    model.eval()

    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids, mask)

    print(f"Model output shape: {output.shape}")

    # Assertions
    # Output should be (Batch_Size,) because of squeeze() in forward
    expected_shape = (CFG.train_batch_size,)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print("Model Architecture verification passed.")

    # Clean up memory
    del model
    torch.cuda.empty_cache()


def test_full_workflow():
    print("\n--- Testing Full Workflow Execution ---")
    # This runs the main function which includes:
    # 1. Classic Feature Extraction (Ridge)
    # 2. NN Training (2 Folds as configured)
    # 3. Stacking (LGBM)
    # 4. Submission generation

    # We disable loading cached data to force execution of the pipeline logic
    run_workflow(load_cached_data=False)

    # Verify Submission
    if not os.path.exists(CFG.submission_file):
        raise AssertionError("Submission file was not generated.")

    sub_df = pd.read_csv(CFG.submission_file)
    print(f"Submission generated with shape: {sub_df.shape}")
    print(sub_df.head())

    # Check format
    if list(sub_df.columns) != ["essay_id", "score"]:
        raise AssertionError(f"Submission columns incorrect: {sub_df.columns}")

    # Check scores are integers (as required by metric/format)
    if not np.issubdtype(sub_df["score"].dtype, np.integer):
        raise AssertionError("Submission scores must be integers.")

    print("Full Workflow verification passed.")


if __name__ == "__main__":
    # 1. Setup
    train_subset, test_subset = setup_demo_environment()

    # 2. Unit Tests
    test_feature_engineering(train_subset)
    batch = test_data_loading()
    test_model_architecture(batch)

    # 3. Integration Test
    test_full_workflow()

    print("\nAll demonstrations completed successfully.")
