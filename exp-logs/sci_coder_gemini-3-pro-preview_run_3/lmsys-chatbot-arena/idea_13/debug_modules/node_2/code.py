import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DualStreamSiameseModel
from library.engine import run_training, generate_submission


def create_subset_data():
    """
    Creates small subsets of the original data for demonstration purposes.
    Saves them to the working directory.
    """
    print("Creating data subsets for fast demonstration...")

    # Define paths
    working_dir = "./working/demo_data"
    os.makedirs(working_dir, exist_ok=True)

    train_out = os.path.join(working_dir, "train_small.csv")
    val_out = os.path.join(working_dir, "val_small.csv")
    test_out = os.path.join(working_dir, "test_small.csv")
    sub_out = os.path.join(working_dir, "sample_submission_small.csv")

    # Load metadata (read-only)
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Subset
    # Use 32 rows for train to have a few batches
    train_small = train_df.head(32).copy()
    val_small = val_df.head(16).copy()
    test_small = test_df.head(16).copy()

    # Save subsets
    train_small.to_csv(train_out, index=False)
    val_small.to_csv(val_out, index=False)
    test_small.to_csv(test_out, index=False)

    # Create matching sample submission
    # The engine expects the sample submission to match the test set length
    sample_sub = pd.DataFrame(
        {
            "id": test_small["id"],
            "winner_model_a": 0.33,
            "winner_model_b": 0.33,
            "winner_tie": 0.33,
        }
    )
    sample_sub.to_csv(sub_out, index=False)

    return train_out, val_out, test_out, sub_out


def configure_demo_settings(train_path, val_path, test_path, sub_path):
    """
    Overrides Config class attributes to point to the subset data
    and reduce training time.
    """
    print("Configuring settings for demo run...")

    # Paths
    Config.TRAIN_PATH = train_path
    Config.VAL_PATH = val_path
    Config.TEST_PATH = test_path
    Config.SAMPLE_SUBMISSION_PATH = sub_path

    # Redirect cache and model output to a specific demo folder
    # to avoid messing with existing caches
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "model")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    Config.OUTPUT_SUBMISSION_PATH = os.path.join(
        Config.SUBMISSION_DIR, "submission.csv"
    )
    Config.BEST_MODEL_PATH = os.path.join(Config.MODEL_DIR, "best_model.pth")

    # Re-run setup to create these new directories
    Config.setup()

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.HIDDEN_SIZE = 768  # Keep default

    # Ensure we don't use old cached data if it exists in the main directory
    # (We pointed CACHE_DIR to a new location, so this is handled)


def verify_dataloader(loader, name="Loader"):
    """
    Fetches one batch from the loader and verifies shapes.
    """
    print(f"Verifying {name}...")
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise AssertionError(f"{name} is empty!")

    # Check keys
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "token_type_ids_a",
        "input_ids_b",
        "attention_mask_b",
        "token_type_ids_b",
        "scalars",
    ]
    for k in expected_keys:
        assert k in batch, f"Missing key {k} in batch"

    # Check shapes
    batch_size = batch["input_ids_a"].size(0)
    seq_len = batch["input_ids_a"].size(1)

    assert (
        batch_size == Config.TRAIN_BATCH_SIZE or batch_size == Config.VALID_BATCH_SIZE
    ), f"Batch size mismatch. Got {batch_size}"

    # Check scalars shape: [batch, 3]
    assert batch["scalars"].dim() == 2
    assert batch["scalars"].size(1) == 3, "Scalars should have 3 features (log lengths)"

    # Check target if present
    if "target" in batch:
        assert batch["target"].size(1) == 3, "Target should have 3 classes"

    print(f"  {name} verification passed. Batch size: {batch_size}, Seq len: {seq_len}")
    return batch


def verify_model_forward(batch):
    """
    Instantiates the model and runs a forward pass.
    """
    print("Verifying Model Forward Pass...")
    device = Config.DEVICE
    model = DualStreamSiameseModel()
    model.to(device)
    model.eval()

    # Move batch to device
    ids_a = batch["input_ids_a"].to(device)
    mask_a = batch["attention_mask_a"].to(device)
    type_a = batch["token_type_ids_a"].to(device)

    ids_b = batch["input_ids_b"].to(device)
    mask_b = batch["attention_mask_b"].to(device)
    type_b = batch["token_type_ids_b"].to(device)

    scalars = batch["scalars"].to(device)

    with torch.no_grad():
        logits = model(
            input_ids_a=ids_a,
            attention_mask_a=mask_a,
            token_type_ids_a=type_a,
            input_ids_b=ids_b,
            attention_mask_b=mask_b,
            token_type_ids_b=type_b,
            scalars=scalars,
        )

    # Check output shape: [batch_size, 3]
    assert logits.dim() == 2
    assert logits.size(0) == ids_a.size(0)
    assert logits.size(1) == 3

    print("  Model forward pass successful. Logits shape:", logits.shape)
    del model
    torch.cuda.empty_cache()


def main():
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Prepare Data
    train_path, val_path, test_path, sub_path = create_subset_data()

    # 3. Configure Config
    configure_demo_settings(train_path, val_path, test_path, sub_path)

    # 4. Get DataLoaders
    # load_cached_data=False ensures we process our new subset CSVs
    # instead of looking for existing npz files
    print("\nInitializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # 5. Verify Data and Model
    sample_batch = verify_dataloader(train_loader, "Train Loader")
    verify_dataloader(val_loader, "Val Loader")
    verify_dataloader(test_loader, "Test Loader")

    verify_model_forward(sample_batch)

    # 6. Run Training
    print("\nStarting Training Demo...")
    # This will train for 1 epoch on the subset and save the best model
    run_training(train_loader, val_loader)

    # Verify model file was created
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.BEST_MODEL_PATH}")
    print(f"Model successfully saved to {Config.BEST_MODEL_PATH}")

    # 7. Run Inference
    print("\nStarting Inference Demo...")
    submission_df = generate_submission(test_loader)

    # 8. Verify Submission
    print("Verifying Submission...")
    assert os.path.exists(Config.OUTPUT_SUBMISSION_PATH), "Submission file not created"

    # Check shape
    expected_len = len(pd.read_csv(test_path))
    assert (
        len(submission_df) == expected_len
    ), f"Submission length {len(submission_df)} != Test length {expected_len}"

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    for col in expected_cols:
        assert col in submission_df.columns, f"Missing column {col} in submission"

    # Check probability sum (approximate)
    probs = submission_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(
        axis=1
    )
    assert np.allclose(probs, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("\nAll demonstrations and verifications passed successfully!")
    print(f"Final submission saved to: {Config.OUTPUT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
