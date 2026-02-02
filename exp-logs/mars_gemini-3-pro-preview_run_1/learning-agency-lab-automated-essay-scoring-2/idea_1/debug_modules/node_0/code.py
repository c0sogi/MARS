import os
import pandas as pd
import torch
import numpy as np
import shutil
import warnings

# Import provided library components
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import get_dataloaders
from library.model import DANRegressor
from library.train import Trainer
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(n_samples=100):
    """
    Creates small subsets of the metadata CSVs to speed up the demonstration.
    """
    print(f"Creating data subsets with {n_samples} samples each...")

    # Define source paths (from original metadata)
    src_train = os.path.join("./metadata", "train.csv")
    src_val = os.path.join("./metadata", "val.csv")
    src_test = os.path.join("./metadata", "test.csv")

    # Define destination paths (in working directory)
    dst_train = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    dst_val = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    dst_test = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Read and sample data
    # We use error_bad_lines=False or on_bad_lines='skip' logic implicitly handled by pandas defaults usually,
    # but here data is clean.
    df_train = pd.read_csv(src_train).head(n_samples)
    df_val = pd.read_csv(src_val).head(n_samples)
    df_test = pd.read_csv(src_test).head(n_samples)

    # Save subsets
    df_train.to_csv(dst_train, index=False)
    df_val.to_csv(dst_val, index=False)
    df_test.to_csv(dst_test, index=False)

    print("Subset creation complete.")
    return dst_train, dst_val, dst_test


def configure_demo_settings(train_path, val_path, test_path):
    """
    Modifies the global Config class to use subsets and faster hyperparameters.
    """
    print("Configuring demo settings...")

    # 1. Update Data Paths
    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path

    # 2. Update Cache Directory to avoid conflicts with real runs
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 3. Update Model/Training Hyperparameters for Speed
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 32  # Smaller batch size
    Config.EMBED_DIM = 64  # Smaller embeddings
    Config.HIDDEN_DIM = 64  # Smaller hidden layer
    Config.VOCAB_SIZE = 5000  # Smaller vocab
    Config.PATIENCE = 1  # Fail fast if no improvement

    # Ensure model save path is in the working directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")


def verify_metric():
    """
    Verifies the Quadratic Weighted Kappa calculation.
    """
    print("\n=== Verifying Metric (QWK) ===")

    # Case 1: Perfect agreement
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred = np.array([1, 2, 3, 4, 5, 6])
    score = compute_qwk(y_true, y_pred)
    print(f"Perfect Agreement Score: {score}")
    assert np.isclose(score, 1.0), "QWK should be 1.0 for perfect agreement"

    # Case 2: Complete disagreement
    y_true_bad = np.array([1, 1, 1])
    y_pred_bad = np.array([6, 6, 6])
    score_bad = compute_qwk(y_true_bad, y_pred_bad)
    print(f"Disagreement Score: {score_bad}")
    # QWK can be 0 or negative depending on chance agreement, but definitely not 1.0
    assert score_bad < 0.5, "QWK should be low for disagreement"

    print("Metric verification passed.")


def verify_data_pipeline():
    """
    Verifies data loading, tokenization, and batching.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Force processing from scratch using the subset data
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=False
    )

    # Check Tokenizer
    print(f"Tokenizer Vocab Size: {len(tokenizer)}")
    assert (
        len(tokenizer) > 2
    ), "Tokenizer should have learned words beyond special tokens"

    # Check Train Loader Batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    scores = batch["scores"]
    essay_ids = batch["essay_ids"]

    print(f"Batch Input Shape: {input_ids.shape}")
    print(f"Batch Scores Shape: {scores.shape}")

    # Assertions
    assert input_ids.dim() == 2, "Input IDs should be (Batch, Seq_Len)"
    assert scores.dim() == 1, "Scores should be (Batch,)"
    assert len(essay_ids) == input_ids.size(0), "Mismatch in batch size vs essay ids"
    assert input_ids.size(0) <= Config.BATCH_SIZE, "Batch size exceeds configuration"

    print("Data pipeline verification passed.")
    return train_loader, val_loader, test_loader


def verify_model_architecture():
    """
    Verifies the model forward pass and output shape.
    """
    print("\n=== Verifying Model Architecture ===")

    model = DANRegressor(Config)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input: Batch size 4, Seq len 10
    dummy_input = torch.randint(0, Config.VOCAB_SIZE, (4, 10)).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"

    print("Model architecture verification passed.")


def verify_training(train_loader, val_loader):
    """
    Verifies the training loop and model saving.
    """
    print("\n=== Verifying Training Loop ===")

    # Remove existing model if any
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    trainer = Trainer(Config)

    # Run training
    # Note: We reduced epochs to 2 in configure_demo_settings
    trainer.fit(train_loader, val_loader)

    # Assertions
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")
    print("Training verification passed.")


def verify_inference():
    """
    Verifies the inference process and submission file generation.
    """
    print("\n=== Verifying Inference Pipeline ===")

    # Remove existing submission if any
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Run inference
    # We use load_cached_data=True here because get_dataloaders was already run in verify_data_pipeline
    # and the cache (demo_cache) is populated.
    generate_submission(load_cached_data=True)

    # Assertions
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    assert "essay_id" in df_sub.columns, "Missing 'essay_id' column"
    assert "score" in df_sub.columns, "Missing 'score' column"

    # Check value range
    scores = df_sub["score"]
    assert scores.min() >= 1, "Scores must be >= 1"
    assert scores.max() <= 6, "Scores must be <= 6"
    assert pd.api.types.is_integer_dtype(scores), "Scores must be integers"

    # Check against test subset size
    # We created a subset of 100 samples
    expected_len = 100
    assert (
        len(df_sub) == expected_len
    ), f"Expected {expected_len} predictions, got {len(df_sub)}"

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(Config.SEED)

    # 2. Setup Data Subsets
    train_csv, val_csv, test_csv = create_subset_data(n_samples=100)

    # 3. Configure Global Settings
    configure_demo_settings(train_csv, val_csv, test_csv)

    # 4. Verify Metric
    verify_metric()

    # 5. Verify Data Pipeline
    train_loader, val_loader, test_loader = verify_data_pipeline()

    # 6. Verify Model
    verify_model_architecture()

    # 7. Verify Training
    verify_training(train_loader, val_loader)

    # 8. Verify Inference
    verify_inference()

    print("\nAll demonstrations and verifications completed successfully.")
