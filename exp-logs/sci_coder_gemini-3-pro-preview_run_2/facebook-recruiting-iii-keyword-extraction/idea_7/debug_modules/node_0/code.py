import pandas as pd
import numpy as np
import os
import sys
import torch
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Import Config and Apply Overrides
#    (Must be done BEFORE importing other library modules that use Config defaults)
# -----------------------------------------------------------------------------
from library.config import Config

# Define a separate working directory for this demo
DEMO_DIR = "./working/demo_run"
# Clean up previous run if exists
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Override Config paths to point to our demo subset files
Config.WORKING_DIR = DEMO_DIR
Config.TRAIN_PATH = os.path.join(DEMO_DIR, "train.csv")
Config.VAL_PATH = os.path.join(DEMO_DIR, "validation.csv")
Config.TEST_PATH = os.path.join(DEMO_DIR, "test.csv")
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

# Update artifact paths to reside in the demo directory
Config.TOKENIZER_PATH = os.path.join(DEMO_DIR, "tokenizer.json")
Config.TFIDF_VECTORIZER_PATH = os.path.join(DEMO_DIR, "tfidf_vectorizer.pkl")
Config.TAG_ENCODER_PATH = os.path.join(DEMO_DIR, "tag_encoder.pkl")
Config.TRAIN_PROCESSED_DATA = os.path.join(DEMO_DIR, "train_data")
Config.VAL_PROCESSED_DATA = os.path.join(DEMO_DIR, "val_data")
Config.TEST_PROCESSED_DATA = os.path.join(DEMO_DIR, "test_data")
Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")

# Reduce hyperparameters for speed
Config.VOCAB_SIZE = 500  # Small vocabulary
Config.MAX_LEN = 50  # Short sequence length
Config.TFIDF_MAX_FEATURES = 200  # Few TF-IDF features
Config.NUM_TAGS = 20  # Predict only top 20 tags
Config.EMBED_DIM = 32  # Small embedding dimension
Config.CNN_FILTERS = 16  # Few CNN filters
Config.BATCH_SIZE = 16  # Small batch size
Config.EPOCHS = 1  # Train for only 1 epoch
Config.NUM_WORKERS = 0  # Disable multiprocessing for small data

# Re-run setup to ensure new directories exist
Config.setup()

# -----------------------------------------------------------------------------
# 2. Create Data Subsets
# -----------------------------------------------------------------------------
print("Creating data subsets for demonstration...")

SUBSET_SIZE = 100  # Number of rows to use for the demo

# Load and save Train subset
df_train = pd.read_csv("./metadata/train.csv", nrows=SUBSET_SIZE)
df_train.to_csv(Config.TRAIN_PATH, index=False)

# Load and save Validation subset
df_val = pd.read_csv("./metadata/validation.csv", nrows=SUBSET_SIZE)
df_val.to_csv(Config.VAL_PATH, index=False)

# Load and save Test subset
df_test = pd.read_csv("./metadata/test.csv", nrows=SUBSET_SIZE)
df_test.to_csv(Config.TEST_PATH, index=False)

print(f"Subsets created in {DEMO_DIR} (Size: {SUBSET_SIZE} rows each)")

# -----------------------------------------------------------------------------
# 3. Import Library Modules
#    (Imported AFTER Config modification to ensure they use updated settings)
# -----------------------------------------------------------------------------
from library.utils import seed_everything
from library.preprocessing import Preprocessor
from library.dataset import StackExchangeDataset, get_dataloader
from library.model import WideDeepTextCNN
from library.loss import FocalLoss
from library.engine import Trainer


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
def main():
    # Set seed for reproducibility
    seed_everything(42)

    print("\n=== Step 1: Preprocessing and Data Loading ===")
    # Initialize Preprocessor
    preprocessor = Preprocessor()

    # Load and process training data
    # load_cached_data=False ensures we process the new subset CSVs from scratch
    print("Processing training data...")
    X_wide, X_deep, y = preprocessor.load_data(split="train", load_cached_data=False)

    # Validate Output Shapes
    print(
        f"Processed Train Shapes -> Wide: {X_wide.shape}, Deep: {X_deep.shape}, Targets: {y.shape}"
    )

    # Assertions to verify correctness
    assert X_wide.shape == (
        SUBSET_SIZE,
        Config.TFIDF_MAX_FEATURES,
    ), "Wide feature shape mismatch"
    assert X_deep.shape == (SUBSET_SIZE, Config.MAX_LEN), "Deep feature shape mismatch"
    assert y.shape == (SUBSET_SIZE, Config.NUM_TAGS), "Target shape mismatch"
    print("Preprocessing verification passed.")

    print("\n=== Step 2: Model Instantiation and Forward Pass Check ===")
    # Create DataLoader
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    batch = next(iter(train_loader))

    # Instantiate Model
    model = WideDeepTextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_tags=Config.NUM_TAGS,
        tfidf_dim=Config.TFIDF_MAX_FEATURES,
        cnn_filters=Config.CNN_FILTERS,
    ).to(Config.DEVICE)

    # Run Forward Pass with a single batch
    inputs = {
        "wide": batch["wide"].to(Config.DEVICE),
        "deep": batch["deep"].to(Config.DEVICE),
    }
    with torch.no_grad():
        logits = model(inputs)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_TAGS,
    ), "Output logits shape mismatch"

    # Test Loss Function
    criterion = FocalLoss()
    targets = batch["target"].to(Config.DEVICE)
    loss = criterion(logits, targets)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model and Loss verification passed.")

    print("\n=== Step 3: Full Training and Inference Pipeline ===")
    # Initialize Trainer
    trainer = Trainer()

    # Run the full pipeline: Train -> Val -> Threshold Opt -> Predict -> Submit
    # This uses the subset data and runs for 1 epoch
    trainer.run()

    print("\n=== Step 4: Submission Verification ===")
    if os.path.exists(Config.SUBMISSION_PATH):
        submission = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission File Loaded. Shape: {submission.shape}")
        print("First 3 rows:")
        print(submission.head(3))

        # Verify Submission Content
        assert len(submission) == SUBSET_SIZE, "Submission row count mismatch"
        assert (
            "Id" in submission.columns and "Tags" in submission.columns
        ), "Submission columns mismatch"

        # Verify Id matching
        test_ids = pd.read_csv(Config.TEST_PATH)["Id"].values
        assert np.all(
            submission["Id"].values == test_ids
        ), "Submission IDs do not match Test IDs"

        print("Verification Complete: Submission generated successfully.")
    else:
        raise RuntimeError("Submission file was not created!")


if __name__ == "__main__":
    main()
