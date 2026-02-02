import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_dataloaders
from library.model import WideAndDeepTextCNN, FocalLoss
from library.trainer import ModelTrainer


def setup_demo_environment():
    """
    Creates a small subset of data and patches the Config class
    to run a fast demonstration.
    """
    print("Setting up demo environment...")

    # Define paths for demo data
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    raw_train_small = os.path.join(demo_dir, "train_small.csv")
    raw_test_small = os.path.join(demo_dir, "test_small.csv")
    meta_train_small = os.path.join(demo_dir, "meta_train_small.csv")
    meta_val_small = os.path.join(demo_dir, "meta_val_small.csv")
    meta_test_small = os.path.join(demo_dir, "meta_test_small.csv")

    # 1. Create Small Raw Data Files
    # Read first N rows from input files to avoid processing huge datasets
    N_SAMPLES = 2000

    # Load raw train (Id, Title, Body, Tags)
    # We only need Id, Title, Body for the 'raw' file mocked in Config
    df_raw_train = pd.read_csv(Config.TRAIN_RAW_FILE, nrows=N_SAMPLES)
    df_raw_train.to_csv(raw_train_small, index=False)

    # Load raw test
    df_raw_test = pd.read_csv(Config.TEST_RAW_FILE, nrows=N_SAMPLES)
    df_raw_test.to_csv(raw_test_small, index=False)

    # 2. Create Small Metadata Files
    # We must ensure the metadata IDs exist in our small raw files
    valid_train_ids = set(df_raw_train["Id"])
    valid_test_ids = set(df_raw_test["Id"])

    # Filter Train Metadata
    df_meta_train = pd.read_csv(Config.TRAIN_META_FILE)
    df_meta_train = df_meta_train[df_meta_train["Id"].isin(valid_train_ids)].head(1500)
    df_meta_train.to_csv(meta_train_small, index=False)

    # Filter Val Metadata (using same raw train file source)
    df_meta_val = pd.read_csv(Config.VAL_META_FILE)
    df_meta_val = df_meta_val[df_meta_val["Id"].isin(valid_train_ids)].head(300)
    df_meta_val.to_csv(meta_val_small, index=False)

    # Filter Test Metadata
    df_meta_test = pd.read_csv(Config.TEST_META_FILE)
    df_meta_test = df_meta_test[df_meta_test["Id"].isin(valid_test_ids)].head(100)
    df_meta_test.to_csv(meta_test_small, index=False)

    print(
        f"Created small datasets: Train={len(df_meta_train)}, Val={len(df_meta_val)}, Test={len(df_meta_test)}"
    )

    # 3. Patch Config
    # We modify the class attributes directly to affect the library modules
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_RAW_FILE = raw_train_small
    Config.TEST_RAW_FILE = raw_test_small
    Config.TRAIN_META_FILE = meta_train_small
    Config.VAL_META_FILE = meta_val_small
    Config.TEST_META_FILE = meta_test_small

    # Update derived paths
    Config.MODEL_PATH = os.path.join(demo_dir, "model_demo.pth")
    Config.VOCAB_PATH = os.path.join(demo_dir, "vocab.json")
    Config.MLB_PATH = os.path.join(demo_dir, "mlb.joblib")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    Config.TRAIN_TOKENS_PATH = os.path.join(demo_dir, "train_tokens.npy")
    Config.VAL_TOKENS_PATH = os.path.join(demo_dir, "val_tokens.npy")
    Config.TEST_TOKENS_PATH = os.path.join(demo_dir, "test_tokens.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(demo_dir, "train_labels.npy")
    Config.VAL_LABELS_PATH = os.path.join(demo_dir, "val_labels.npy")

    # Optimize Hyperparameters for Speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.VOCAB_SIZE = 1000
    Config.MAX_LEN = 50
    Config.NUM_TAGS = 50  # Predict only top 50 tags
    Config.EMBED_DIM = 64
    Config.CNN_FILTERS = 32

    # Re-run setup to create directories
    Config.setup()


def main():
    # 1. Reproducibility
    seed_everything(42)

    # 2. Prepare Environment
    setup_demo_environment()

    # 3. Data Loading
    # force reload to use our small datasets
    print("\n--- Initializing Data Loaders ---")
    train_loader, val_loader, test_loader, mlb = get_dataloaders(load_cached_data=False)

    # Validation: Check batch shapes
    sample_batch, sample_labels = next(iter(train_loader))
    print(
        f"Train Batch Shape: Inputs={sample_batch.shape}, Labels={sample_labels.shape}"
    )
    assert sample_batch.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Incorrect input shape"
    assert sample_labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_TAGS,
    ), "Incorrect label shape"

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = WideAndDeepTextCNN(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_tags=Config.NUM_TAGS,
        cnn_filters=Config.CNN_FILTERS,
        cnn_kernel_sizes=[3, 4],
        dropout=0.2,
    )

    # 5. Setup Training Components
    criterion = FocalLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # 6. Training Loop
    print("\n--- Starting Training ---")
    trainer.fit(num_epochs=Config.NUM_EPOCHS, patience=1)

    # Validation: Check if model file was created
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."

    # 7. Prediction
    print("\n--- Generating Predictions ---")
    # Load best model state (simulating inference phase)
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    probs = trainer.predict(test_loader)

    # Validation: Check prediction shape
    assert probs.shape == (
        len(test_loader.dataset),
        Config.NUM_TAGS,
    ), "Prediction shape mismatch"

    # 8. Create Submission
    print("\n--- creating Submission File ---")
    # Use the best threshold found during validation
    threshold = checkpoint.get("best_threshold", 0.5)
    print(f"Using threshold: {threshold}")

    preds_bin = (probs >= threshold).astype(int)
    predicted_tags = mlb.inverse_transform(preds_bin)

    # Load test IDs from metadata
    df_test_meta = pd.read_csv(Config.TEST_META_FILE)
    submission_ids = df_test_meta["Id"].values

    # Format tags as space-delimited strings
    formatted_tags = [
        " ".join(tags) if tags else "python" for tags in predicted_tags
    ]  # Default to 'python' if empty for safety

    submission_df = pd.DataFrame({"Id": submission_ids, "Tags": formatted_tags})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file missing"
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(saved_df) == len(df_test_meta), "Submission row count mismatch"
    assert (
        "Id" in saved_df.columns and "Tags" in saved_df.columns
    ), "Submission columns mismatch"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
