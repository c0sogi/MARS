import os
import sys
import pandas as pd
import torch
import torch.optim as optim
import shutil
import time

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import (
    prepare_artifacts,
    process_data,
    TextNormalizationDataset,
    get_weighted_dataloader,
    get_dataloader,
)
from library.model import TransformerNumNorm
from library.trainer import Trainer
from library.inference import Predictor, generate_submission


def create_subset_data(source_path, dest_path, n_rows=1000):
    """
    Helper to create a small subset of data for demonstration speed.
    """
    print(f"Creating subset of {source_path} -> {dest_path} ({n_rows} rows)")
    # Read only the first n_rows
    # We read as string to preserve formatting, similar to the main pipeline
    df = pd.read_csv(source_path, dtype=str, nrows=n_rows)
    df.to_csv(dest_path, index=False)


if __name__ == "__main__":
    # 1. Setup & Configuration Overrides
    print("--- 1. Setup & Configuration ---")
    set_seed(42)

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config paths to point to our demo directory
    # This prevents modifying the original 'idea_2' directory and ensures isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.VOCAB_FILE = os.path.join(DEMO_DIR, "vocab.npy")
    Config.CLASS_MAP_FILE = os.path.join(DEMO_DIR, "class_map.npy")
    Config.MODEL_CHECKPOINT = os.path.join(DEMO_DIR, "model_checkpoint.pt")
    Config.TRAIN_PROCESSED = os.path.join(DEMO_DIR, "train_processed.parquet")
    Config.VAL_PROCESSED = os.path.join(DEMO_DIR, "val_processed.parquet")
    Config.TEST_PROCESSED = os.path.join(DEMO_DIR, "test_processed.parquet")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Override Training Hyperparameters for Speed
    Config.DEBUG_SAMPLE_SIZE = None  # We will manually subset files instead
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # 2. Create Data Subsets
    # We create mini versions of the metadata files to make prepare_artifacts fast
    print("\n--- 2. Creating Data Subsets ---")

    # Define paths for mini datasets
    mini_train_path = os.path.join(DEMO_DIR, "train_mini.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val_mini.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test_mini.csv")

    # Create subsets from the provided metadata
    # Note: We assume the metadata files exist as per the problem description
    create_subset_data(
        os.path.join("./metadata", "train.csv"), mini_train_path, n_rows=500
    )
    create_subset_data(os.path.join("./metadata", "val.csv"), mini_val_path, n_rows=100)
    create_subset_data(
        os.path.join("./metadata", "test.csv"), mini_test_path, n_rows=100
    )

    # Point Config to these new mini files
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    # 3. Data Processing
    print("\n--- 3. Data Processing ---")

    # Generate Vocab and Class Map from the mini train set
    tokenizer, class_map = prepare_artifacts(load_cached_data=False)
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Classes: {list(class_map.keys())}")

    # Process Data into Parquet (Tokenization)
    df_train = process_data("train", tokenizer, class_map, load_cached_data=False)
    df_val = process_data("val", tokenizer, class_map, load_cached_data=False)

    # Create Datasets
    train_ds = TextNormalizationDataset(df_train)
    val_ds = TextNormalizationDataset(df_val)

    # Create DataLoaders
    train_loader = get_weighted_dataloader(
        train_ds, batch_size=Config.BATCH_SIZE, num_workers=0
    )
    val_loader = get_dataloader(val_ds, batch_size=Config.BATCH_SIZE, num_workers=0)

    # Validation: Check batch structure
    sample_batch = next(iter(train_loader))
    print(f"Sample Batch Keys: {sample_batch.keys()}")
    assert "src" in sample_batch
    assert "tgt_out" in sample_batch
    assert sample_batch["src"].shape == (Config.BATCH_SIZE, Config.MAX_SEQ_LEN)
    print("Data processing and loading verified.")

    # 4. Model Initialization
    print("\n--- 4. Model Initialization ---")

    # Initialize model with reduced capacity for demonstration speed
    model = TransformerNumNorm(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(class_map),
        d_model=32,  # Reduced from 256
        nhead=2,  # Reduced from 8
        num_encoder_layers=2,  # Reduced from 6
        num_decoder_layers=2,  # Reduced from 6
        dim_feedforward=64,  # Reduced from 1024
        dropout=0.1,
    )

    # Validation: Forward pass
    model.to(Config.DEVICE)
    src = sample_batch["src"].to(Config.DEVICE)
    tgt_in = sample_batch["tgt_in"].to(Config.DEVICE)

    with torch.no_grad():
        text_logits, class_logits = model(src, tgt_in)

    assert text_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
        tokenizer.vocab_size,
    )
    assert class_logits.shape == (Config.BATCH_SIZE, len(class_map))
    print("Model forward pass verified.")

    # 5. Training Loop
    print("\n--- 5. Training ---")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # Run training
    final_loss = trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Validation: Checkpoint creation
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not created!"
    print(f"Training complete. Checkpoint saved to {Config.MODEL_CHECKPOINT}")

    # 6. Inference & Submission
    print("\n--- 6. Inference ---")

    # Initialize Predictor (loads the checkpoint we just saved)
    predictor = Predictor(
        Config.MODEL_CHECKPOINT, tokenizer, class_map, device=Config.DEVICE
    )

    # Predict on a single batch from validation to verify output format
    val_batch = next(iter(val_loader))
    preds = predictor.predict_batch(val_batch, beam_width=1)  # Width 1 for speed

    print(f"Sample Predictions (first 3): {preds[:3]}")
    assert len(preds) == Config.BATCH_SIZE
    assert isinstance(preds[0], tuple) and len(preds[0]) == 2

    # Generate Submission File
    # This function processes the test set (mini_test.csv) and writes to CSV
    generate_submission(
        test_file=Config.TEST_CSV,
        submission_file=Config.SUBMISSION_FILE,
        model_path=Config.MODEL_CHECKPOINT,
        batch_size=Config.BATCH_SIZE,
    )

    # Validation: Submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created!"
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file shape: {df_sub.shape}")
    assert list(df_sub.columns) == ["id", "after"]
    assert len(df_sub) > 0

    print("\n=== Demonstration Completed Successfully ===")
