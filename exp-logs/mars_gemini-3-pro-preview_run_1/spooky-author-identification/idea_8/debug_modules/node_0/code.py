import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import get_dataloaders
from library.features import get_meta_features, get_tfidf_features
from library.modeling import DebertaClassifier
from library.engine import run_transformer_fold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("--- Starting Demonstration of Author Identification Pipeline ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Update working directory for this demo to avoid overwriting real runs
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Create necessary directories
    Config.create_directories()

    # Set seeds
    seed_everything(Config.SEED)
    print("    Configuration updated. Debug mode: ON.")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data and Initializing Loaders...")

    # This function loads metadata, subsamples (due to DEBUG=True), and returns loaders
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(debug=True)

    # Verify DataLoader output
    print("    Verifying DataLoader batch structure...")
    sample_batch = next(iter(train_loader))

    # Check keys
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch

    # Check shapes
    batch_size = sample_batch["input_ids"].shape[0]
    seq_len = sample_batch["input_ids"].shape[1]

    assert (
        batch_size == Config.TRAIN_BATCH_SIZE
    ), f"Expected batch size {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
    assert (
        seq_len == Config.MAX_LEN
    ), f"Expected sequence length {Config.MAX_LEN}, got {seq_len}"

    print(f"    Batch verification passed. Shape: {sample_batch['input_ids'].shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Generation (Expert B & Meta-Features)
    # -------------------------------------------------------------------------
    print("\n[3] Generating Features (TF-IDF & Meta)...")

    # Load the specific subsampled dataframes used by the loaders for consistency
    # (In a real run, we'd load the full CSVs, but here we match the debug size)
    train_df = pd.read_csv(Config.TRAIN_META_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    val_df = pd.read_csv(Config.VAL_META_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    test_df = pd.read_csv(Config.TEST_META_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # 3a. Meta Features
    print("    Computing Meta-Features (Length, Punctuation)...")
    meta_train, meta_val, meta_test = get_meta_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    assert meta_train.shape == (Config.DEBUG_SAMPLE_SIZE, 3)
    assert meta_val.shape == (Config.DEBUG_SAMPLE_SIZE, 3)
    print(f"    Meta-features shape verified: {meta_train.shape}")

    # 3b. TF-IDF Features (Expert B)
    print("    Computing Hybrid TF-IDF Features...")
    tfidf_train, tfidf_val, tfidf_test = get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Check if it is a sparse matrix and has correct rows
    assert tfidf_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    print(f"    TF-IDF train shape verified: {tfidf_train.shape}")

    # -------------------------------------------------------------------------
    # 4. Expert B Model Demonstration (Logistic Regression)
    # -------------------------------------------------------------------------
    print("\n[4] Training Expert B (Logistic Regression on TF-IDF)...")

    # Prepare labels
    label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
    y_train = train_df["author"].map(label_map).values
    y_val = val_df["author"].map(label_map).values

    # Initialize and fit
    clf = LogisticRegression(
        C=Config.LOGREG_C,
        solver=Config.LOGREG_SOLVER,
        max_iter=100,  # Reduced for demo
        n_jobs=1,
    )
    clf.fit(tfidf_train, y_train)

    # Predict
    expert_b_preds = clf.predict_proba(tfidf_val)

    # Validate Score
    expert_b_loss = calculate_log_loss(y_val, expert_b_preds)
    print(f"    Expert B (TF-IDF) Validation Log Loss: {expert_b_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Expert A Model Demonstration (Transformer)
    # -------------------------------------------------------------------------
    print("\n[5] Initializing Expert A (DeBERTa w/ Weighted Layer Pooling)...")

    device = torch.device(Config.DEVICE)
    model = DebertaClassifier(Config.MODEL_NAME, num_classes=3)
    model.to(device)

    # 5a. Forward Pass Check
    print("    Running dummy forward pass...")
    ids = sample_batch["input_ids"].to(device)
    mask = sample_batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(ids, mask)

    assert outputs.shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, 3)}, got {outputs.shape}"
    print("    Forward pass successful.")

    # Clean up to save memory before training loop
    del model, ids, mask, outputs
    torch.cuda.empty_cache()

    # 5b. Training Loop (Single Fold)
    print(f"    Starting training loop for Fold 0 ({Config.EPOCHS} epoch)...")

    # run_transformer_fold handles model init, optimizer, scheduler, training, and validation
    best_loss, best_preds = run_transformer_fold(0, train_loader, val_loader)

    print(f"    Training complete. Best Validation Loss: {best_loss:.4f}")

    # Verify predictions shape
    expected_val_size = len(val_loader.dataset)
    assert best_preds.shape == (
        expected_val_size,
        3,
    ), f"Prediction shape mismatch. Expected {(expected_val_size, 3)}, got {best_preds.shape}"

    # -------------------------------------------------------------------------
    # 6. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Metric Calculation...")

    # Extract actual targets from validation loader to ensure alignment
    val_targets = []
    for batch in val_loader:
        val_targets.extend(batch["labels"].numpy())
    val_targets = np.array(val_targets)

    # Recalculate loss using the utility function
    calculated_loss = calculate_log_loss(val_targets, best_preds)

    # Allow small float tolerance
    assert (
        abs(calculated_loss - best_loss) < 1e-5
    ), "Metric calculation discrepancy between training loop and manual check."

    print(f"    Manual metric check passed. Loss: {calculated_loss:.4f}")

    # -------------------------------------------------------------------------
    # 7. Checkpointing Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Artifacts...")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "expert_a_fold_0.pt")
    if os.path.exists(checkpoint_path):
        print(f"    Checkpoint found at: {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"    Checkpoint size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
