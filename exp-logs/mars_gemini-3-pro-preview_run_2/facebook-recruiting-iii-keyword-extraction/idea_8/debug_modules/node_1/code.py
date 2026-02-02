import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, FocalLoss
from library.data_processing import prepare_data, get_dataloaders
from library.model import WideAndDeepModel
from library.trainer import Trainer
from library.inference import find_optimal_threshold, generate_submission
import scipy.sparse

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # --------------------------------------------------------------------------
    # We monkey-patch the Config class to use a demo directory and small data subset
    # so the script runs quickly and doesn't overwrite main experiment files.

    print("Configuring environment for demo run...")

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.TAG_ENCODER_PATH = os.path.join(DEMO_DIR, "tag_encoder.json")
    Config.TFIDF_VECTORIZER_PATH = os.path.join(DEMO_DIR, "tfidf_vectorizer.joblib")
    Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.json")

    Config.TRAIN_WIDE_PATH = os.path.join(DEMO_DIR, "train_wide.npz")
    Config.TRAIN_DEEP_PATH = os.path.join(DEMO_DIR, "train_deep.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(DEMO_DIR, "train_labels.npz")

    Config.VAL_WIDE_PATH = os.path.join(DEMO_DIR, "val_wide.npz")
    Config.VAL_DEEP_PATH = os.path.join(DEMO_DIR, "val_deep.npy")
    Config.VAL_LABELS_PATH = os.path.join(DEMO_DIR, "val_labels.npz")

    Config.TEST_WIDE_PATH = os.path.join(DEMO_DIR, "test_wide.npz")
    Config.TEST_DEEP_PATH = os.path.join(DEMO_DIR, "test_deep.npy")
    Config.TEST_IDS_PATH = os.path.join(DEMO_DIR, "test_ids.npy")

    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Hyperparameters for Speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500  # Small subset for demo
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}, Size: {Config.DEBUG_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------------------------------
    print("\n--- Step 1: Data Processing ---")

    # Run data preparation (generates .npy/.npz files in DEMO_DIR)
    prepare_data(load_cached_data=False, debug=Config.DEBUG)

    # Verify files were created
    assert os.path.exists(Config.TRAIN_WIDE_PATH), "Train Wide data not found"
    assert os.path.exists(Config.TAG_ENCODER_PATH), "Tag Encoder not found"

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Update Config with actual data dimensions to handle debug/small data scenarios
    # Cite {debug_lesson_2}
    actual_wide_dim = train_loader.dataset.wide_data.shape[1]
    actual_num_classes = train_loader.dataset.labels.shape[1]
    Config.TFIDF_MAX_FEATURES = actual_wide_dim
    Config.NUM_CLASSES = actual_num_classes
    print(
        f"Config updated from data: TFIDF_MAX_FEATURES={actual_wide_dim}, NUM_CLASSES={actual_num_classes}"
    )

    # Verify DataLoader shapes
    sample_batch = next(iter(train_loader))
    deep_shape = sample_batch["deep"].shape
    wide_shape = sample_batch["wide"].shape
    label_shape = sample_batch["label"].shape

    print(
        f"Batch Shapes -> Deep: {deep_shape}, Wide: {wide_shape}, Label: {label_shape}"
    )

    assert deep_shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Incorrect deep feature shape"
    assert wide_shape == (
        Config.BATCH_SIZE,
        Config.TFIDF_MAX_FEATURES,
    ), "Incorrect wide feature shape"
    assert label_shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label shape"

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Model Initialization ---")

    device = torch.device(Config.DEVICE)
    model = WideAndDeepModel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        wide_dim=Config.TFIDF_MAX_FEATURES,
        num_classes=Config.NUM_CLASSES,
        filter_sizes=Config.FILTER_SIZES,
        num_filters=Config.NUM_FILTERS,
        attention_dim=Config.ATTENTION_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_deep = sample_batch["deep"].to(device)
        dummy_wide = sample_batch["wide"].to(device)
        logits = model(dummy_deep, dummy_wide)

    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Training ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=Config.USE_AMP,
    )

    # Train for 1 epoch (as configured)
    best_f1 = trainer.fit(epochs=Config.EPOCHS)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved"
    print(f"Training finished. Best F1: {best_f1:.4f}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Inference & Submission ---")

    # Load validation labels for threshold tuning
    # Note: In debug mode, prepare_data saves sliced data, so we load that.
    val_labels_sparse = scipy.sparse.load_npz(Config.VAL_LABELS_PATH)

    # Find optimal threshold
    best_threshold = find_optimal_threshold(trainer, val_loader, val_labels_sparse)

    # Generate submission
    generate_submission(trainer, test_loader, best_threshold)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head()}")

    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns missing"
    assert (
        len(df_sub) == Config.DEBUG_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SIZE}, got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
