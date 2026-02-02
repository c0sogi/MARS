import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed
from library.data_processing import get_dataloaders
from library.model import NBOWModel
from library.trainer import Trainer, run_inference


def main():
    print("=== Starting Demo Pipeline ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[Step 1] Configuring environment for demo...")

    # Override Config for a fast, lightweight demo run
    demo_dir = "./working/demo_pipeline"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.VOCAB_PATH = os.path.join(demo_dir, "vocab.json")
    Config.MLB_PATH = os.path.join(demo_dir, "mlb.joblib")
    Config.TRAIN_TOKENS_PATH = os.path.join(
        demo_dir, "train_features.npz"
    )  # Using .npz for demo names, though code uses .npy
    Config.TRAIN_OFFSETS_PATH = os.path.join(demo_dir, "train_offsets.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(demo_dir, "train_labels.npy")
    Config.VAL_TOKENS_PATH = os.path.join(demo_dir, "val_tokens.npy")
    Config.VAL_OFFSETS_PATH = os.path.join(demo_dir, "val_offsets.npy")
    Config.VAL_LABELS_PATH = os.path.join(demo_dir, "val_labels.npy")
    Config.TEST_TOKENS_PATH = os.path.join(demo_dir, "test_tokens.npy")
    Config.TEST_OFFSETS_PATH = os.path.join(demo_dir, "test_offsets.npy")
    Config.TEST_IDS_PATH = os.path.join(demo_dir, "test_ids.npy")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset
    Config.VOCAB_SIZE = 1000
    Config.NUM_TAGS = 50
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 2
    Config.EMBED_DIM = 64
    Config.HIDDEN_DIM = 64
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    set_seed(Config.SEED)
    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print("\n[Step 2] Loading and processing data...")

    # Force processing from scratch by setting load_cached_data=False
    train_loader, val_loader, test_loader, mlb = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=False
    )

    # Verify DataLoaders
    print("Verifying DataLoader shapes...")
    batch = next(iter(train_loader))
    inputs, offsets, targets = batch

    # Assertions
    assert isinstance(inputs, torch.Tensor), "Inputs should be a Tensor"
    assert isinstance(offsets, torch.Tensor), "Offsets should be a Tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a Tensor"

    # Check dimensions
    # offsets length should equal batch size (or smaller if last batch)
    assert offsets.size(0) == inputs.size(0) or offsets.size(0) <= Config.BATCH_SIZE
    # targets shape should be (batch_size, num_classes)
    assert targets.dim() == 2, "Targets should be 2D"
    assert targets.size(1) == len(mlb.classes_), "Target columns must match num classes"

    print(
        f"Batch verification passed. Input size: {inputs.size()}, Targets size: {targets.size()}"
    )
    print(f"Vocabulary size used: {Config.VOCAB_SIZE}")
    print(f"Number of tags (classes): {len(mlb.classes_)}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Step 3] Initializing NBOW Model...")

    # We need to know the actual vocab size used (it might be smaller than max if data is small)
    # Since Vocabulary logic is encapsulated, we rely on Config.VOCAB_SIZE or load the json.
    # However, EmbeddingBag handles indices < num_embeddings.

    model = NBOWModel(
        vocab_size=Config.VOCAB_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=len(mlb.classes_),
        dropout=Config.DROPOUT,
    )

    # Move model to device
    model.to(Config.DEVICE)

    # Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print("Model initialized successfully.")

    # ==========================================
    # 4. Training
    # ==========================================
    print("\n[Step 4] Starting Training...")

    trainer = Trainer(model, optimizer, criterion, device=Config.DEVICE)

    best_f1 = trainer.fit(
        train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=1
    )

    print(f"Training completed. Best Val F1: {best_f1:.4f}")

    # Assert that training produced a valid score
    assert isinstance(best_f1, float), "Best F1 score should be a float"
    assert 0.0 <= best_f1 <= 1.0, "F1 score must be between 0 and 1"

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 5] Running Inference and Generating Submission...")

    run_inference(
        trainer,
        test_loader,
        mlb,
        threshold=0.2,  # Lower threshold for demo to ensure some tags are predicted
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Verify Schema
    assert list(df_sub.columns) == ["Id", "Tags"], "Submission columns must be Id, Tags"
    assert len(df_sub) > 0, "Submission file is empty"

    # Verify Id consistency with test loader
    # Note: Since we used debug sampling, the count matches the sample size
    # We just check if it's not empty and IDs are integers
    assert pd.api.types.is_integer_dtype(df_sub["Id"]), "Id column should be integer"

    print("\n=== Demo Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
