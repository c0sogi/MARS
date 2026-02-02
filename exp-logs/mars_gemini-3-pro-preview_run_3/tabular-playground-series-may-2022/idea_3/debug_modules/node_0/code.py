import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import preprocess_features, ManufacturingDataset
from library.model import DenoisingAutoencoder, ManufacturingClassifier, Encoder
from library.train import pretrain_dae, train_classifier, generate_submission


def main():
    print("Initializing Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We override Config attributes to run on a small subset of data for demonstration
    # and to ensure the script completes quickly.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Model checkpoints in demo dir
    Config.PRETRAINED_MODEL_PATH = os.path.join(DEMO_DIR, "dae_autoencoder.pth")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_classifier.pth")

    # Training params: Limit epochs and batch size for speed
    Config.PRETRAIN_EPOCHS = 1
    Config.FINETUNE_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.LOAD_CACHED_DATA = False  # Force reprocessing to demonstrate logic

    # Setup directories
    Config.setup()

    print("Configuration updated for demo execution.")

    # -------------------------------------------------------------------------
    # 2. Create Mini-Datasets
    # -------------------------------------------------------------------------
    print("\nCreating mini-datasets (1000 samples each)...")

    # Define source paths (existing metadata)
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Define dest paths (demo)
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")

    # Read head and save to simulate a smaller dataset
    pd.read_csv(orig_train_path, nrows=1000).to_csv(demo_train_path, index=False)
    pd.read_csv(orig_val_path, nrows=1000).to_csv(demo_val_path, index=False)
    pd.read_csv(orig_test_path, nrows=1000).to_csv(demo_test_path, index=False)

    # Update Config to point to these new files
    Config.TRAIN_PATH = demo_train_path
    Config.VAL_PATH = demo_val_path
    Config.TEST_PATH = demo_test_path

    assert os.path.exists(Config.TRAIN_PATH), "Demo train file creation failed"

    # -------------------------------------------------------------------------
    # 3. Demonstrate Data Processing
    # -------------------------------------------------------------------------
    print("\n--- Running Data Processing ---")
    # This function loads raw CSVs, engineers features, normalizes, and encodes
    train_data, val_data, test_data, metadata = preprocess_features(
        load_cached_data=False
    )

    # Verification
    print("Verifying processed data...")
    assert isinstance(train_data, dict)
    assert "cont" in train_data and "cat" in train_data
    # Check shape matches our mini-dataset size
    assert train_data["cont"].shape[0] == 1000
    assert train_data["cat"].shape[0] == 1000
    assert "cat_cardinalities" in metadata
    assert "cont_dim" in metadata

    print(f"Continuous Feature Dim: {metadata['cont_dim']}")
    print(f"Categorical Cardinalities: {metadata['cat_cardinalities']}")
    print("Data processing verification passed.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Dataset Class
    # -------------------------------------------------------------------------
    print("\n--- Testing Dataset Class ---")

    # Test Supervised Mode (Standard (X, y) pairs)
    ds_sup = ManufacturingDataset(
        train_data["cont"], train_data["cat"], train_data["target"], mode="supervised"
    )
    sample_sup = ds_sup[0]
    # Expect ((cont, cat), target)
    assert len(sample_sup) == 2
    assert len(sample_sup[0]) == 2
    assert torch.is_tensor(sample_sup[0][0])  # cont tensor
    assert torch.is_tensor(sample_sup[1])  # target tensor

    # Test Pretrain Mode (Swap Noise Augmentation)
    ds_pre = ManufacturingDataset(
        train_data["cont"], train_data["cat"], mode="pretrain"
    )
    sample_pre = ds_pre[0]
    # Expect ((noisy_cont, noisy_cat), (clean_cont, clean_cat))
    assert len(sample_pre) == 2
    assert len(sample_pre[0]) == 2  # noisy inputs
    assert len(sample_pre[1]) == 2  # clean targets

    print("Dataset class verification passed.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Model Architecture
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Architecture ---")
    device = get_device()

    # Instantiate DAE (Denoising Autoencoder)
    dae = DenoisingAutoencoder(metadata["cont_dim"], metadata["cat_cardinalities"]).to(
        device
    )

    # Create dummy batch for verification
    batch_size = 4
    dummy_cont = torch.randn(batch_size, metadata["cont_dim"]).to(device)

    # Generate random categorical indices within range for each feature
    dummy_cat_list = []
    for card in metadata["cat_cardinalities"]:
        dummy_cat_list.append(torch.randint(0, card, (batch_size, 1)))
    dummy_cat = torch.cat(dummy_cat_list, dim=1).to(device)

    # Forward Pass DAE
    rec_cont, rec_cats = dae(dummy_cont, dummy_cat)

    # Verify DAE Output Shapes
    assert rec_cont.shape == dummy_cont.shape
    assert len(rec_cats) == len(metadata["cat_cardinalities"])
    assert rec_cats[0].shape == (batch_size, metadata["cat_cardinalities"][0])

    # Instantiate Classifier using the DAE's encoder
    classifier = ManufacturingClassifier(dae.encoder).to(device)
    logits = classifier(dummy_cont, dummy_cat)

    # Verify Classifier Output Shape
    assert logits.shape == (batch_size, 1)

    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Training Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Running Training Pipeline (Fast Mode) ---")

    # Step A: Pretrain DAE (Unsupervised)
    print("1. Pretraining DAE...")
    encoder = pretrain_dae(train_data, val_data, test_data, metadata)

    # Verify encoder is returned and checkpoint exists
    assert isinstance(encoder, Encoder)
    assert os.path.exists(Config.PRETRAINED_MODEL_PATH)

    # Step B: Train Classifier (Supervised Fine-tuning)
    print("2. Fine-tuning Classifier...")
    best_auc = train_classifier(train_data, val_data, metadata, encoder)

    print(f"Training finished with Best AUC: {best_auc}")
    assert isinstance(best_auc, float)
    assert os.path.exists(Config.BEST_MODEL_PATH)

    # Step C: Generate Submission
    print("3. Generating Submission...")
    generate_submission(test_data, metadata)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH)
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_sub.shape == (1000, 2)
    assert list(df_sub.columns) == ["id", "target"]

    print("Pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    seed_everything(42)
    main()
