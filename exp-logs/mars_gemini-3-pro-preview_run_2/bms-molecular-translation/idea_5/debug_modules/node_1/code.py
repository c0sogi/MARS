import os
import sys
import importlib
import library.train
import library.model

# Cite debug_lesson_1: Reload modules to ensure patches are picked up in persistent environments
importlib.reload(library.train)
importlib.reload(library.model)

import shutil
import numpy as np
import pandas as pd
import torch
import random

# Import from library
from library.config import Config
from library.utils import parse_inchi_stoichiometry, get_atom_vector
from library.dataset import ChemicalDataset
from library.model import StoichiometryEncoder
from library.train import run_training
from library.retrieval import run_retrieval_inference


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Initializing Demonstration Script...")
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed/Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.VAL_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Use a specific working directory for this demo to avoid side effects
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.TRAIN_EMBEDDINGS_PATH = os.path.join(
        Config.WORKING_DIR, "train_embeddings.npy"
    )
    Config.TRAIN_LABELS_CACHE_PATH = os.path.join(
        Config.WORKING_DIR, "train_labels_cache.npy"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.setup_directories()
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    test_inchi = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    print(f"Testing parsing for: {test_inchi}")

    # Test stoichiometry parsing
    counts = parse_inchi_stoichiometry(test_inchi)
    print(f"Parsed counts: {counts}")
    assert counts["C"] == 6, "Incorrect Carbon count"
    assert counts["H"] == 6, "Incorrect Hydrogen count"

    # Test vector generation
    vector = get_atom_vector(test_inchi)
    print(f"Generated atom vector: {vector}")

    # Check indices based on Config.ATOM_LIST = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I"]
    # C is index 0, H is index 1
    assert vector[0] == 6.0, "Vector index 0 (C) should be 6.0"
    assert vector[1] == 6.0, "Vector index 1 (H) should be 6.0"
    assert vector[2] == 0.0, "Vector index 2 (N) should be 0.0"
    assert vector.shape == (
        Config.NUM_ATOMS,
    ), f"Vector shape mismatch. Expected ({Config.NUM_ATOMS},), got {vector.shape}"

    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Class
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset Class...")

    # Load a tiny slice of metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH).head(10)

    # Instantiate dataset
    ds_train = ChemicalDataset(df_train_meta, mode="train")
    print(f"Dataset length: {len(ds_train)}")

    # Fetch one item
    img, target = ds_train[0]
    print(f"Sample 0 image shape: {img.shape}")
    print(f"Sample 0 target: {target}")

    # Validations
    assert len(ds_train) == 10, "Dataset length mismatch"
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"
    assert target.shape == (Config.NUM_ATOMS,), "Target tensor shape mismatch"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = StoichiometryEncoder(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Speed up, don't download weights if not needed for shape check
        embedding_dim=Config.EMBEDDING_DIM,
        num_atoms=Config.NUM_ATOMS,
    )
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )

    with torch.no_grad():
        embedding, preds = model(dummy_input)

    print(f"Output Embedding shape: {embedding.shape}")
    print(f"Output Predictions shape: {preds.shape}")

    assert embedding.shape == (
        2,
        Config.EMBEDDING_DIM,
    ), "Embedding output shape mismatch"
    assert preds.shape == (2, Config.NUM_ATOMS), "Prediction output shape mismatch"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Run Training Loop (Integration Test)
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Integration Test)...")

    # Using the library function run_training
    # This will use the Config settings we overrode earlier (DEBUG=True, EPOCHS=1, etc.)
    run_training(debug=True, epochs=Config.EPOCHS)

    # Verify model was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"Training successful. Model saved at {Config.MODEL_PATH}")
    else:
        # If validation loss didn't improve (unlikely with 1 epoch starting from infinity),
        # the code might not save. However, the logic says if val_loss < best (inf), save.
        # So it should save.
        raise AssertionError(
            f"Model file not found at {Config.MODEL_PATH} after training."
        )

    # -------------------------------------------------------------------------
    # 6. Run Retrieval Inference (Integration Test)
    # -------------------------------------------------------------------------
    print("\n[6] Running Retrieval Inference (Integration Test)...")

    # We need to ensure we don't use cached embeddings from a previous run if we want to test the full pipeline
    if os.path.exists(Config.TRAIN_EMBEDDINGS_PATH):
        os.remove(Config.TRAIN_EMBEDDINGS_PATH)

    run_retrieval_inference(
        model_path=Config.MODEL_PATH,
        batch_size=Config.VAL_BATCH_SIZE,
        device=Config.DEVICE,
        load_cached_index=False,  # Force rebuild index
        debug=True,
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Inference successful. Submission saved at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        assert (
            len(df_sub) == Config.DEBUG_SAMPLE_SIZE
        ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"
    else:
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
