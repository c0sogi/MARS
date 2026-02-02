import os
import sys
import torch
import pandas as pd
import warnings
import logging
from transformers import logging as hf_logging

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.models import LocatorModel, InfillerModel
from library.trainer import Trainer
from library.inference import BeamSearchPipeline


def run_demo():
    # --------------------------------------------------------------------------
    # 0. Setup & Configuration Overrides
    # --------------------------------------------------------------------------
    print(">>> [Step 0] Configuring environment for fast demo run...")

    # Suppress verbose logs for clean output
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    logging.getLogger("Trainer").setLevel(logging.WARNING)
    logging.getLogger("Inference").setLevel(logging.WARNING)

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update derived paths to point to the demo directory
    Config.LOCATOR_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "locator_train_processed.parquet"
    )
    Config.LOCATOR_VAL_CACHE = os.path.join(
        Config.WORKING_DIR, "locator_val_processed.parquet"
    )
    Config.INFILLER_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "infiller_train.parquet"
    )
    Config.INFILLER_VAL_CACHE = os.path.join(Config.WORKING_DIR, "infiller_val.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.parquet")
    Config.BEST_LOCATOR_PATH = os.path.join(Config.WORKING_DIR, "best_locator.pth")
    Config.BEST_INFILLER_PATH = os.path.join(Config.WORKING_DIR, "best_infiller.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for fast execution
    Config.EPOCHS = 1
    Config.TRAIN_SIZE = 500  # Small subset for training
    Config.VAL_SIZE = 100  # Small subset for validation
    Config.DEBUG_SIZE = 200  # Small subset for test/debug
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.BEAM_WIDTH = 2  # Reduced beam width for speed

    # Initialize environment (creates dirs, sets seeds)
    Config.initialize()
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    print("\n>>> [Step 1] Loading DataLoaders...")

    # get_dataloaders handles caching, tokenization, and dataset creation
    loaders = get_dataloaders(debug=True)

    # Validate Loader Keys
    expected_keys = [
        "train_locator",
        "val_locator",
        "train_infiller",
        "val_infiller",
        "test",
        "tokenizer_locator",
        "tokenizer_infiller",
    ]
    for k in expected_keys:
        assert k in loaders, f"Missing key {k} in loaders dict"

    # Validate Batch Structure (Locator)
    sample_batch_loc = next(iter(loaders["train_locator"]))
    assert "input_ids" in sample_batch_loc
    assert "labels" in sample_batch_loc
    assert sample_batch_loc["input_ids"].shape[0] == Config.BATCH_SIZE
    print("DataLoaders initialized and verified successfully.")

    # --------------------------------------------------------------------------
    # 2. Model Logic Verification
    # --------------------------------------------------------------------------
    print("\n>>> [Step 2] Verifying Model Architectures...")

    device = Config.DEVICE

    # Test Locator Model
    locator = LocatorModel().to(device)
    loc_input = sample_batch_loc["input_ids"].to(device)
    loc_mask = sample_batch_loc["attention_mask"].to(device)

    with torch.no_grad():
        loc_out = locator(loc_input, loc_mask)

    # Locator output should be (Batch, Seq_Len) - logits per token
    assert loc_out.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Locator output shape mismatch. Expected {(Config.BATCH_SIZE, Config.MAX_LEN)}, got {loc_out.shape}"
    print("LocatorModel forward pass verified.")

    # Test Infiller Model
    infiller = InfillerModel().to(device)

    # Use correct input from infiller loader to avoid vocabulary mismatch (DeBERTa vs RoBERTa)
    sample_batch_inf = next(iter(loaders["train_infiller"]))
    inf_input = sample_batch_inf["input_ids"].to(device)
    inf_mask = sample_batch_inf["attention_mask"].to(device)

    # Note: Infiller expects labels for loss calculation in forward, or none for inference
    with torch.no_grad():
        inf_out = infiller(inf_input, inf_mask)

    # Infiller output is MaskedLMOutput, contains logits of shape (Batch, Seq_Len, Vocab_Size)
    vocab_size = loaders["tokenizer_infiller"].vocab_size
    assert inf_out.logits.shape[:2] == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Infiller logits shape mismatch (batch/seq dims)."
    print("InfillerModel forward pass verified.")

    # --------------------------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------------------------
    print("\n>>> [Step 3] Running Training Loop (Fast Mode)...")

    trainer = Trainer()

    # Train Locator
    print("Training Locator...")
    trainer.train_locator(loaders["train_locator"], loaders["val_locator"])
    assert os.path.exists(
        Config.BEST_LOCATOR_PATH
    ), "Locator model checkpoint not found after training."

    # Train Infiller
    print("Training Infiller...")
    trainer.train_infiller(loaders["train_infiller"], loaders["val_infiller"])
    assert os.path.exists(
        Config.BEST_INFILLER_PATH
    ), "Infiller model checkpoint not found after training."

    print("Training complete. Checkpoints saved.")

    # --------------------------------------------------------------------------
    # 4. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n>>> [Step 4] Running Inference Pipeline...")

    pipeline = BeamSearchPipeline()

    # Test prediction logic on a single batch first (internal verification)
    test_loader = loaders["test"]

    # Run full submission generation
    print("Generating submission file...")
    pipeline.generate_submission(test_loader)

    # --------------------------------------------------------------------------
    # 5. Final Validation
    # --------------------------------------------------------------------------
    print("\n>>> [Step 5] Validating Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check file content format
    # Expected: id,"sentence"
    # We use pandas to read it back and verify
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "sentence" in df_sub.columns, "Submission missing 'sentence' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if sentences look reasonable (basic check: string type and length)
    sample_sent = df_sub.iloc[0]["sentence"]
    assert isinstance(sample_sent, str), "Sentence column does not contain strings"
    assert len(sample_sent) > 5, "Sentence seems too short to be valid"

    print(f"Submission verified. Contains {len(df_sub)} rows.")
    print(f"File location: {Config.SUBMISSION_PATH}")
    print("\n>>> Demo Run Completed Successfully.")


if __name__ == "__main__":
    run_demo()
