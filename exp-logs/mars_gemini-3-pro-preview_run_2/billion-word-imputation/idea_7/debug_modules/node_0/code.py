import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging
from functools import partial

# Disable tqdm globally to meet "no progress bars" requirement
from tqdm import tqdm


def noop_tqdm(*args, **kwargs):
    if "disable" not in kwargs:
        kwargs["disable"] = True
    return tqdm.__init__(*args, **kwargs)


tqdm.__init__ = partial(tqdm.__init__, disable=True)

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import (
    process_and_cache_data,
    get_dataloaders,
    TestDataset,
    LocatorDataset,
    InfillerDataset,
    VerifierDataset,
)
from library.models import LocatorModel, InfillerModel, VerifierModel
from library.engine import Trainer
from library.pipeline import InferencePipeline


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("1. Setting up configuration for fast demo execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.WORKING_DIR = "./working/demo_run"
    Config.SEED = 42

    # Reduce sizes to bare minimum for functional testing
    Config.DEBUG_TRAIN_SIZE = 32
    Config.DEBUG_VAL_SIZE = 16
    Config.LOCATOR_BATCH_SIZE = 4
    Config.INFILLER_BATCH_SIZE = 4
    Config.VERIFIER_BATCH_SIZE = 4

    # 1 Epoch is enough to demonstrate the loop
    Config.LOCATOR_EPOCHS = 1
    Config.INFILLER_EPOCHS = 1
    Config.VERIFIER_EPOCHS = 1

    # Update checkpoint paths to point to the demo directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.LOCATOR_CKPT_PATH = os.path.join(Config.WORKING_DIR, "best_locator.pth")
    Config.INFILLER_CKPT_PATH = os.path.join(Config.WORKING_DIR, "best_infiller.pth")
    Config.VERIFIER_CKPT_PATH = os.path.join(Config.WORKING_DIR, "best_verifier.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set seeds
    seed_everything(Config.SEED)

    # Suppress warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # -------------------------------------------------------------------------
    # 2. Data Processing and Caching
    # -------------------------------------------------------------------------
    print("\n2. Generating and validating datasets...")

    # Generate data
    loc_train, loc_val, inf_train, inf_val, ver_train, ver_val = process_and_cache_data(
        load_cached_data=False, debug_size=Config.DEBUG_TRAIN_SIZE  # Force regeneration
    )

    # Validate Locator Data
    assert not loc_train.empty, "Locator training data is empty"
    assert "gap_index" in loc_train.columns, "Locator data missing 'gap_index'"
    assert isinstance(
        loc_train.iloc[0]["words"], (list, np.ndarray)
    ), "Locator 'words' should be list/array"

    # Validate Infiller Data
    assert not inf_train.empty, "Infiller training data is empty"
    assert "masked_text" in inf_train.columns, "Infiller data missing 'masked_text'"
    assert (
        "<mask>" in inf_train.iloc[0]["masked_text"]
    ), "Infiller text missing <mask> token"

    # Validate Verifier Data
    assert not ver_train.empty, "Verifier training data is empty"
    assert "label" in ver_train.columns, "Verifier data missing 'label'"
    assert set(ver_train["label"].unique()).issubset(
        {0, 1}
    ), "Verifier labels must be 0 or 1"

    print("   -> Data generation successful. Shapes verified.")

    # -------------------------------------------------------------------------
    # 3. DataLoader Creation
    # -------------------------------------------------------------------------
    print("\n3. Creating DataLoaders...")

    # This function loads tokenizers and creates torch DataLoaders
    dataloaders = get_dataloaders(load_cached_data=True)

    # Verify dictionary structure
    assert "locator" in dataloaders
    assert "infiller" in dataloaders
    assert "verifier" in dataloaders
    assert "tokenizers" in dataloaders

    # Verify a batch from Locator loader
    train_loader_loc = dataloaders["locator"][0]
    batch = next(iter(train_loader_loc))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] <= Config.LOCATOR_BATCH_SIZE

    print("   -> DataLoaders created and batch structure verified.")

    # -------------------------------------------------------------------------
    # 4. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n4. Training Models (1 Epoch each)...")

    # --- Train Locator ---
    print("   [Locator] Initializing...")
    locator_model = LocatorModel(pretrained=True)
    loc_train_loader, loc_val_loader = dataloaders["locator"]

    print("   [Locator] Training...")
    Trainer.train_locator(locator_model, loc_train_loader, loc_val_loader, epochs=1)

    assert os.path.exists(Config.LOCATOR_CKPT_PATH), "Locator checkpoint was not saved."

    # --- Train Infiller ---
    print("   [Infiller] Initializing...")
    infiller_model = InfillerModel(pretrained=True)
    inf_train_loader, inf_val_loader = dataloaders["infiller"]

    print("   [Infiller] Training...")
    Trainer.train_infiller(infiller_model, inf_train_loader, inf_val_loader, epochs=1)

    assert os.path.exists(
        Config.INFILLER_CKPT_PATH
    ), "Infiller checkpoint was not saved."

    # --- Train Verifier ---
    print("   [Verifier] Initializing...")
    verifier_model = VerifierModel(pretrained=True)
    ver_train_loader, ver_val_loader = dataloaders["verifier"]

    print("   [Verifier] Training...")
    Trainer.train_verifier(verifier_model, ver_train_loader, ver_val_loader, epochs=1)

    assert os.path.exists(
        Config.VERIFIER_CKPT_PATH
    ), "Verifier checkpoint was not saved."

    # Clean up memory
    del locator_model, infiller_model, verifier_model
    torch.cuda.empty_cache()

    print("   -> All models trained and checkpoints saved.")

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Running Inference Pipeline...")

    # Create a small manual test set to avoid loading the huge test file
    test_data = [
        {
            "id": 1,
            "sentence": "The quick brown fox jumps over the dog .",
        },  # Missing 'lazy'
        {"id": 2, "sentence": "She sells sea shells by the shore ."},  # Missing 'sea'
        {"id": 3, "sentence": "To be or not to be that is the ."},  # Missing 'question'
    ]
    df_test_demo = pd.DataFrame(test_data)

    # Create DataLoader for inference
    tokenizer_loc = dataloaders["tokenizers"]["deberta"]
    test_ds = TestDataset(df_test_demo, tokenizer_loc, Config.MAX_LENGTH)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.LOCATOR_BATCH_SIZE, shuffle=False
    )

    # Initialize Pipeline (loads models from the checkpoints we just created)
    pipeline = InferencePipeline(load_models=True)

    # Run Inference
    results = pipeline.run_inference(test_loader)

    # Verify Results
    assert len(results) == len(test_data), "Result count mismatch"
    assert isinstance(results[0], tuple), "Result item should be a tuple"
    assert results[0][0] == 1, "ID mismatch in results"
    assert isinstance(results[0][1], str), "Predicted sentence should be a string"

    print("   -> Inference successful.")
    print(f"   -> Sample Prediction: ID={results[0][0]}, Sent='{results[0][1]}'")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n6. Formatting Submission...")

    df_sub = pd.DataFrame(results, columns=["id", "sentence"])
    df_sub.to_csv(
        Config.SUBMISSION_PATH, index=False, quoting=1
    )  # QUOTE_ALL=1 or QUOTE_NONNUMERIC

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Check content
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_check) == 3
    assert "id" in df_check.columns and "sentence" in df_check.columns

    print(f"   -> Submission saved to {Config.SUBMISSION_PATH}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
