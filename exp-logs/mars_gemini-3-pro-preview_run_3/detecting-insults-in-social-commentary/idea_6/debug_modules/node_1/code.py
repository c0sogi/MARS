import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, decode_text, get_device
from library.data import (
    load_and_preprocess_data,
    create_dataloaders,
    prepare_augmented_data,
    InsultDataset,
)
from library.model import DebertaV3Classifier
from library.engine import train_runner, inference_fn


def run_demo():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print("[Demo] Setting up configuration for fast execution...")

    # Override Config for speed and demonstration purposes
    Config.debug = True
    Config.debug_subset_size = 50  # Small subset for quick checks
    Config.num_epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.gradient_accumulation_steps = 1

    # Use a tiny model for demonstration to avoid large downloads and OOM on smaller environments
    # This model has a compatible architecture (embeddings, encoder.layer) for the DebertaV3Classifier wrapper
    demo_model_name = "prajjwal1/bert-tiny"

    # Ensure directories exist (Config.setup() is called on import, but good to confirm)
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.seed)
    print("[Demo] Seed set and configuration updated.")

    # ==========================================
    # 2. Test Utils
    # ==========================================
    print("\n[Demo] Testing library.utils...")

    # Test decode_text
    raw_text = "Hello\\nWorld"
    decoded = decode_text(raw_text)
    assert (
        decoded == "Hello\nWorld"
    ), f"decode_text failed: expected 'Hello\\nWorld', got {repr(decoded)}"
    print("  - decode_text: Passed")

    # Test get_device
    device = get_device()
    print(f"  - Device detected: {device}")

    # ==========================================
    # 3. Test Data Loading and Processing
    # ==========================================
    print("\n[Demo] Testing library.data...")

    # Load data (this handles caching and decoding)
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=False)

    # Verify data loading
    assert not train_df.empty, "Train DataFrame is empty"
    assert not val_df.empty, "Val DataFrame is empty"
    assert not test_df.empty, "Test DataFrame is empty"
    assert (
        "Comment" in train_df.columns and "Insult" in train_df.columns
    ), "Train DataFrame missing columns"
    print(
        f"  - Data Loaded: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}"
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(demo_model_name)

    # Create DataLoaders
    # Note: Config.debug = True will slice the dataframes inside create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        load_cached_data=True,
    )

    # Verify DataLoaders
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "labels" in batch, "Batch missing labels"
    assert (
        batch["input_ids"].shape[0] == Config.train_batch_size
    ), "Incorrect batch size"
    print("  - DataLoaders created and batch structure verified.")

    # Test Augmentation Logic (Pseudo-labeling)
    # Create dummy probabilities for the test set subset
    current_test_size = len(test_loader.dataset)
    dummy_probs = np.random.uniform(0, 1, current_test_size)

    # We need the sliced test_df that matches the debug loader
    debug_test_df = test_df.iloc[: Config.debug_subset_size].copy()
    debug_train_df = train_df.iloc[: Config.debug_subset_size].copy()

    augmented_df = prepare_augmented_data(debug_train_df, debug_test_df, dummy_probs)

    # Check if augmented df is larger than original train df
    assert len(augmented_df) >= len(
        debug_train_df
    ), "Augmented dataset should not be smaller than train set"
    assert "Insult" in augmented_df.columns, "Augmented dataset missing target column"
    print("  - Data Augmentation logic verified.")

    # ==========================================
    # 4. Test Model Initialization & Forward Pass
    # ==========================================
    print("\n[Demo] Testing library.model...")

    # Instantiate model with the tiny backbone
    model = DebertaV3Classifier(pretrained_model_name=demo_model_name)
    model.to(device)

    # Run a forward pass with the batch fetched earlier
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    # Verify output shape: [batch_size, 1]
    assert logits.shape == (
        Config.train_batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(Config.train_batch_size, 1)}, got {logits.shape}"
    print("  - Model instantiated and forward pass verified.")

    # ==========================================
    # 5. Test Training Engine
    # ==========================================
    print("\n[Demo] Testing library.engine (Training Loop)...")

    # Run training loop
    # This will run for 1 epoch on the debug subset (50 samples)
    trained_model, best_auc = train_runner(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        save_name="demo_model.bin",
    )

    assert isinstance(best_auc, float), "Best AUC should be a float"
    assert os.path.exists(
        os.path.join(Config.output_dir, "demo_model.bin")
    ), "Model file was not saved"
    print(f"  - Training loop completed. Best AUC: {best_auc}")

    # ==========================================
    # 6. Test Inference
    # ==========================================
    print("\n[Demo] Testing library.engine (Inference)...")

    predictions = inference_fn(test_loader, trained_model, device)

    # Verify predictions shape matches test dataset size
    assert len(predictions) == len(test_loader.dataset), "Prediction count mismatch"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]"
    print("  - Inference completed and output verified.")

    print("\n[Demo] All demonstrations and assertions passed successfully.")


if __name__ == "__main__":
    run_demo()
