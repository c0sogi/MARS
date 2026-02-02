import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, logging as transformers_logging

# Import library components
from library.configuration import Config
from library.utilities import set_seed, get_optimizer_params
from library.features import SVDFeatureExtractor
from library.data import create_loaders
from library.architecture import HybridModel
from library.trainer import Trainer

# Suppress verbose warnings for cleaner output
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Starting Demonstration of Insult Detection Pipeline...")

    # =========================================================================
    # 1. Configuration Setup (Optimized for Speed)
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and debugging
    Config.debug = True
    Config.debug_sample_size = 50  # Use only 50 samples
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.num_workers = 0  # Avoid multiprocessing overhead for small data

    # Use a tiny model for demonstration speed
    Config.model_a_name = "prajjwal1/bert-tiny"

    # Adjust SVD settings to be compatible with small sample size
    Config.svd_components = 10
    Config.svd_embedding_size = 10

    # Set device
    device = Config.device
    print(f"    Device: {device}")
    print(f"    Model: {Config.model_a_name}")
    print(f"    Debug Mode: {Config.debug}")

    # Set seed for reproducibility
    set_seed(Config.seed)

    # =========================================================================
    # 2. Feature Extraction
    # =========================================================================
    print("\n[2] Extracting SVD Features...")

    # Initialize Extractor
    feature_extractor = SVDFeatureExtractor()

    # Process features (force re-computation by setting load_cached_data=False)
    # In a real run, we would likely use True, but here we want to demo the generation logic.
    train_svd, val_svd, test_svd = feature_extractor.process(load_cached_data=False)

    # Verify shapes
    print(f"    Train SVD Shape: {train_svd.shape}")
    print(f"    Val SVD Shape:   {val_svd.shape}")

    assert train_svd.shape == (
        Config.debug_sample_size,
        Config.svd_components,
    ), f"Expected train_svd shape ({Config.debug_sample_size}, {Config.svd_components}), got {train_svd.shape}"
    assert val_svd.shape == (
        Config.debug_sample_size,
        Config.svd_components,
    ), f"Expected val_svd shape ({Config.debug_sample_size}, {Config.svd_components}), got {val_svd.shape}"

    print("    Feature extraction verified.")

    # =========================================================================
    # 3. Data Loading & Tokenization
    # =========================================================================
    print("\n[3] Preparing DataLoaders...")

    # Load DataFrames (Manually slicing to match the debug logic in features.py)
    train_df = pd.read_csv(Config.train_path).iloc[: Config.debug_sample_size]
    val_df = pd.read_csv(Config.val_path).iloc[: Config.debug_sample_size]

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_a_name)

    # Create DataLoaders
    train_loader = create_loaders(
        df=train_df,
        svd_features=train_svd,
        tokenizer=tokenizer,
        batch_size=Config.train_batch_size,
        labels=train_df["Insult"].values,
        shuffle=True,
    )

    val_loader = create_loaders(
        df=val_df,
        svd_features=val_svd,
        tokenizer=tokenizer,
        batch_size=Config.valid_batch_size,
        labels=val_df["Insult"].values,
        shuffle=False,
    )

    # Verify DataLoader
    batch = next(iter(train_loader))
    print(f"    Batch Keys: {batch.keys()}")
    print(f"    Input IDs Shape: {batch['input_ids'].shape}")

    assert "input_ids" in batch
    assert "svd_features" in batch
    assert "target" in batch
    assert batch["input_ids"].shape[0] == Config.train_batch_size

    print("    Data loading verified.")

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("\n[4] Initializing Hybrid Model...")

    model = HybridModel(model_name=Config.model_a_name, pretrained=True)
    model.to(device)

    # Verify Model Architecture
    print(f"    Backbone Hidden Size: {model.hidden_size}")
    print(f"    Fused Dimension: {model.fused_dim}")

    # Check if fused dimension calculation is correct (Hidden + SVD)
    # bert-tiny hidden size is 128
    expected_fused = 128 + Config.svd_embedding_size
    assert (
        model.fused_dim == expected_fused
    ), f"Expected fused dim {expected_fused}, got {model.fused_dim}"

    # Test Forward Pass
    with torch.no_grad():
        dummy_input_ids = batch["input_ids"].to(device)
        dummy_mask = batch["attention_mask"].to(device)
        dummy_svd = batch["svd_features"].to(device)

        output = model(dummy_input_ids, dummy_mask, dummy_svd)

    print(f"    Forward Pass Output Shape: {output.shape}")
    assert output.shape == (Config.train_batch_size, 1), "Model output shape mismatch."

    print("    Model initialization verified.")

    # =========================================================================
    # 5. Training Loop Execution
    # =========================================================================
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Setup Optimizer
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=Config.lr_backbone,
        decoder_lr=Config.lr_head,
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    # Setup Scheduler (Linear for demo)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=10
    )

    # Initialize Trainer
    trainer = Trainer(
        config=Config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Run Training
    best_auc = trainer.fit(epochs=Config.epochs)

    print(f"    Training complete. Best Validation AUC: {best_auc:.4f}")
    assert isinstance(best_auc, float), "Trainer did not return a float metric."
    assert 0.0 <= best_auc <= 1.0, "AUC score out of bounds."

    # =========================================================================
    # 6. Inference
    # =========================================================================
    print("\n[6] Running Inference on Validation Set...")

    preds = trainer.predict(val_loader, return_logits=False)

    print(f"    Predictions Shape: {preds.shape}")
    print(f"    Sample Predictions: {preds[:5].flatten()}")

    assert len(preds) == Config.debug_sample_size, "Prediction count mismatch."
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]."

    print("    Inference verified.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
