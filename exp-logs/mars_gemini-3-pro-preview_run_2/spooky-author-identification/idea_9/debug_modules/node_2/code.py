import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library import data_factory
from library import neural_model
from library import optimization
from library import engine
from library import classical_engine
from library.utils import seed_everything


def main():
    print("=== Starting Author Identification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo speed...")

    # Enable debug mode to use small data subsets (100 train, 50 val, 50 test)
    Config.debug = True

    # Reduce training complexity
    Config.epochs = 1
    Config.n_folds = 2  # Minimum folds for CV
    Config.svd_n_components = 5  # Small SVD dimension
    Config.tfidf_min_df = 1  # Ensure vocab exists even with small data

    # Enable AWP from epoch 0 to verify the code path
    Config.awp_start_epoch = 0

    # Reduce batch sizes for speed/memory safety
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.gradient_accumulation_steps = 1

    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.seed)

    print(f"Debug Mode: {Config.debug}")
    print(f"Output Directory: {Config.output_dir}")
    print(f"Device: {Config.device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data...")
    train_df, val_df, test_df = data_factory.load_data()

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Validations
    assert not train_df.empty, "Training dataframe is empty."
    assert not val_df.empty, "Validation dataframe is empty."
    assert not test_df.empty, "Test dataframe is empty."
    assert "text" in train_df.columns and "author" in train_df.columns

    # -------------------------------------------------------------------------
    # 3. Classical Features Generation
    # -------------------------------------------------------------------------
    print("\n[3] Generating Classical Features (TF-IDF + SVD)...")

    # Force computation from scratch by setting load_cached_data=False
    # This verifies the feature engineering pipeline logic.
    features_dict = data_factory.get_classical_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validations
    required_keys = [
        "train_sparse",
        "val_sparse",
        "test_sparse",
        "train_dense",
        "val_dense",
        "test_dense",
    ]
    for key in required_keys:
        assert key in features_dict, f"Missing key in features dict: {key}"

    # Check dimensions
    assert features_dict["train_dense"].shape[1] == Config.svd_n_components
    assert features_dict["train_dense"].shape[0] == len(train_df)

    # -------------------------------------------------------------------------
    # 4. Classical Models (LR, NB, XGB)
    # -------------------------------------------------------------------------
    print("\n[4] Running Classical Models CV...")

    # Run CV and get OOF/Test predictions
    # We pass load_cached_preds=False to ensure models are actually trained
    oof_preds, test_preds = classical_engine.run_classical_cv(
        features_dict, train_df, val_df, load_cached_preds=False
    )

    # Validations
    model_keys = ["lr", "nb", "xgb"]
    total_labeled = len(train_df) + len(val_df)

    for m in model_keys:
        assert m in oof_preds and m in test_preds
        assert oof_preds[m].shape == (total_labeled, Config.num_classes)
        assert test_preds[m].shape == (len(test_df), Config.num_classes)

    print("Classical models executed successfully.")

    # -------------------------------------------------------------------------
    # 5. Neural Model Setup
    # -------------------------------------------------------------------------
    print("\n[5] Setting up Neural Model Pipeline...")

    # Initialize Tokenizer
    # Note: This requires internet access or cached model.
    # Config.model_name is 'microsoft/deberta-v3-large'.
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    # Using a short max_len to speed up the demo
    demo_max_len = 32
    train_dataset = data_factory.TextDataset(train_df, tokenizer, max_len=demo_max_len)
    val_dataset = data_factory.TextDataset(val_df, tokenizer, max_len=demo_max_len)
    test_dataset = data_factory.TextDataset(
        test_df, tokenizer, max_len=demo_max_len, is_test=True
    )

    # Create DataLoaders
    # num_workers=0 to avoid multiprocessing overhead in this short demo
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.valid_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.valid_batch_size, shuffle=False, num_workers=0
    )

    # Initialize Model
    print(f"Initializing {Config.model_name}...")
    model = neural_model.CustomDeberta(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer_grouped_parameters = optimization.get_optimizer_grouped_parameters(model)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # -------------------------------------------------------------------------
    # 6. Neural Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Testing Training Loop (1 Epoch with AWP)...")

    # Train for one epoch
    train_loss = engine.train_fn(
        train_loader, model, optimizer, scheduler, epoch=0, device=Config.device
    )

    assert isinstance(train_loss, float), "Training loss must be a float"
    assert not np.isnan(train_loss), "Training loss is NaN"
    print(f"Train Loss: {train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 7. Neural Evaluation
    # -------------------------------------------------------------------------
    print("\n[7] Testing Evaluation Loop...")

    val_loss, val_probs = engine.eval_fn(val_loader, model, device=Config.device)

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert val_probs.shape == (len(val_df), Config.num_classes)

    # Check probability properties (rows sum to ~1)
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # -------------------------------------------------------------------------
    # 8. Neural Inference
    # -------------------------------------------------------------------------
    print("\n[8] Testing Inference Loop...")

    test_probs = engine.inference_fn(test_loader, model, device=Config.device)

    assert test_probs.shape == (len(test_df), Config.num_classes)
    print(f"Inference shape: {test_probs.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
