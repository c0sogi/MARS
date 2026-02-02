import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging

# =========================================================================
# 0. Environment & Patching
# =========================================================================
# Patch tqdm to disable progress bars globally before importing libraries
import tqdm.auto


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.auto.tqdm = silent_tqdm

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.feature_engineering import StructuralFeatureGenerator
from library.dataset import InsultDataset
from library.model import HybridDebertaModel
from library.engine import train_fn, valid_fn, inference_fn

# Suppress warnings and transformer logs for clean output
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Initializing Insult Detection Pipeline Demo...")

    # =========================================================================
    # 1. Configuration Override for Speed and Demo
    # =========================================================================
    # We modify the Config class directly to run a fast, minimal version
    # This ensures the demo completes within seconds/minutes
    Config.debug = True  # Uses subset of data (100 train, 50 val/test)
    Config.seed = 42
    Config.epochs = 1
    Config.batch_size = 4  # Small batch size for demo
    Config.max_len = 32  # Short sequence length for speed
    Config.svd_output_dim = 16  # Low dimension for structural features
    Config.tfidf_word_ngram_range = (1, 1)  # Minimal TF-IDF
    Config.tfidf_char_ngram_range = (2, 2)  # Minimal TF-IDF
    Config.use_awp = False  # Disable Adversarial Weight Perturbation for speed
    Config.num_workers = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.seed)
    device = get_device()
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    print("Generating structural features (TF-IDF + SVD)...")
    feature_gen = StructuralFeatureGenerator()

    # Generate features in debug mode (computes on subset, returns numpy arrays)
    # load_cached_data=False forces re-computation to demonstrate the logic
    # debug=True causes the library to slice the first 100/50/50 rows
    train_feats, val_feats, test_feats = feature_gen.generate_features(
        load_cached_data=False, debug=True
    )

    # Verify feature shapes match Config settings and debug subset sizes
    assert train_feats.shape == (
        100,
        Config.svd_output_dim,
    ), f"Mismatch in train features: {train_feats.shape}"
    assert val_feats.shape == (
        50,
        Config.svd_output_dim,
    ), f"Mismatch in val features: {val_feats.shape}"
    assert test_feats.shape == (
        50,
        Config.svd_output_dim,
    ), f"Mismatch in test features: {test_feats.shape}"
    print("Feature generation successful.")

    # =========================================================================
    # 3. Dataset Preparation
    # =========================================================================
    print("Preparing datasets and dataloaders...")

    # Load the same data subsets used by feature_engineering.py in debug mode
    # to ensure alignment between text and structural features
    df_train = pd.read_csv(Config.train_path).head(100)
    df_val = pd.read_csv(Config.val_path).head(50)
    df_test = pd.read_csv(Config.test_path).head(50)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    train_dataset = InsultDataset(
        texts=df_train["Comment"].values,
        struct_features=train_feats,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        labels=df_train["Insult"].values,
    )

    val_dataset = InsultDataset(
        texts=df_val["Comment"].values,
        struct_features=val_feats,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        labels=df_val["Insult"].values,
    )

    test_dataset = InsultDataset(
        texts=df_test["Comment"].values,
        struct_features=test_feats,
        tokenizer=tokenizer,
        max_len=Config.max_len,
        labels=None,  # Test set has no labels
    )

    # Verify Dataset Output Structure
    sample = train_dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "struct_features" in sample
    assert "label" in sample
    assert sample["input_ids"].shape == (Config.max_len,)
    assert sample["struct_features"].shape == (Config.svd_output_dim,)

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    print("Datasets prepared successfully.")

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("Initializing HybridDebertaModel...")
    model = HybridDebertaModel(pretrained=True)
    model.to(device)

    # Verify Forward Pass with a dummy batch
    dummy_batch = next(iter(train_loader))
    with torch.no_grad():
        dummy_out = model(
            dummy_batch["input_ids"].to(device),
            dummy_batch["attention_mask"].to(device),
            dummy_batch["struct_features"].to(device),
        )
    # Output should be logits of shape (batch_size, 1)
    assert dummy_out.shape == (
        Config.batch_size,
        1,
    ), f"Model output shape mismatch: {dummy_out.shape}"
    print("Model initialized and verified.")

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    print("Starting training (1 epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Run training for 1 epoch
    train_loss = train_fn(
        dataloader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=1,
        scheduler=None,  # Skip scheduler for demo simplicity
        device=device,
        config=Config,
    )
    print(f"Training complete. Average Loss: {train_loss:.4f}")

    # =========================================================================
    # 6. Validation Loop
    # =========================================================================
    print("Starting validation...")
    val_loss, val_auc = valid_fn(val_loader, model, criterion, device)
    print(f"Validation complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # =========================================================================
    # 7. Inference
    # =========================================================================
    print("Starting inference...")
    predictions = inference_fn(test_loader, model, device)

    # Verify predictions
    assert len(predictions) == 50, "Prediction count mismatch"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range"

    print("Inference complete.")
    print(f"Sample predictions (first 5): {predictions[:5]}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
