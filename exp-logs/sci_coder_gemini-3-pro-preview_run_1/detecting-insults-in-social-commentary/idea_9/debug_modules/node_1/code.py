import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    logging as transformers_logging,
)

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, get_auc_score
from library.features import process_and_cache
from library.data import InsultDataset
from library.model import HybridDebertaModel
from library.awp import AWP
from library.engine import train_fn, eval_fn, inference_fn, get_optimizer_params


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    print("[1/7] Setting up configuration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast demo run
    Config.seed = 42
    Config.debug = True
    Config.debug_subset_size = 32  # Small subset for quick execution
    Config.svd_dim = 16  # Ensure svd_dim < subset_size for TruncatedSVD
    Config.epochs = 1
    Config.batch_size = 4
    Config.working_dir = "./working/demo_run/"
    Config.create_directories()

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # Initialize logger
    logger = get_logger(os.path.join(Config.working_dir, "demo.log"))
    logger.info("Configuration setup complete.")

    # ---------------------------------------------------------
    # 2. Feature Extraction (Structural Features)
    # ---------------------------------------------------------
    print("[2/7] Generating structural features...")

    # process_and_cache handles loading metadata, slicing for debug,
    # fitting SVD on train, and transforming all sets.
    # We set load_cached_data=False to force computation for the demo.
    train_feats, val_feats, test_feats = process_and_cache(load_cached_data=False)

    # Verify feature shapes
    expected_shape = (Config.debug_subset_size, Config.svd_dim)
    assert (
        train_feats.shape == expected_shape
    ), f"Train feats shape mismatch: {train_feats.shape}"
    assert (
        val_feats.shape == expected_shape
    ), f"Val feats shape mismatch: {val_feats.shape}"
    assert (
        test_feats.shape == expected_shape
    ), f"Test feats shape mismatch: {test_feats.shape}"
    logger.info("Structural features generated and verified.")

    # ---------------------------------------------------------
    # 3. Data Preparation
    # ---------------------------------------------------------
    print("[3/7] Preparing Datasets and DataLoaders...")

    # Load raw text corresponding to the debug subset
    # Note: process_and_cache uses the first N rows when debug=True
    train_df = pd.read_csv(Config.train_meta_path).iloc[: Config.debug_subset_size]
    val_df = pd.read_csv(Config.val_meta_path).iloc[: Config.debug_subset_size]
    test_df = pd.read_csv(Config.test_meta_path).iloc[: Config.debug_subset_size]

    # Instantiate Datasets
    train_dataset = InsultDataset(
        texts=train_df["Comment"].values,
        svd_features=train_feats,
        labels=train_df["Insult"].values,
    )

    val_dataset = InsultDataset(
        texts=val_df["Comment"].values,
        svd_features=val_feats,
        labels=val_df["Insult"].values,
    )

    test_dataset = InsultDataset(
        texts=test_df["Comment"].values, svd_features=test_feats, labels=None
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Train batches: {len(train_loader)}")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("[4/7] Initializing Model...")

    device = Config.device
    model = HybridDebertaModel(pretrained=True)
    model.to(device)

    logger.info(f"Model initialized on {device}.")

    # ---------------------------------------------------------
    # 5. Optimizer, Scheduler, and AWP Setup
    # ---------------------------------------------------------
    print("[5/7] Setting up Optimizer and AWP...")

    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Initialize Adversarial Weight Perturbation
    # We set start_epoch=0 to ensure it runs during this demo epoch
    awp = AWP(
        model=model,
        optimizer=optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=0,
    )

    # ---------------------------------------------------------
    # 6. Training & Validation Loop
    # ---------------------------------------------------------
    print("[6/7] Running Training and Validation...")

    # Run Training
    avg_train_loss = train_fn(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
    )

    print(f"  Training Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss resulted in NaN."

    # Run Validation
    avg_val_loss, val_preds, val_labels = eval_fn(val_loader, model, device)

    print(f"  Validation Loss: {avg_val_loss:.4f}")

    # Calculate AUC
    # Handle edge case where subset might contain only one class
    try:
        if len(np.unique(val_labels)) > 1:
            auc_score = get_auc_score(val_labels, val_preds)
            print(f"  Validation AUC: {auc_score:.4f}")
        else:
            print("  Validation AUC: Skipped (Single class in debug subset)")
    except Exception as e:
        print(f"  Validation AUC: Error ({e})")

    assert len(val_preds) == Config.debug_subset_size, "Prediction count mismatch."

    # ---------------------------------------------------------
    # 7. Inference
    # ---------------------------------------------------------
    print("[7/7] Running Inference on Test Set...")

    test_preds = inference_fn(test_loader, model, device)

    assert (
        len(test_preds) == Config.debug_subset_size
    ), "Test prediction count mismatch."
    assert (test_preds >= 0).all() and (
        test_preds <= 1
    ).all(), "Predictions outside [0,1] range."

    print("Inference successful. Sample predictions:")
    print(test_preds[:5])

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
