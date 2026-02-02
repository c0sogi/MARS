import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_data
from library.model import WDPIRVModel, train_model, predict


def run_training_pipeline(
    debug=False, load_cached_data=True, epochs=None, batch_size=None
):
    """
    Orchestrates the training and inference pipeline.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
        load_cached_data (bool): If True, attempts to load features from cache.
        epochs (int, optional): Override default number of epochs.
        batch_size (int, optional): Override default batch size.
    """
    # 1. Setup & Configuration
    seed_everything(Config.SEED)

    if epochs is not None:
        Config.EPOCHS = epochs
    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    print(
        f"Starting pipeline (Debug={debug}, Cache={load_cached_data}, "
        f"Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE})"
    )

    # 2. Data Loading
    # get_data handles caching and debug sampling internally
    train_dataset, val_dataset, test_dataset, feature_dim = get_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    print(f"Initializing WDPIRVModel with input dimension: {feature_dim}")
    model = WDPIRVModel(input_dim=feature_dim)

    # 5. Training
    # train_model handles the loop, FocalLoss, validation, and early stopping
    # It saves the best model to Config.MODEL_PATH and returns the best threshold
    best_threshold = train_model(model, train_loader, val_loader, Config.DEVICE)
    print(f"Optimal Threshold found: {best_threshold}")

    # 6. Inference
    print("Generating predictions on test set...")
    # Load best model weights to ensure we use the best checkpoint
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # Get raw probabilities
    test_probs = predict(model, test_loader, Config.DEVICE)

    # Apply threshold
    test_preds = (test_probs > best_threshold).astype(int)

    # 7. Submission Generation
    print("Creating submission file...")
    # Load test metadata to get the correct contact_ids in order
    # The test dataset features were generated from this file, so order is preserved
    df_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    if debug:
        # If debugging, get_data sampled the input, so we must sample metadata to match
        # However, get_data logic for test set sampling in library/data_processing.py
        # uses df_test_meta.sample(2000), but we don't have the exact indices here easily
        # unless we reload with the same seed.
        # Since get_data sets seed_everything(Config.SEED), re-sampling here works.
        df_test_meta = df_test_meta.sample(2000, random_state=Config.SEED)

    # Ensure lengths match
    if len(df_test_meta) != len(test_preds):
        print(
            f"Warning: Metadata length ({len(df_test_meta)}) does not match "
            f"predictions length ({len(test_preds)}). This may happen in debug mode "
            "if sampling logic differs."
        )
        # In a real run, this should be an error. In debug, we might truncate.
        min_len = min(len(df_test_meta), len(test_preds))
        df_test_meta = df_test_meta.iloc[:min_len]
        test_preds = test_preds[:min_len]

    submission = pd.DataFrame(
        {"contact_id": df_test_meta["contact_id"], "contact": test_preds}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
