import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as transformers_logging

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_data, EssayDataset
from library.model import EssayModel
from library.trainer import run_fold
from library.stacking import LGBMStacker

# Suppress Transformers logging
transformers_logging.set_verbosity_error()


def extract_features(model, loader, device, config):
    """
    Helper function to extract embeddings (pooled output) and scalar predictions
    from the backbone model for stacking.
    """
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass through backbone
            outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # pooling
            pooled_output = model.pool(last_hidden_state, attention_mask)

            embeddings_list.append(pooled_output.cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Initializing Configuration...")
    config = Config()

    # Modify config for fast demo execution
    config.debug = True
    config.debug_subset_size = 50  # Small subset for speed
    config.epochs = 1
    config.n_folds = 2  # Minimum folds to trigger splitting logic
    config.train_batch_size = 2
    config.valid_batch_size = 2
    config.working_dir = "./working/demo_run"
    config.output_dir = os.path.join(config.working_dir, "output")
    config.checkpoint_dir = os.path.join(config.working_dir, "checkpoints")
    config.cache_dir = os.path.join(config.working_dir, "cache")
    config.submission_dir = os.path.join(config.working_dir, "submission")

    # Update LGBM params for speed
    config.lgbm_params["n_estimators"] = 10
    config.lgbm_params["verbose"] = -1

    # Re-create directories since we changed working_dir
    for d in [
        config.working_dir,
        config.output_dir,
        config.checkpoint_dir,
        config.cache_dir,
        config.submission_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    seed_everything(config.seed)
    logger = get_logger(os.path.join(config.output_dir, "demo.log"))

    # -------------------------------------------------------------------------
    # 2. Data Processing
    # -------------------------------------------------------------------------
    logger.info(">>> Processing Data...")
    # Force processing to ensure we use the debug subset
    train_df, test_df = process_data(config, load_cached_data=False)

    # Verification
    assert not train_df.empty, "Train DataFrame is empty"
    assert not test_df.empty, "Test DataFrame is empty"
    assert "fold" in train_df.columns, "Fold column missing in Train DF"
    for feature in config.meta_features:
        assert feature in train_df.columns, f"Meta-feature {feature} missing"

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Backbone Model Training (Fold 0)
    # -------------------------------------------------------------------------
    logger.info(">>> Preparing Fold 0...")
    fold = 0
    df_train_fold = train_df[train_df["fold"] != fold].reset_index(drop=True)
    df_valid_fold = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Ensure we have data for the fold
    if df_valid_fold.empty:
        # Fallback for extremely small debug sizes where stratification might fail
        logger.warning("Fold 0 empty due to small debug size. Splitting manually.")
        df_train_fold = train_df.iloc[:30]
        df_valid_fold = train_df.iloc[30:]

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    train_dataset = EssayDataset(df_train_fold, config, tokenizer, is_test=False)
    valid_dataset = EssayDataset(df_valid_fold, config, tokenizer, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers for simple debug
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(">>> Running Backbone Training (DeBERTa)...")
    # This runs training and saves the best model to checkpoints/
    val_preds = run_fold(fold, train_loader, valid_loader, config, logger)

    # Verification
    checkpoint_path = os.path.join(config.checkpoint_dir, f"backbone_fold_{fold}.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    assert len(val_preds) == len(df_valid_fold), "Prediction count mismatch."

    # -------------------------------------------------------------------------
    # 4. Feature Extraction for Stacking
    # -------------------------------------------------------------------------
    logger.info(">>> Extracting Embeddings for Stacking...")

    # Load the trained model
    model = EssayModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.to(config.device)

    # Extract embeddings for validation set
    # In a real pipeline, this would be OOF embeddings from all folds.
    # Here we use the validation set of Fold 0.
    val_embeddings = extract_features(model, valid_loader, config.device, config)

    # Prepare Meta-Features
    val_meta = df_valid_fold[config.meta_features].values

    # Combine Embeddings + Meta Features
    # Embeddings shape: (N, hidden_size), Meta shape: (N, num_meta)
    X_val = np.concatenate([val_embeddings, val_meta], axis=1)
    y_val = df_valid_fold["score"].values

    logger.info(f"Stacking Input Shape: {X_val.shape}")

    # -------------------------------------------------------------------------
    # 5. Stacking Model (LightGBM)
    # -------------------------------------------------------------------------
    logger.info(">>> Training LightGBM Stacker...")
    stacker = LGBMStacker(config)
    stacker.train(X_val, y_val)

    # Verification
    stacker_preds = stacker.predict(X_val)
    assert stacker_preds.shape == y_val.shape
    assert np.all(
        (stacker_preds >= 1) & (stacker_preds <= 6)
    ), "Predictions out of range"

    stacker.save("demo_lgbm.txt")
    assert os.path.exists(os.path.join(config.output_dir, "demo_lgbm.txt"))

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    logger.info(">>> Generating Submission...")

    test_dataset = EssayDataset(test_df, config, tokenizer, is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=0
    )

    # 1. Extract Test Embeddings
    test_embeddings = extract_features(model, test_loader, config.device, config)

    # 2. Get Test Meta-Features
    test_meta = test_df[config.meta_features].values

    # 3. Combine
    X_test = np.concatenate([test_embeddings, test_meta], axis=1)

    # 4. Predict with Stacker
    submission_df = stacker.make_submission(test_df["essay_id"].tolist(), X_test)

    # Verification
    assert not submission_df.empty
    assert list(submission_df.columns) == ["essay_id", "score"]
    assert len(submission_df) == len(test_df)

    print("\n>>> Demo Completed Successfully!")
    print(submission_df.head())


if __name__ == "__main__":
    run_demo()
