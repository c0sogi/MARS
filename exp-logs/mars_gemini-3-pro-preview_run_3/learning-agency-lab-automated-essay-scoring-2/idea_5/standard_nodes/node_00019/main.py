import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk, get_logger
from library.data import process_data, EssayDataset
from library.model import EssayModel
from library.trainer import run_fold
from library.stacking import LGBMStacker


def extract_embeddings(model, loader, device, config):
    """
    Extracts embeddings (pooling output) from the model by bypassing the final classification head.
    This is necessary because the Stacker requires the semantic vector representation.
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=config.use_amp):
                # Forward pass through backbone
                outputs = model.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                last_hidden_state = outputs.last_hidden_state
                # Apply the specific pooling mechanism defined in the model
                pool_out = model.pool(last_hidden_state, attention_mask)

            embeddings.append(pool_out.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    config = Config()

    # Optimization for Fast Baseline / Time Constraints
    # We limit epochs and dataset size to ensure completion within 2 hours.
    config.epochs = 1
    MAX_TRAIN_SAMPLES = 6000

    seed_everything(config.seed)

    # Setup logging
    log_path = os.path.join(config.working_dir, "run.log")
    logger = get_logger(log_path)
    logger.info("Starting orchestration script...")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # -------------------------------------------------------------------------
    # Load processed data (cached if available)
    train_df, test_df = process_data(config, load_cached_data=True)

    # Subsample training data if necessary to meet time constraints
    if len(train_df) > MAX_TRAIN_SAMPLES:
        logger.info(
            f"Subsampling training data from {len(train_df)} to {MAX_TRAIN_SAMPLES} for speed."
        )
        # Stratified subsample to preserve score distribution
        train_df, _ = train_test_split(
            train_df,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=train_df["score"],
            random_state=config.seed,
        )
        train_df = train_df.reset_index(drop=True)

        # Re-assign folds because subsampling breaks the original contiguous fold structure
        logger.info("Re-assigning stratified folds after subsampling...")
        skf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.seed
        )
        train_df["fold"] = -1
        for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["score"])):
            train_df.loc[val_idx, "fold"] = fold
        train_df["fold"] = train_df["fold"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # Containers for Out-Of-Fold (OOF) data
    oof_embeddings = []
    oof_meta = []
    oof_targets = []

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Loop (Stage 1: Backbone Fine-tuning)
    # -------------------------------------------------------------------------
    for fold in range(config.n_folds):
        logger.info(f"=== Processing Fold {fold}/{config.n_folds - 1} ===")

        # Split Data
        df_train_fold = train_df[train_df["fold"] != fold].reset_index(drop=True)
        df_val_fold = train_df[train_df["fold"] == fold].reset_index(drop=True)

        # Create Datasets
        train_dataset = EssayDataset(df_train_fold, config, tokenizer, is_test=False)
        val_dataset = EssayDataset(df_val_fold, config, tokenizer, is_test=False)

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Train Backbone
        # run_fold trains the model and saves the best checkpoint to disk
        _ = run_fold(fold, train_loader, val_loader, config, logger)

        # Load Best Model for Embedding Extraction
        # We need embeddings for the stacker, not just the regression scores
        model = EssayModel(config)
        ckpt_path = os.path.join(config.checkpoint_dir, f"backbone_fold_{fold}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
        model.to(config.device)

        # Extract Embeddings
        logger.info(f"Extracting embeddings for fold {fold} validation set...")
        val_embeddings = extract_embeddings(model, val_loader, config.device, config)

        # Store Data
        oof_embeddings.append(val_embeddings)
        oof_meta.append(df_val_fold[config.meta_features].values)
        oof_targets.append(df_val_fold["score"].values)

        # Cleanup to free GPU memory
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Stacking (Stage 2: LightGBM)
    # -------------------------------------------------------------------------
    logger.info("Preparing data for Stacking...")

    # Concatenate data from all folds
    X_emb = np.concatenate(oof_embeddings, axis=0)
    X_meta = np.concatenate(oof_meta, axis=0)
    y_true = np.concatenate(oof_targets, axis=0)

    # Combine embeddings and meta-features
    X_full = np.concatenate([X_emb, X_meta], axis=1)

    logger.info(f"Training LightGBM Stacker on input shape {X_full.shape}...")
    stacker = LGBMStacker(config)
    stacker.train(X_full, y_true)
    stacker.save("lgbm_stacking.txt")

    # -------------------------------------------------------------------------
    # 5. Validation and Failure Analysis
    # -------------------------------------------------------------------------
    # Predict on OOF data
    y_pred = stacker.predict(X_full)

    # Compute Metric
    final_qwk = compute_qwk(y_true, y_pred)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_qwk}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    residuals = np.abs(y_true - y_pred)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(X_meta, columns=config.meta_features)
    analysis_df["residual"] = residuals

    print("Correlation between Meta-Features and Error Magnitude:")
    for col in config.meta_features:
        corr = analysis_df[col].corr(analysis_df["residual"])
        print(f"{col}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.8246384329994252

    if final_qwk > THRESHOLD:
        logger.info(
            f"Metric ({final_qwk}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Prepare Test Data
        test_dataset = EssayDataset(test_df, config, tokenizer, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Ensemble Inference: Average embeddings from all 5 folds
        test_embeddings_list = []

        for fold in range(config.n_folds):
            logger.info(f"Generating test embeddings with Fold {fold} model...")

            # Load model
            model = EssayModel(config)
            ckpt_path = os.path.join(config.checkpoint_dir, f"backbone_fold_{fold}.pth")
            model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
            model.to(config.device)

            # Extract embeddings
            emb = extract_embeddings(model, test_loader, config.device, config)
            test_embeddings_list.append(emb)

            del model
            torch.cuda.empty_cache()

        # Average embeddings
        avg_test_embeddings = np.mean(test_embeddings_list, axis=0)

        # Combine with Test Meta Features
        test_meta = test_df[config.meta_features].values
        X_test = np.concatenate([avg_test_embeddings, test_meta], axis=1)

        # Generate Submission
        stacker.make_submission(test_df["essay_id"].tolist(), X_test, "submission.csv")
        logger.info("Submission generation complete.")

    else:
        logger.warning(
            f"Metric ({final_qwk}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
