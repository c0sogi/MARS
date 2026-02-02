import os
import gc
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, get_logger, CacheManager, compute_qwk
from library.data import load_processed_data, get_folds, EssayDataset
from library.model_backbone import EssayBackbone, AWP
from library.engine import train_one_epoch, valid_one_epoch, extract_embeddings
from library.model_head import StackingTrainer, make_submission

logger = get_logger("Pipeline")


def get_optimizer_params(model, learning_rate, weight_decay):
    """
    Configures optimizer parameters, separating those that need weight decay
    from those that do not (bias, LayerNorm).
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    return optimizer_parameters


def run_cv():
    """
    Executes the 5-fold cross-validation pipeline.
    1. Trains Backbone (DeBERTa) on 5 folds (or loads cached OOF embeddings).
    2. Generates OOF embeddings.
    3. Trains Stacking Head (LightGBM) on OOF embeddings + Meta Features.
    """
    seed_everything(Config.seed)
    Config.create_directories()

    cache_manager = CacheManager(Config.cache_dir)

    # 1. Load Data
    logger.info("Initializing Tokenizer and Loading Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    train_dataset = load_processed_data(tokenizer, mode="train", load_cached_data=True)

    # Get Hidden Size for OOF Array Initialization
    backbone_config = AutoConfig.from_pretrained(Config.model_name)
    hidden_size = backbone_config.hidden_size

    # Initialize OOF arrays
    oof_embeddings = np.zeros((len(train_dataset), hidden_size), dtype=np.float32)
    # We also track indices to ensure correct alignment, though dataset order is preserved

    # 2. Iterate Folds
    folds = get_folds(train_dataset, n_folds=Config.n_folds, seed=Config.seed)

    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"\n{'='*20} Fold {fold+1} / {Config.n_folds} {'='*20}")

        # Define Cache Key for this fold's OOF embeddings
        fold_config = {"fold": fold, "model": Config.model_name, "debug": Config.debug}
        cached_oof = cache_manager.load(
            f"oof_embeddings_fold_{fold}", config_dict=fold_config
        )

        if cached_oof is not None:
            logger.info(f"Loaded cached OOF embeddings for Fold {fold+1}")
            oof_embeddings[val_idx] = cached_oof
            continue

        # --- Train Loop for this Fold ---

        # Create DataLoaders
        train_sub = Subset(train_dataset, train_idx)
        val_sub = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            train_sub,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = EssayBackbone(pretrained=True)
        model.to(Config.device)

        # Optimizer & Scheduler
        optimizer_params = get_optimizer_params(model, Config.lr, Config.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_params, lr=Config.lr, eps=Config.eps)

        num_train_steps = int(len(train_loader) * Config.epochs)
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # AWP
        awp = None
        if Config.use_awp:
            awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

        # Best Model Tracking
        best_loss = float("inf")
        best_model_path = os.path.join(
            Config.checkpoint_dir, f"backbone_fold_{fold}.pth"
        )

        for epoch in range(Config.epochs):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, Config.device, epoch, awp=awp
            )

            # Validation
            val_loss, val_preds = valid_one_epoch(model, val_loader, Config.device)

            # Metrics
            # Reconstruct targets for QWK calculation
            val_targets = [train_dataset.scores[i] for i in val_idx]
            qwk = compute_qwk(val_targets, val_preds)

            logger.info(
                f"Epoch {epoch+1} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val QWK: {qwk}"
            )

            # Save Best
            if val_loss < best_loss:
                best_loss = val_loss
                logger.info(f"New best model found! Saving to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)

        # --- Extract Embeddings for OOF ---
        logger.info("Loading best model for embedding extraction...")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.device))

        val_embeddings = extract_embeddings(model, val_loader, Config.device)

        # Save to Cache
        cache_manager.save(
            val_embeddings, f"oof_embeddings_fold_{fold}", config_dict=fold_config
        )

        # Update global array
        oof_embeddings[val_idx] = val_embeddings

        # Cleanup
        del model, optimizer, scheduler, awp, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Train Stacking Head
    logger.info("\n" + "=" * 20 + " Training Stacking Head " + "=" * 20)

    stacking_trainer = StackingTrainer()

    # We use the full OOF data to train the stacker
    # LightGBM will handle internal splitting for early stopping if we don't provide explicit val set
    stacking_trainer.fit(
        embeddings=oof_embeddings,
        meta_features=train_dataset.meta_features,
        labels=train_dataset.scores,
    )

    stacking_trainer.save(Config.output_dir, filename="lgbm_stacking.txt")

    # Calculate CV Score (Approximation based on OOF predictions of the stacker)
    # Note: To get a true CV score of the stacker, we would need to cross-validate the stacker itself.
    # Here we just predict on the training set (which is OOF for the backbone) to see fit.
    oof_preds = stacking_trainer.predict(oof_embeddings, train_dataset.meta_features)
    cv_qwk = compute_qwk(train_dataset.scores, oof_preds)
    logger.info(f"Final OOF QWK Score: {cv_qwk}")


def generate_submission():
    """
    Generates submission file for the test set.
    1. Loads test data.
    2. Generates Ensemble Embeddings (averaging backbone outputs from 5 folds).
    3. Predicts scores using the trained Stacking Head.
    4. Saves to CSV.
    """
    seed_everything(Config.seed)
    cache_manager = CacheManager(Config.cache_dir)

    logger.info("Loading Test Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_dataset = load_processed_data(tokenizer, mode="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 1. Generate Ensemble Embeddings
    test_config = {
        "mode": "test_ensemble",
        "model": Config.model_name,
        "debug": Config.debug,
    }
    avg_test_embeddings = cache_manager.load(
        "test_embeddings_ensemble", config_dict=test_config
    )

    if avg_test_embeddings is None:
        logger.info("Computing Test Embeddings (Ensemble)...")

        backbone_config = AutoConfig.from_pretrained(Config.model_name)
        cumulative_embeddings = np.zeros(
            (len(test_dataset), backbone_config.hidden_size), dtype=np.float32
        )

        for fold in range(Config.n_folds):
            logger.info(f"Inference with Fold {fold+1} model...")
            checkpoint_path = os.path.join(
                Config.checkpoint_dir, f"backbone_fold_{fold}.pth"
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}. Run run_cv() first."
                )

            model = EssayBackbone(pretrained=False)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=Config.device)
            )
            model.to(Config.device)

            embeddings = extract_embeddings(model, test_loader, Config.device)
            cumulative_embeddings += embeddings

            del model
            torch.cuda.empty_cache()
            gc.collect()

        avg_test_embeddings = cumulative_embeddings / Config.n_folds
        cache_manager.save(
            avg_test_embeddings, "test_embeddings_ensemble", config_dict=test_config
        )
    else:
        logger.info("Loaded cached Test Embeddings.")

    # 2. Predict with Stacking Head
    logger.info("Predicting with Stacking Head...")
    stacking_model_path = os.path.join(Config.output_dir, "lgbm_stacking.txt")

    stacking_trainer = StackingTrainer()
    stacking_trainer.load(stacking_model_path)

    predictions = stacking_trainer.predict(
        avg_test_embeddings, test_dataset.meta_features
    )

    # 3. Create Submission
    make_submission(
        ids=test_dataset.essay_ids,
        predictions=predictions,
        output_path=Config.submission_path,
    )
