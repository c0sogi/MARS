import os
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, logging

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_pearson
from library.data import get_dataloaders, preprocess_data, PearsonDataset, collate_fn
from library.model import CustomDeberta
from library.engine import train_fn, eval_fn
from library.awp import AWP
from library.stacking import prepare_stacking_features

# Suppress Transformers logging
logging.set_verbosity_error()


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # Override Config for Fast Baseline
    Config.epochs = 2
    Config.n_folds = 2
    print(f"Config overrides: Epochs={Config.epochs}, Folds={Config.n_folds}")

    # 2. Data Loading & Preprocessing
    # We load the full dataframes. Preprocess_data handles caching and feature engineering.
    df_train = preprocess_data("train", load_cached_data=True)
    df_val = preprocess_data("val", load_cached_data=True)
    df_test = preprocess_data("test", load_cached_data=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Level 1: DeBERTa Training (2-Fold CV)
    # We split df_train into 2 folds to generate OOF predictions for the entire train set.
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Arrays to store predictions
    oof_preds = np.zeros(len(df_train))
    val_preds_accum = np.zeros((Config.n_folds, len(df_val)))
    test_preds_accum = np.zeros((Config.n_folds, len(df_test)))

    # Target for stratification
    y_stratify = df_train["score"].values

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df_train, y_stratify)):
        print(f"\n=== Starting Fold {fold} ===")

        # Split Data
        train_fold_df = df_train.iloc[train_idx].reset_index(drop=True)
        valid_fold_df = df_train.iloc[valid_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = PearsonDataset(train_fold_df, tokenizer, is_train=True)
        valid_dataset = PearsonDataset(valid_fold_df, tokenizer, is_train=False)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomDeberta()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        num_train_steps = len(train_loader) * Config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_train_steps, eta_min=1e-6
        )

        # AWP
        awp = (
            AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)
            if Config.use_awp
            else None
        )

        # Training Loop
        for epoch in range(Config.epochs):
            avg_loss = train_fn(
                train_loader, model, optimizer, epoch, scheduler, device, awp
            )
            print(
                f"Fold {fold} | Epoch {epoch+1}/{Config.epochs} | Train Loss: {avg_loss:.4f}"
            )

        # Inference: OOF (Validation part of the fold)
        _, preds_fold_val, _ = eval_fn(valid_loader, model, device)
        oof_preds[valid_idx] = preds_fold_val

        # Inference: Global Validation Set
        # We need a loader for the full validation set
        global_val_dataset = PearsonDataset(df_val, tokenizer, is_train=False)
        global_val_loader = torch.utils.data.DataLoader(
            global_val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        _, preds_global_val, _ = eval_fn(global_val_loader, model, device)
        val_preds_accum[fold] = preds_global_val

        # Inference: Test Set
        test_dataset = PearsonDataset(df_test, tokenizer, is_train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        _, preds_test, _ = eval_fn(test_loader, model, device)
        test_preds_accum[fold] = preds_test

        # Cleanup
        del model, optimizer, scheduler, awp, train_loader, valid_loader
        torch.cuda.empty_cache()

    # Aggregate Predictions
    avg_val_preds = np.mean(val_preds_accum, axis=0)
    avg_test_preds = np.mean(test_preds_accum, axis=0)

    # 4. Level 2: Stacking
    print("\n=== Preparing Stacking Features ===")
    X_train, y_train, X_val, y_val, X_test, test_ids = prepare_stacking_features(
        oof_preds, avg_val_preds, avg_test_preds, load_cached_data=True
    )

    print("Training LightGBM Stacker...")
    lgb_params = Config.lgb_params.copy()
    es_rounds = lgb_params.pop("early_stopping_rounds", 100)

    stacker = lgb.LGBMRegressor(**lgb_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds),
        lgb.log_evaluation(period=0),  # Silent
    ]

    stacker.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    # 5. Validation & Failure Analysis
    val_final_preds = stacker.predict(X_val)
    val_final_preds = np.clip(val_final_preds, 0, 1)

    final_score = compute_pearson(y_val, val_final_preds)
    print(f"Final Validation Metric: {final_score}")

    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    abs_error = np.abs(y_val - val_final_preds)

    # Correlate with structural features present in df_val
    # Note: df_val aligns with y_val and val_final_preds
    features_to_analyze = [
        "normalized_levenshtein",
        "jaccard_similarity",
        "length_ratio",
    ]
    print("Correlation between Error Magnitude and Structural Features:")
    for feat in features_to_analyze:
        if feat in df_val.columns:
            feat_values = df_val[feat].values
            corr = np.corrcoef(abs_error, feat_values)[0, 1]
            print(f"  {feat}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.8654320295612139
    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        test_final_preds = stacker.predict(X_test)
        test_final_preds = np.clip(test_final_preds, 0, 1)

        submission = pd.DataFrame({"id": test_ids, "score": test_final_preds})
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
