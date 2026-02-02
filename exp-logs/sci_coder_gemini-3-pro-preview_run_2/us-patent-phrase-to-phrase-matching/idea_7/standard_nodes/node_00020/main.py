import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, logging

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_score
from library.dataset import PearsonDataset
from library.model import HybridModel
from library.engine import train_fn, valid_fn
from library.awp import AWP

# Suppress transformer warnings
logging.set_verbosity_error()


def run_training():
    # --- Setup ---
    seed_everything(Config.seed)
    device = Config.device

    # --- Runtime Optimization for "Fast Baseline" ---
    # We override Config to ensure execution within 2 hours while maintaining performance.
    # 3 folds * 3 epochs * 2 models = 18 training units.
    # Approx 4-5 mins per unit on A100 -> ~90 mins total.
    Config.epochs = 3
    Config.n_folds = 3

    print(f"Running on device: {device}")
    print(f"Configuration: {Config.epochs} epochs, {Config.n_folds} folds")

    # --- Data Loading ---
    # We use the tokenizer from the first model to initialize the dataset
    # Note: The dataset class handles tokenization internally based on the tokenizer passed.
    # We will re-initialize dataset for the second model to ensure correct tokenization.

    # Load Hold-out Validation Set (used for final scoring)
    # We load this once to ensure we have the targets for stratification if needed,
    # but mainly for the final evaluation.
    # Actually, for training, we need the TRAIN dataset.

    # We will iterate models first, then folds.

    trained_model_paths = []

    for model_cfg in Config.models:
        model_name = model_cfg["model_name"]
        short_name = model_cfg["short_name"]
        tokenizer_name = model_cfg["tokenizer_name"]

        print(f"\n{'='*40}")
        print(f"Processing Model: {short_name} ({model_name})")
        print(f"{'='*40}")

        # 1. Initialize Tokenizer & Dataset
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_dataset = PearsonDataset(
            mode="train",
            tokenizer=tokenizer,
            short_name=short_name,
            load_cached_data=True,
        )

        # 2. Stratified K-Fold Split
        # We split the training data into K folds
        skf = StratifiedKFold(
            n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
        )

        # We use the 'score' column (converted to class labels or bins) for stratification
        # The dataset has .labels which are class indices
        stratify_labels = train_dataset.labels

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(stratify_labels)), stratify_labels)
        ):
            print(f"\n--- Fold {fold+1}/{Config.n_folds} ---")

            # Create Subsets
            train_sub = Subset(train_dataset, train_idx)
            val_sub = Subset(train_dataset, val_idx)

            # DataLoaders
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
                batch_size=Config.batch_size * 2,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Model Initialization
            model = HybridModel(model_name=model_name, pretrained=True)
            model.to(device)

            # Optimizer & Scheduler
            optimizer_parameters = [
                {
                    "params": [p for n, p in model.backbone.named_parameters()],
                    "lr": Config.encoder_lr,
                },
                {
                    "params": [p for n, p in model.pooler.named_parameters()],
                    "lr": Config.decoder_lr,
                },
                {
                    "params": [p for n, p in model.fc.named_parameters()],
                    "lr": Config.decoder_lr,
                },
            ]

            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=Config.encoder_lr,
                weight_decay=Config.weight_decay,
            )

            num_train_steps = int(
                len(train_loader) * Config.epochs / Config.gradient_accumulation_steps
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
                num_training_steps=num_train_steps,
            )

            # Loss Function
            criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

            # AWP
            awp = None
            if Config.use_awp:
                awp = AWP(
                    model,
                    optimizer,
                    adv_lr=Config.awp_lr,
                    adv_eps=Config.awp_eps,
                    start_epoch=Config.awp_start_epoch,
                )

            # Training Loop
            best_score = -1.0
            best_model_path = os.path.join(
                Config.output_dir, f"{short_name}_fold_{fold}.bin"
            )

            for epoch in range(1, Config.epochs + 1):
                # Train
                avg_loss = train_fn(
                    fold,
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    device,
                    awp,
                )

                # Validate
                val_loss, val_score, _ = valid_fn(val_loader, model, criterion, device)

                print(
                    f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}"
                )

                # Save Best
                if val_score > best_score:
                    best_score = val_score
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best score! Model saved to {best_model_path}")

            trained_model_paths.append(
                {
                    "path": best_model_path,
                    "short_name": short_name,
                    "model_name": model_name,
                    "tokenizer_name": tokenizer_name,
                }
            )

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                train_sub,
                val_sub,
            )
            torch.cuda.empty_cache()
            gc.collect()

        # Cleanup dataset/tokenizer for this model
        del train_dataset, tokenizer
        gc.collect()

    return trained_model_paths


def run_evaluation(trained_models):
    print(f"\n{'='*40}")
    print("Running Final Evaluation on Hold-out Validation Set")
    print(f"{'='*40}")

    device = Config.device

    # We need to predict on the hold-out validation set using ALL models
    # and average the predictions.

    # 1. Load Hold-out Validation Data
    # We need a tokenizer to load the dataset. We'll load it per model group.

    # Group models by type to reuse dataset/tokenizer
    model_groups = {}
    for m in trained_models:
        key = (m["short_name"], m["model_name"], m["tokenizer_name"])
        if key not in model_groups:
            model_groups[key] = []
        model_groups[key].append(m["path"])

    ensemble_preds = None
    ground_truth = None
    val_df = None

    for (short_name, model_name, tokenizer_name), paths in model_groups.items():
        print(f"Evaluating ensemble part: {short_name}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        val_dataset = PearsonDataset(
            mode="val",
            tokenizer=tokenizer,
            short_name=short_name,
            load_cached_data=True,
        )

        if val_df is None:
            val_df = val_dataset.df
            # Store ground truth once
            ground_truth = val_dataset.raw_scores

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Predict with each fold model
        for model_path in paths:
            model = HybridModel(model_name=model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            preds = []
            score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    structural_features = batch["structural_features"].to(device)

                    logits = model(input_ids, attention_mask, structural_features)
                    probs = torch.softmax(logits, dim=1)
                    pred_scores = torch.sum(probs * score_values, dim=1)
                    preds.append(pred_scores.cpu().numpy())

            fold_preds = np.concatenate(preds)

            if ensemble_preds is None:
                ensemble_preds = fold_preds
            else:
                ensemble_preds += fold_preds

            del model
            torch.cuda.empty_cache()

        del val_dataset, tokenizer, val_loader
        gc.collect()

    # Average predictions
    num_models = len(trained_models)
    avg_preds = ensemble_preds / num_models

    # Compute Metric
    final_score = compute_score(ground_truth, avg_preds)
    print(f"Final Validation Metric: {final_score}")

    return final_score, avg_preds, val_df


def failure_analysis(val_df, y_true, y_pred):
    print(f"\n{'='*40}")
    print("Failure Analysis")
    print(f"{'='*40}")

    # Calculate Absolute Error
    errors = np.abs(y_true - y_pred)

    # Get structural features from dataframe
    # We need to re-compute or extract them.
    # Since we have the dataframe, we can quickly compute the ones we want to correlate.
    # Note: 'val_df' is the raw dataframe.

    from library.features import get_norm_levenshtein, get_jaccard

    # Compute features
    print("Computing features for analysis...")
    anchors = val_df["anchor"].astype(str).tolist()
    targets = val_df["target"].astype(str).tolist()

    feat_lev = np.array([get_norm_levenshtein(a, t) for a, t in zip(anchors, targets)])
    feat_jac = np.array([get_jaccard(a, t) for a, t in zip(anchors, targets)])
    feat_len_diff = np.abs(
        val_df["anchor"].astype(str).str.len() - val_df["target"].astype(str).str.len()
    ).values

    # Correlations
    correlations = {
        "Error vs Norm Levenshtein": compute_score(errors, feat_lev),
        "Error vs Jaccard": compute_score(errors, feat_jac),
        "Error vs Length Diff": compute_score(errors, feat_len_diff),
    }

    print("Correlation between Error Magnitude and Features:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.4f}")


def generate_submission(trained_models):
    print(f"\n{'='*40}")
    print("Generating Submission")
    print(f"{'='*40}")

    device = Config.device

    # Group models
    model_groups = {}
    for m in trained_models:
        key = (m["short_name"], m["model_name"], m["tokenizer_name"])
        if key not in model_groups:
            model_groups[key] = []
        model_groups[key].append(m["path"])

    ensemble_preds = None
    test_ids = None

    for (short_name, model_name, tokenizer_name), paths in model_groups.items():
        print(f"Predicting with: {short_name}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        test_dataset = PearsonDataset(
            mode="test",
            tokenizer=tokenizer,
            short_name=short_name,
            load_cached_data=True,
        )

        if test_ids is None:
            test_ids = test_dataset.df["id"].values

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        for model_path in paths:
            model = HybridModel(model_name=model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            preds = []
            score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    structural_features = batch["structural_features"].to(device)

                    logits = model(input_ids, attention_mask, structural_features)
                    probs = torch.softmax(logits, dim=1)
                    pred_scores = torch.sum(probs * score_values, dim=1)
                    preds.append(pred_scores.cpu().numpy())

            fold_preds = np.concatenate(preds)

            if ensemble_preds is None:
                ensemble_preds = fold_preds
            else:
                ensemble_preds += fold_preds

            del model
            torch.cuda.empty_cache()

        del test_dataset, tokenizer, test_loader
        gc.collect()

    # Average
    num_models = len(trained_models)
    avg_preds = ensemble_preds / num_models

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "score": avg_preds})

    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def main():
    # 1. Train
    trained_models = run_training()

    # 2. Evaluate
    val_score, val_preds, val_df = run_evaluation(trained_models)

    # 3. Failure Analysis
    # Get ground truth from val_df (loaded from metadata/val.csv)
    y_true = val_df["score"].values
    failure_analysis(val_df, y_true, val_preds)

    # 4. Submission
    threshold = 0.8654320295612139
    if val_score > threshold:
        print(
            f"Validation score ({val_score:.4f}) > Threshold ({threshold:.4f}). Generating submission..."
        )
        generate_submission(trained_models)
    else:
        print(
            f"Validation score ({val_score:.4f}) <= Threshold ({threshold:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
