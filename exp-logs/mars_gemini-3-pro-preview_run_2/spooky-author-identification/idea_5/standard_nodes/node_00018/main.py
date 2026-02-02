import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from transformers import AdamW, get_linear_schedule_with_warmup

# Import from library
from library.config import Config, seed_everything
from library.utils import save_numpy, load_numpy, save_submission, clip_probabilities
from library.data_loader import load_data, get_classical_features, TextDataset
from library.models_classical import run_classical_models
from library.models_transformer import CustomDeberta, CustomRoberta
from library.engine import run_training, predict_fn
from library.ensemble import run_ensemble


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Device: {Config.DEVICE}")

    # 2. Load Data
    print("Loading data...")
    # We use load_cached_data=True to speed up if artifacts exist
    df_train, df_test = load_data(load_cached_data=True, debug=Config.DEBUG)

    # Map string labels to integers
    label_map = {"EAP": 0, "HPL": 1, "MWS": 2}
    y_true = df_train["author"].map(label_map).values

    # 3. Classical Features
    print("Generating classical features...")
    # We pass df_train['text'] twice because the function expects train/val split,
    # but we want features for the whole train set to be split later by indices in run_classical_models.
    # This is consistent with the pipeline design.
    full_tfidf, _, test_tfidf, full_svd, _, test_svd = get_classical_features(
        df_train["text"], df_train["text"], df_test["text"]
    )

    # 4. Run Classical Models
    print("Running classical models...")
    oof_preds, test_preds = run_classical_models(
        df_train,
        df_test,
        full_tfidf,
        test_tfidf,
        full_svd,
        test_svd,
        load_cached_data=True,
    )

    # 5. Run Transformer Models
    # Define models configuration
    transformer_configs = [
        {
            "name": "deberta",
            "model_class": CustomDeberta,
            "model_path": Config.MODEL_DEBERTA,
            "lr": 1e-5,
        },
        {
            "name": "roberta",
            "model_class": CustomRoberta,
            "model_path": Config.MODEL_ROBERTA,
            "lr": 2e-5,
        },
    ]

    for config in transformer_configs:
        model_name = config["name"]
        print(f"\n=== Processing {model_name.upper()} ===")

        # Check cache
        oof_cache_path = f"oof_{model_name}.npy"
        test_cache_path = f"pred_test_{model_name}.npy"

        cached_oof = load_numpy(oof_cache_path)
        cached_test = load_numpy(test_cache_path)

        if cached_oof is not None and cached_test is not None:
            print(f"Loaded {model_name} predictions from cache.")
            oof_preds[model_name] = cached_oof
            test_preds[model_name] = cached_test
            continue

        # Initialize containers
        n_train = len(df_train)
        n_test = len(df_test)
        oof_preds[model_name] = np.zeros((n_train, 3))
        test_preds[model_name] = np.zeros((n_test, 3))

        # Cross-Validation Loop
        for fold in range(Config.NUM_FOLDS):
            print(f"  Fold {fold + 1}/{Config.NUM_FOLDS}")

            # Split Data
            train_df_fold = df_train[df_train["fold"] != fold]
            val_df_fold = df_train[df_train["fold"] == fold]

            val_idx = val_df_fold.index.values

            # Create Datasets
            train_dataset = TextDataset(
                train_df_fold["text"].values,
                train_df_fold["author"].values,
                tokenizer_name=config["model_path"],
                max_len=Config.MAX_LEN,
            )
            val_dataset = TextDataset(
                val_df_fold["text"].values,
                val_df_fold["author"].values,
                tokenizer_name=config["model_path"],
                max_len=Config.MAX_LEN,
            )

            # Create DataLoaders
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

            # Initialize Model
            model = config["model_class"](
                model_name=config["model_path"], num_classes=3
            )

            # Optimizer & Scheduler
            optimizer = AdamW(
                model.parameters(), lr=config["lr"], weight_decay=Config.WEIGHT_DECAY
            )

            # Use a limited number of epochs for the fast baseline requirement
            # 2 epochs is usually enough for these large models to converge reasonably well
            num_epochs = 2

            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

            # Train
            best_loss, best_val_preds = run_training(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                device=Config.DEVICE,
                num_epochs=num_epochs,
                patience=Config.PATIENCE,
                fold=fold,
                model_name=model_name,
                scheduler=scheduler,
            )

            # Store OOF
            oof_preds[model_name][val_idx] = best_val_preds

            # Predict on Test
            # Load best model state to ensure we use the best version
            model.load_state_dict(
                torch.load(
                    os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.bin"),
                    map_location=Config.DEVICE,
                )
            )

            test_dataset = TextDataset(
                df_test["text"].values,
                labels=None,
                tokenizer_name=config["model_path"],
                max_len=Config.MAX_LEN,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE * 2,  # Larger batch for inference
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            fold_test_preds = predict_fn(test_loader, model, Config.DEVICE)
            test_preds[model_name] += fold_test_preds / Config.NUM_FOLDS

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            )
            torch.cuda.empty_cache()

        # Cache predictions
        save_numpy(oof_preds[model_name], oof_cache_path)
        save_numpy(test_preds[model_name], test_cache_path)

    # 6. Ensemble
    print("\n=== Running Ensemble ===")
    test_ids = df_test["id"].values

    # run_ensemble trains the meta-learner and generates test predictions
    # It also prints an internal OOF score, but we will calculate it explicitly below
    final_test_probs = run_ensemble(oof_preds, test_preds, y_true, test_ids)

    # 7. Validation & Failure Analysis
    print("\n=== Validation & Failure Analysis ===")

    # Re-train meta-learner on OOF to get the exact OOF probabilities for analysis
    # (run_ensemble does this internally but we want the objects for analysis)
    from library.ensemble import prepare_meta_features, MetaLearner

    X_train_meta = prepare_meta_features(oof_preds)
    meta_learner = MetaLearner()
    meta_learner.fit(X_train_meta, y_true)
    final_oof_probs = meta_learner.predict_proba(X_train_meta)
    final_oof_probs_clipped = clip_probabilities(final_oof_probs)

    # Compute Final Metric
    final_metric = log_loss(y_true, final_oof_probs_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Word Count
    # Calculate sample-wise log loss
    # We need to index the probability of the true class for each sample
    # log_loss = -log(p_true)

    # Get probability assigned to true class
    rows = np.arange(len(y_true))
    true_class_probs = final_oof_probs_clipped[rows, y_true]
    sample_losses = -np.log(true_class_probs)

    # Calculate word counts
    word_counts = df_train["text"].apply(lambda x: len(str(x).split())).values

    # Calculate correlation
    correlation = np.corrcoef(word_counts, sample_losses)[0, 1]
    print(f"Correlation (Word Count vs Error): {correlation:.6f}")

    # 8. Submission
    THRESHOLD = 0.23237805822413304
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Saving submission...")
        save_submission(test_ids, final_test_probs, filename="submission.csv")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
