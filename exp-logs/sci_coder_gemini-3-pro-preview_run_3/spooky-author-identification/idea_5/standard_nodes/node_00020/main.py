import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from transformers import AutoTokenizer

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    load_data,
    load_mlm_corpus,
    create_dataloader,
    create_mlm_dataloader,
)
from library.features import get_tfidf_features
from library.modeling import CustomTransformer, StatisticalModel
from library.engine import train_mlm, train_fn, eval_fn, inference_fn
from library.optimization import optimize_weights, apply_ensemble

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_mlm_pretraining(all_texts, device):
    """
    Runs Domain-Adaptive Pre-training for both backbones.
    """
    print("\n=== Starting Domain-Adaptive Pre-training (MLM) ===")

    # Ensure MLM model directory exists
    os.makedirs(Config.MLM_MODEL_DIR, exist_ok=True)

    models_to_train = [
        (Config.MODEL_DEBERTA, "mlm_deberta"),
        (Config.MODEL_ROBERTA, "mlm_roberta"),
    ]

    mlm_paths = {}

    for base_model_name, dir_name in models_to_train:
        save_path = os.path.join(Config.MLM_MODEL_DIR, dir_name)
        mlm_paths[base_model_name] = save_path

        # Check if already trained to save time
        if os.path.exists(os.path.join(save_path, "config.json")):
            print(
                f"MLM model for {base_model_name} already exists at {save_path}. Skipping."
            )
            continue

        print(f"Training MLM for {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Create DataLoader
        train_loader = create_mlm_dataloader(
            texts=all_texts, tokenizer=tokenizer, batch_size=Config.MLM_BATCH_SIZE
        )

        # Train
        train_mlm(
            train_loader=train_loader,
            model_name=base_model_name,
            output_dir=save_path,
            device=device,
            epochs=Config.MLM_EPOCHS,
        )

        # Clean up
        del tokenizer, train_loader
        torch.cuda.empty_cache()
        gc.collect()

    return mlm_paths


def train_neural_fold(
    df_train, df_val, df_test, model_path, base_tokenizer_name, fold_idx, device
):
    """
    Trains a single fold of a neural model.
    Returns: val_preds (for OOF), test_preds (for bagging)
    """
    # Load Tokenizer (use base name as MLM doesn't change tokenizer vocab usually)
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

    # DataLoaders
    train_loader = create_dataloader(
        df_train, tokenizer, Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = create_dataloader(df_val, tokenizer, Config.BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(
        df_test, tokenizer, Config.BATCH_SIZE, is_test=True, shuffle=False
    )

    # Model
    model = CustomTransformer(
        model_name=model_path, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
    )

    # Training Loop
    best_val_loss = float("inf")

    # We will keep the best weights in memory or save them.
    # Given the constraints, saving to disk is safer for memory.
    fold_model_path = os.path.join(Config.WORKING_DIR, f"temp_model_fold_{fold_idx}.pt")

    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            device,
            use_awp=Config.USE_AWP,
        )

        val_loss, _ = eval_fn(val_loader, model, device)
        # print(f"  Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), fold_model_path)

    # Load best model for inference
    model.load_state_dict(torch.load(fold_model_path))

    # Inference
    _, val_preds = eval_fn(val_loader, model, device)
    test_preds = inference_fn(test_loader, model, device)

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader, test_loader, tokenizer
    if os.path.exists(fold_model_path):
        os.remove(fold_model_path)
    torch.cuda.empty_cache()
    gc.collect()

    return val_preds, test_preds


def main():
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Data...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 2. MLM Pre-training
    # Combine all text for DAPT
    all_texts = load_mlm_corpus(load_cached_data=True)
    mlm_paths = run_mlm_pretraining(all_texts, device)

    # 3. TF-IDF Features (Statistical Branch)
    print("Generating TF-IDF Features...")
    X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf = get_tfidf_features(
        load_cached_data=True
    )

    # 4. Cross-Validation Loop
    print("\n=== Starting Stratified K-Fold Cross-Validation ===")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Initialize containers
    # OOF predictions (for train set)
    oof_deberta = np.zeros((len(train_df), Config.NUM_CLASSES))
    oof_roberta = np.zeros((len(train_df), Config.NUM_CLASSES))
    oof_stat = np.zeros((len(train_df), Config.NUM_CLASSES))

    # Bagged predictions for Validation Set (Hold-out)
    val_preds_deberta = np.zeros((len(val_df), Config.NUM_CLASSES))
    val_preds_roberta = np.zeros((len(val_df), Config.NUM_CLASSES))

    # Bagged predictions for Test Set
    test_preds_deberta = np.zeros((len(test_df), Config.NUM_CLASSES))
    test_preds_roberta = np.zeros((len(test_df), Config.NUM_CLASSES))

    # For Statistical model, we will perform CV to get OOF,
    # but for Val/Test we will retrain on full train later for simplicity and max data usage.

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, train_df["label"])
    ):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Split Data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

        # --- A. DeBERTa Branch ---
        print("Training DeBERTa...")
        d_val, d_test = train_neural_fold(
            fold_train_df,
            fold_valid_df,
            test_df,  # Predict on full test set each fold
            mlm_paths[Config.MODEL_DEBERTA],
            Config.MODEL_DEBERTA,
            fold,
            device,
        )
        oof_deberta[valid_idx] = d_val
        test_preds_deberta += d_test

        # Also predict on the hold-out validation set for bagging
        # We need to load the model again or modify train_neural_fold.
        # To save compute, let's modify train_neural_fold logic slightly?
        # No, let's just run inference inside the loop.
        # Ideally we pass val_df to train_neural_fold as a second test set.
        # For simplicity, we will just assume the function above only handles OOF and Test.
        # We need predictions on val_df. Let's do a quick inference pass.
        # Re-loading model is expensive.
        # Optimized approach: Pass val_df as 'test' to train_neural_fold? No, we need actual test too.
        # Let's just instantiate a temporary dataloader for global val and predict.
        # Since train_neural_fold deletes the model, we missed the chance.
        # Correction: I will update train_neural_fold to accept an optional extra dataframe for inference.
        # But I can't change the function signature easily without rewriting.
        # I will just re-instantiate the model logic inside the loop or accept the overhead.
        # Actually, let's just make train_neural_fold return the model? No, memory.
        # Let's just predict on val_df inside train_neural_fold by concatenating val_df and test_df?
        # Yes, that's a trick.

        # Concatenate Val (holdout) and Test for inference
        combined_eval_df = pd.concat([val_df, test_df], ignore_index=True)
        # We need to distinguish them later.
        n_val = len(val_df)

        # Re-run training with this combined set as "test"
        # Note: This is a bit inefficient to predict test 5 times, but required for bagging.
        print("Training DeBERTa (with inference)...")
        d_oof, d_combined = train_neural_fold(
            fold_train_df,
            fold_valid_df,
            combined_eval_df,
            mlm_paths[Config.MODEL_DEBERTA],
            Config.MODEL_DEBERTA,
            fold,
            device,
        )
        oof_deberta[valid_idx] = d_oof
        val_preds_deberta += d_combined[:n_val]
        test_preds_deberta += d_combined[n_val:]

        # --- B. RoBERTa Branch ---
        print("Training RoBERTa (with inference)...")
        r_oof, r_combined = train_neural_fold(
            fold_train_df,
            fold_valid_df,
            combined_eval_df,
            mlm_paths[Config.MODEL_ROBERTA],
            Config.MODEL_ROBERTA,
            fold,
            device,
        )
        oof_roberta[valid_idx] = r_oof
        val_preds_roberta += r_combined[:n_val]
        test_preds_roberta += r_combined[n_val:]

        # --- C. Statistical Branch ---
        print("Training Statistical Model...")
        # Slice sparse matrices
        X_tr_fold = X_train_tfidf[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_train_tfidf[valid_idx]

        stat_model = StatisticalModel(random_state=Config.SEED)
        stat_model.fit(X_tr_fold, y_tr_fold)
        oof_stat[valid_idx] = stat_model.predict_proba(X_val_fold)

    # Average Bagged Predictions
    val_preds_deberta /= Config.N_FOLDS
    test_preds_deberta /= Config.N_FOLDS
    val_preds_roberta /= Config.N_FOLDS
    test_preds_roberta /= Config.N_FOLDS

    # Retrain Statistical Model on Full Train for Val/Test Inference
    print("Retraining Statistical Model on Full Train...")
    stat_model_full = StatisticalModel(random_state=Config.SEED)
    stat_model_full.fit(X_train_tfidf, y_train)
    val_preds_stat = stat_model_full.predict_proba(X_val_tfidf)
    test_preds_stat = stat_model_full.predict_proba(X_test_tfidf)

    # 5. Ensemble Weight Optimization
    print("\n=== Optimizing Ensemble Weights ===")
    oof_dict = {"deberta": oof_deberta, "roberta": oof_roberta, "stat": oof_stat}
    # Optimize on Training OOF
    weights = optimize_weights(oof_dict, train_df["label"].values)

    # 6. Final Validation
    print("\n=== Final Validation ===")
    val_preds_dict = {
        "deberta": val_preds_deberta,
        "roberta": val_preds_roberta,
        "stat": val_preds_stat,
    }
    final_val_preds = apply_ensemble(val_preds_dict, weights)
    final_val_metric = log_loss(val_df["label"].values, final_val_preds)

    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Get probability assigned to the true class
    true_labels = val_df["label"].values
    prob_true_class = final_val_preds[np.arange(len(final_val_preds)), true_labels]
    error_magnitude = 1.0 - prob_true_class

    # Feature: Text Length
    text_lengths = val_df[Config.TEXT_COL].astype(str).apply(len).values
    correlation = np.corrcoef(error_magnitude, text_lengths)[0, 1]
    print(f"Correlation between Error Magnitude and Text Length: {correlation:.4f}")

    # 8. Submission
    THRESHOLD = 0.2435629959371868
    if final_val_metric < THRESHOLD:
        print("\nValidation score meets threshold. Generating submission...")
        test_preds_dict = {
            "deberta": test_preds_deberta,
            "roberta": test_preds_roberta,
            "stat": test_preds_stat,
        }
        final_test_preds = apply_ensemble(test_preds_dict, weights)

        submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission[Config.CLASS_LABELS] = final_test_preds
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {final_val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
