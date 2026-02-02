import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.features import get_tfidf_features, get_meta_features
from library.dataset import AuthorDataset
from library.modeling import DebertaClassifier
from library.engine import run_transformer_fold, eval_fn

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# Limit epochs to ensure execution within 2 hours.
# DeBERTa-Large is heavy; 3 epochs with early stopping is sufficient for stacking.
Config.EPOCHS = 3
Config.create_directories()


def load_data():
    """Loads metadata CSVs."""
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)
    return train_df, val_df, test_df


def get_transformer_preds(model_path, loader, device):
    """
    Loads a trained transformer model and generates predictions for a dataloader.
    """
    model = DebertaClassifier(Config.MODEL_NAME, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # eval_fn returns (loss, preds)
    # We only need preds here. We pass dummy loss calculation if labels exist,
    # but for test set labels might be missing or dummy.
    # eval_fn handles labels, so we rely on it returning preds correctly.
    _, preds = eval_fn(model, loader, device)

    del model
    torch.cuda.empty_cache()
    return preds


def analyze_failures(y_true, y_pred, meta_features, feature_names):
    """
    Analyzes correlation between error magnitude and meta-features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate Log Loss per sample (Cross Entropy)
    # y_pred is (N, 3), y_true is (N,) class indices
    # We pick the probability assigned to the true class

    # Clip preds for stability
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Gather prob of true class
    n_samples = len(y_true)
    rows = np.arange(n_samples)
    true_class_probs = y_pred[rows, y_true]

    # Loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    print(f"Average Sample Loss: {np.mean(sample_losses):.5f}")

    # Correlation with meta features
    print("\nCorrelation between Error (LogLoss) and Meta-features:")
    for i, name in enumerate(feature_names):
        feat_values = meta_features[:, i]
        corr = np.corrcoef(sample_losses, feat_values)[0, 1]
        print(f"  {name}: {corr:.4f}")

    print("=" * 40 + "\n")


def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting Pipeline...")

    # 1. Load Data
    train_df, val_df, test_df = load_data()

    # Combine train and val for TF-IDF/Meta computation consistency if needed,
    # but library functions handle them separately.
    # Note: The library functions fit on train_df only.

    # 2. Feature Engineering
    print("\n--- Generating Features ---")
    # TF-IDF (Sparse)
    train_tfidf, val_tfidf, test_tfidf = get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Meta-features (Dense: char_len, word_count, punct_density)
    train_meta, val_meta, test_meta = get_meta_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 3. Prepare Containers for Stacking
    # We need OOF predictions for the training set
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    # Level 1 Predictions
    # Expert A: Transformer
    oof_preds_a = np.zeros((n_train, 3))
    val_preds_a = np.zeros((n_val, 3))
    test_preds_a = np.zeros((n_test, 3))

    # Expert B: Linear
    oof_preds_b = np.zeros((n_train, 3))
    val_preds_b = np.zeros((n_val, 3))
    test_preds_b = np.zeros((n_test, 3))

    # Labels
    y_train = train_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
    y_val = val_df["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values

    # 4. Cross-Validation Loop
    print("\n--- Starting 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Prepare Tokenizer once
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create fixed DataLoaders for Hold-out Val and Test (used for inference in every fold)
    val_ds_full = AuthorDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_ds_full = AuthorDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader_full = DataLoader(
        test_ds_full,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    for fold, (train_idx, oof_idx) in enumerate(skf.split(train_df, y_train)):
        print(f"\nProcessing Fold {fold}...")

        # --- Expert B: Linear Model ---
        print("Training Expert B (Linear)...")
        X_tr_tfidf = train_tfidf[train_idx]
        X_oof_tfidf = train_tfidf[oof_idx]
        y_tr = y_train[train_idx]

        clf = LogisticRegression(
            C=Config.LOGREG_C,
            solver=Config.LOGREG_SOLVER,
            max_iter=Config.LOGREG_MAX_ITER,
            multi_class="multinomial",
            random_state=Config.SEED,
            n_jobs=-1,
        )
        clf.fit(X_tr_tfidf, y_tr)

        # Predict
        oof_preds_b[oof_idx] = clf.predict_proba(X_oof_tfidf)
        val_preds_b += clf.predict_proba(val_tfidf) / Config.N_FOLDS
        test_preds_b += clf.predict_proba(test_tfidf) / Config.N_FOLDS

        # --- Expert A: Transformer ---
        print("Training Expert A (Transformer)...")

        # Subset DataFrames
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_oof_df = train_df.iloc[oof_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = AuthorDataset(
            fold_train_df, tokenizer, Config.MAX_LEN, is_test=False
        )
        oof_ds = AuthorDataset(fold_oof_df, tokenizer, Config.MAX_LEN, is_test=False)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        oof_loader = DataLoader(
            oof_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Train
        # run_transformer_fold saves the model to checkpoint dir and returns best OOF preds
        _, best_oof_preds = run_transformer_fold(fold, train_loader, oof_loader)
        oof_preds_a[oof_idx] = best_oof_preds

        # Inference on Hold-out Val and Test using the best checkpoint
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"expert_a_fold_{fold}.pt"
        )
        print(f"Running inference on Val/Test with checkpoint: {checkpoint_path}")

        fold_val_preds = get_transformer_preds(checkpoint_path, val_loader_full, device)
        fold_test_preds = get_transformer_preds(
            checkpoint_path, test_loader_full, device
        )

        val_preds_a += fold_val_preds / Config.N_FOLDS
        test_preds_a += fold_test_preds / Config.N_FOLDS

        # Cleanup
        del clf, train_ds, oof_ds, train_loader, oof_loader
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Meta-Learner (XGBoost)
    print("\n--- Training Meta-Learner (XGBoost) ---")

    # Construct Level 2 Datasets
    # Features: [Prob_A (3), Prob_B (3), Meta (3)]
    X_meta_train = np.column_stack([oof_preds_a, oof_preds_b, train_meta])
    X_meta_val = np.column_stack([val_preds_a, val_preds_b, val_meta])
    X_meta_test = np.column_stack([test_preds_a, test_preds_b, test_meta])

    print(f"Meta-Feature Shape: {X_meta_train.shape}")

    xgb_model = xgb.XGBClassifier(**Config.XGB_PARAMS)

    # Fit on OOF predictions
    # Use X_meta_val for early stopping to prevent overfitting the meta-learner
    xgb_model.fit(X_meta_train, y_train, eval_set=[(X_meta_val, y_val)], verbose=False)

    # 6. Evaluation
    print("\n--- Evaluating ---")

    # Predict on Hold-out Validation
    final_val_probs = xgb_model.predict_proba(X_meta_val)
    final_score = calculate_log_loss(y_val, final_val_probs)

    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    meta_feature_names = ["Char Length", "Word Count", "Punct Density"]
    analyze_failures(y_val, final_val_probs, val_meta, meta_feature_names)

    # 7. Submission
    THRESHOLD = 0.25336663725445785
    if final_score < THRESHOLD:
        print("Score meets threshold. Generating submission...")
        final_test_probs = xgb_model.predict_proba(X_meta_test)

        # Rescale and clip as per metric (handled by calculate_log_loss logic, but here for submission)
        # The metric function does rescaling, but for submission we just output probabilities.
        # It's good practice to ensure they sum to 1.
        row_sums = final_test_probs.sum(axis=1, keepdims=True)
        final_test_probs = final_test_probs / row_sums

        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "EAP": final_test_probs[:, 0],
                "HPL": final_test_probs[:, 1],
                "MWS": final_test_probs[:, 2],
            }
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Score {final_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
