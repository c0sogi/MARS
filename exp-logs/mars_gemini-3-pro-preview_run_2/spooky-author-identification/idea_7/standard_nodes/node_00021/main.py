import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.features import FeatureEngineer
from library.trainers import ModelTrainer
from library.meta_learner import StackingEnsemble
from library.neural_net import DebertaWithMSD
from library.dataset import AuthorDataset


def predict_neural(model_state, texts, tokenizer, device):
    """
    Helper function to perform inference on test data using the Neural Model.
    """
    dataset = AuthorDataset(texts, None, tokenizer, max_length=Config.MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = DebertaWithMSD(num_classes=3)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def main():
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Feature Engineering & Data Loading
    print("\n--- Loading Data & Features ---")
    fe = FeatureEngineer()
    data = fe.process_data(load_cached_data=True)

    train_df = data["train_df"]
    test_df = data["test_df"]
    y_train = data["y_train"]
    X_train_sparse = data["X_train_sparse"]
    X_test_sparse = data["X_test_sparse"]
    X_train_dense = data["X_train_dense"]
    X_test_dense = data["X_test_dense"]

    # Prepare text data for Neural Net
    train_texts_full = train_df["text"].fillna("").astype(str).values
    test_texts = test_df["text"].fillna("").astype(str).values

    # 2. Initialization for CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    trainer = ModelTrainer()

    # Storage for OOF predictions (N_samples, N_classes)
    n_samples = len(y_train)
    n_classes = 3

    oof_preds = {
        "nn": np.zeros((n_samples, n_classes)),
        "lr": np.zeros((n_samples, n_classes)),
        "nb": np.zeros((n_samples, n_classes)),
        "xgb": np.zeros((n_samples, n_classes)),
    }

    # Storage for Test predictions (List of arrays, to be averaged)
    test_preds_fold = {"nn": [], "lr": [], "nb": [], "xgb": []}

    # 3. Cross-Validation Loop
    print("\n--- Starting Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts_full, y_train)):
        print(f"\nProcessing Fold {fold + 1}/{Config.N_FOLDS}")

        # Split Data
        # Text (for NN)
        X_tr_text, X_val_text = train_texts_full[train_idx], train_texts_full[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Sparse (for LR, NB)
        X_tr_sparse, X_val_sparse = X_train_sparse[train_idx], X_train_sparse[val_idx]

        # Dense (for XGB)
        X_tr_dense, X_val_dense = X_train_dense[train_idx], X_train_dense[val_idx]

        # --- A. Neural Network ---
        val_probs_nn, best_state_nn = trainer.train_neural_fold(
            X_tr_text, y_tr, X_val_text, y_val, fold
        )
        oof_preds["nn"][val_idx] = val_probs_nn

        # Inference on Test Set
        test_probs_nn = predict_neural(
            best_state_nn, test_texts, trainer.tokenizer, device
        )
        test_preds_fold["nn"].append(test_probs_nn)

        # --- B. Logistic Regression ---
        val_probs_lr, model_lr = trainer.train_classical_fold(
            X_tr_sparse, y_tr, X_val_sparse, y_val, "lr", fold
        )
        oof_preds["lr"][val_idx] = val_probs_lr
        test_preds_fold["lr"].append(model_lr.predict_proba(X_test_sparse))

        # --- C. Naive Bayes ---
        val_probs_nb, model_nb = trainer.train_classical_fold(
            X_tr_sparse, y_tr, X_val_sparse, y_val, "nb", fold
        )
        oof_preds["nb"][val_idx] = val_probs_nb
        test_preds_fold["nb"].append(model_nb.predict_proba(X_test_sparse))

        # --- D. XGBoost ---
        val_probs_xgb, model_xgb = trainer.train_classical_fold(
            X_tr_dense, y_tr, X_val_dense, y_val, "xgb", fold
        )
        oof_preds["xgb"][val_idx] = val_probs_xgb
        test_preds_fold["xgb"].append(model_xgb.predict_proba(X_test_dense))

    # 4. Aggregate Test Predictions
    print("\n--- Aggregating Predictions ---")
    test_preds_avg = {}
    for key in test_preds_fold:
        test_preds_avg[key] = np.mean(test_preds_fold[key], axis=0)

    # 5. Stacking (Meta-Learner)
    print("\n--- Training Meta-Learner ---")
    stacker = StackingEnsemble()
    stacker.fit(oof_preds, y_train)

    # Calculate Final Validation Metric
    # We use the meta-learner's prediction on the OOF data (which acts as the validation set for the stacker)
    # Note: stacker.fit() prints the OOF log loss, but we calculate it here explicitly for the requirement.
    # To get the exact metric, we predict on the OOF input again.
    # Re-construct X_meta as done in StackingEnsemble
    model_keys = sorted(oof_preds.keys())
    X_meta = np.hstack([oof_preds[k] for k in model_keys])
    final_val_probs = stacker.meta_model.predict_proba(X_meta)

    final_metric = calculate_log_loss(y_train, final_val_probs)
    print(f"Final Validation Metric: {final_val_probs.shape}")  # Debug print
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    # y_train is (N,), final_val_probs is (N, 3)
    # Select the probability assigned to the true class
    row_indices = np.arange(len(y_train))
    true_class_probs = final_val_probs[row_indices, y_train]
    # Clip for safety
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)
    sample_losses = -np.log(true_class_probs)

    # Extract metadata features from Dense matrix
    # Based on features.py: [char_len, word_len, avg_word_len, p1, p2...]
    # SVD components are first. Config.SVD_N_COMPONENTS = 100.
    svd_dim = Config.SVD_N_COMPONENTS

    char_lens = X_train_dense[:, svd_dim]
    word_lens = X_train_dense[:, svd_dim + 1]

    # Calculate correlations
    corr_char = np.corrcoef(sample_losses, char_lens)[0, 1]
    corr_word = np.corrcoef(sample_losses, word_lens)[0, 1]

    print(f"Correlation (Error vs Char Length): {corr_char:.6f}")
    print(f"Correlation (Error vs Word Length): {corr_word:.6f}")

    # 7. Submission
    TARGET_METRIC = 0.23237805822413304

    if final_metric < TARGET_METRIC:
        print("\nMetric threshold met. Generating submission...")
        final_test_probs = stacker.predict(test_preds_avg)
        test_ids = test_df["id"].values
        stacker.create_submission(test_ids, final_test_probs)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
