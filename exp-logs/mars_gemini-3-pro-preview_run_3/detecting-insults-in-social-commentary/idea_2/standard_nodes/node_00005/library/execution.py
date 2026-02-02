import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import roc_auc_score

from library.utils import load_data, save_submission, set_seed
from library.data_factory import get_tfidf_features, InsultDataset
from library.modeling import NBSVM, RoBERTaClassifier


def run_nblr(load_cached_data=True, max_samples=None, C=1.0, dual=True):
    """
    Executes the Statistical Stream (NBSVM).

    Args:
        load_cached_data (bool): Whether to use cached data/features.
        max_samples (int, optional): Number of training samples to use (for debugging).
        C (float): Inverse of regularization strength for Logistic Regression.
        dual (bool): Dual or primal formulation for Linear SVC/LR.

    Returns:
        tuple: (val_preds, test_preds) as numpy arrays.
    """
    print("--- Starting NBSVM Stream ---")

    # Load raw dataframes (needed for consistency, though features are loaded separately)
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Get TF-IDF features (cached or computed)
    X_train, y_train, X_val, y_val, X_test = get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Debugging: Truncate training data if max_samples is set
    if max_samples is not None and max_samples < X_train.shape[0]:
        print(f"NBSVM: Truncating training data to {max_samples} samples.")
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]

    # Initialize and Train NBSVM
    print("Training NBSVM model...")
    model = NBSVM(C=C, dual=dual, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Inference
    print("Generating NBSVM predictions...")
    # predict_proba returns [n_samples, 2], we take the probability of class 1
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    # Evaluation
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"NBSVM Validation AUC: {val_auc}")

    return val_preds, test_preds


def run_neural(
    load_cached_data=True,
    max_samples=None,
    epochs=3,
    batch_size=16,
    lr=2e-5,
    device=None,
):
    """
    Executes the Neural Stream (RoBERTa).

    Args:
        load_cached_data (bool): Whether to use cached data.
        max_samples (int, optional): Truncate training data for debugging.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        lr (float): Learning rate.
        device (torch.device, optional): Device to run on.

    Returns:
        tuple: (val_preds, test_preds) as numpy arrays.
    """
    print("--- Starting Neural Stream ---")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Stream using device: {device}")

    # Load Data
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # Debugging: Truncate training data
    if max_samples is not None and max_samples < len(train_df):
        print(f"Neural: Truncating training data to {max_samples} samples.")
        train_df = train_df.iloc[:max_samples].reset_index(drop=True)

    # Initialize Tokenizer
    print("Initializing Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create Datasets
    train_dataset = InsultDataset(train_df, tokenizer)
    val_dataset = InsultDataset(val_df, tokenizer)
    test_dataset = InsultDataset(test_df, tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model
    print("Initializing RoBERTa Model...")
    model = RoBERTaClassifier(model_name="roberta-base")

    # Train Model
    # train_model handles the training loop and early stopping, and loads best weights
    model.train_model(train_loader, val_loader, device, epochs=epochs, lr=lr)

    # Inference
    print("Generating Neural predictions...")
    val_preds = model.predict(val_loader, device)
    test_preds = model.predict(test_loader, device)

    # Evaluation
    y_val = val_df["Insult"].values
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Neural Validation AUC: {val_auc}")

    return val_preds, test_preds


def optimize_ensemble(val_true, pred_a, pred_b):
    """
    Finds the optimal weight w to combine two prediction vectors.
    Formula: w * pred_a + (1 - w) * pred_b

    Args:
        val_true: True labels.
        pred_a: Predictions from model A.
        pred_b: Predictions from model B.

    Returns:
        tuple: (best_weight, best_auc)
    """
    best_auc = 0.0
    best_w = 0.5

    # Grid search for weight w
    for w in np.arange(0.0, 1.01, 0.01):
        combined_preds = w * pred_a + (1 - w) * pred_b
        current_auc = roc_auc_score(val_true, combined_preds)

        if current_auc > best_auc:
            best_auc = current_auc
            best_w = w

    return best_w, best_auc


def run_pipeline(load_cached_data=True, max_samples=None, epochs=3, batch_size=16):
    """
    Main execution pipeline:
    1. Train NBSVM
    2. Train Neural Model
    3. Optimize Ensemble
    4. Generate Submission
    """
    set_seed(42)

    # 1. Run Statistical Stream
    nb_val_preds, nb_test_preds = run_nblr(
        load_cached_data=load_cached_data, max_samples=max_samples
    )

    # 2. Run Neural Stream
    neural_val_preds, neural_test_preds = run_neural(
        load_cached_data=load_cached_data,
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
    )

    # 3. Ensemble Optimization
    print("\n--- Optimizing Ensemble Weights ---")
    val_df = load_data("val", load_cached_data=load_cached_data)
    y_val = val_df["Insult"].values

    best_w, best_auc = optimize_ensemble(y_val, nb_val_preds, neural_val_preds)

    print(f"Optimal Weight for NBSVM: {best_w}")
    print(f"Optimal Weight for Neural: {1 - best_w}")
    print(f"Ensemble Validation AUC: {best_auc}")

    # 4. Generate Final Predictions
    final_test_preds = best_w * nb_test_preds + (1 - best_w) * neural_test_preds

    # 5. Save Submission
    print("\n--- Saving Submission ---")
    test_df = load_data("test", load_cached_data=load_cached_data)
    save_submission(final_test_preds, test_df, output_dir="./submission")
    print("Submission saved to ./submission/submission.csv")
