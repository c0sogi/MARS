import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.utils import set_seed, create_submission_file
from library.data_loader import load_dataset
from library.model_factory import (
    build_linear_branch,
    build_generative_branch,
    build_kernel_branch,
)


def train_ensemble(X_train, y_train, random_state=42):
    """
    Trains the three branches of the ensemble on the provided data.
    Calculates and prints training log loss for each branch to verify convergence.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing the trained models.
    """
    print("Initializing ensemble training...")

    # 1. Linear Branch (Discriminative Linear)
    print("Training Linear Branch (Logistic Regression)...")
    linear_model = build_linear_branch(random_state=random_state)
    linear_model.fit(X_train, y_train)

    # Calculate training metric
    train_probs_linear = linear_model.predict_proba(X_train)
    loss_linear = log_loss(y_train, train_probs_linear)
    print(f"Linear Branch Training Log Loss: {loss_linear}")

    # 2. Generative Branch (LDA)
    print("Training Generative Branch (LDA)...")
    generative_model = build_generative_branch()
    generative_model.fit(X_train, y_train)

    # Calculate training metric
    train_probs_gen = generative_model.predict_proba(X_train)
    loss_gen = log_loss(y_train, train_probs_gen)
    print(f"Generative Branch Training Log Loss: {loss_gen}")

    # 3. Kernel Branch (Discriminative Non-Linear)
    print("Training Kernel Branch (Nystroem + Logistic Regression)...")
    kernel_model = build_kernel_branch(random_state=random_state)
    kernel_model.fit(X_train, y_train)

    # Calculate training metric
    train_probs_kernel = kernel_model.predict_proba(X_train)
    loss_kernel = log_loss(y_train, train_probs_kernel)
    print(f"Kernel Branch Training Log Loss: {loss_kernel}")

    print("Ensemble training complete.")

    return {
        "linear": linear_model,
        "generative": generative_model,
        "kernel": kernel_model,
    }


def predict_ensemble(models, X_test):
    """
    Generates predictions using Soft Voting from the trained ensemble.

    Args:
        models (dict): Dictionary of trained models.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Averaged probability matrix.
    """
    print("Generating predictions from Linear Branch...")
    prob_linear = models["linear"].predict_proba(X_test)

    print("Generating predictions from Generative Branch...")
    prob_generative = models["generative"].predict_proba(X_test)

    print("Generating predictions from Kernel Branch...")
    prob_kernel = models["kernel"].predict_proba(X_test)

    # Soft Voting (Average)
    print("Aggregating predictions (Soft Voting)...")
    avg_probs = (prob_linear + prob_generative + prob_kernel) / 3.0

    return avg_probs


def run_pipeline(load_cached_data=True, random_state=42):
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Inference -> Submission.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
        random_state (int): Seed for reproducibility.
    """
    # Set global seed
    set_seed(random_state)

    # 1. Load Data
    # load_dataset concatenates train and val to maximize sample size for final training
    X_train, y_train, X_test, test_ids, label_encoder = load_dataset(
        load_cached_data=load_cached_data
    )

    print(f"Data loaded. Training shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Train Ensemble
    models = train_ensemble(X_train, y_train, random_state=random_state)

    # 3. Inference
    final_probs = predict_ensemble(models, X_test)

    # 4. Create Submission
    submission_path = "./submission/submission.csv"
    class_names = list(label_encoder.classes_)

    create_submission_file(test_ids, class_names, final_probs, submission_path)

    print("Pipeline execution completed successfully.")
