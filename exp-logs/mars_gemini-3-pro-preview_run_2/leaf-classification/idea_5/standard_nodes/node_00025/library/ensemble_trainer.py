import time
import numpy as np
from sklearn.metrics import log_loss
from library.utils import set_seed
from library.model_factory import (
    build_linear_pipeline,
    build_generative_pipeline,
)


def train_ensemble(X_train, y_train, random_state=42, n_jobs=-1):
    """
    Trains the components of the Soft-Voting Ensemble:
    1. Discriminative Linear (Logistic Regression)
    2. Generative Linear (LDA)

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target vector for training.
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of CPU cores to use.

    Returns:
        dict: A dictionary containing the fitted models with keys 'linear', 'generative'.
    """
    set_seed(random_state)

    models = {}

    print(
        f"Starting ensemble training with {len(X_train)} samples and {X_train.shape[1]} features."
    )

    # ---------------------------------------------------------
    # 1. Linear Component
    # ---------------------------------------------------------
    print("\n[1/2] Training Linear Component (Logistic Regression)...")
    start_time = time.time()

    model_linear = build_linear_pipeline(random_state=random_state, n_jobs=n_jobs)
    model_linear.fit(X_train, y_train)

    # Evaluate on training data to confirm convergence/fit
    y_pred_linear = model_linear.predict_proba(X_train)
    loss_linear = log_loss(y_train, y_pred_linear)

    elapsed_linear = time.time() - start_time
    print(f"Linear Component trained in {elapsed_linear:.2f}s.")
    print(f"Training Log Loss: {loss_linear}")

    models["linear"] = model_linear

    # ---------------------------------------------------------
    # 2. Generative Component
    # ---------------------------------------------------------
    print("\n[2/2] Training Generative Component (LDA)...")
    start_time = time.time()

    model_gen = build_generative_pipeline()
    model_gen.fit(X_train, y_train)

    y_pred_gen = model_gen.predict_proba(X_train)
    loss_gen = log_loss(y_train, y_pred_gen)

    elapsed_gen = time.time() - start_time
    print(f"Generative Component trained in {elapsed_gen:.2f}s.")
    print(f"Training Log Loss: {loss_gen}")

    models["generative"] = model_gen

    print("\nEnsemble training complete.")
    return models
