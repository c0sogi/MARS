import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config
from library.neural_arch import MLPTrainer


def train_random_forest(train_data, val_data):
    """
    Trains the Random Forest Classifier for Stream A (Lexical-Tabular).

    Args:
        train_data: Tuple (X_train, y_train) - Sparse matrix and target array.
        val_data: Tuple (X_val, y_val) - Sparse matrix and target array.

    Returns:
        model: The trained RandomForestClassifier instance.
    """
    print("Initializing Random Forest Classifier...")
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Initialize model with configuration parameters
    model = RandomForestClassifier(**config.RF_PARAMS)

    # Train the model
    print("Fitting Random Forest...")
    model.fit(X_train, y_train)

    # Validate
    print("Validating Random Forest...")
    # Predict probabilities for the positive class
    val_preds = model.predict_proba(X_val)[:, 1]

    # Calculate AUC
    val_auc = roc_auc_score(y_val, val_preds)

    # Print full precision metric
    print(f"Random Forest Validation AUC: {val_auc}")

    return model


def train_dual_branch_mlp(train_data, val_data):
    """
    Trains the Dual-Branch MLP for Stream B (Semantic-Tabular).
    Uses the MLPTrainer to handle the PyTorch training loop and early stopping.

    Args:
        train_data: Tuple (X_sem_train, X_meta_train, y_train)
        val_data: Tuple (X_sem_val, X_meta_val, y_val)

    Returns:
        trainer: The MLPTrainer instance containing the trained model.
    """
    print("Initializing Dual-Branch MLP Trainer...")

    # Initialize trainer with configuration parameters
    trainer = MLPTrainer(config.MLP_PARAMS)

    # Execute training loop (handles optimization, validation, early stopping)
    # The trainer prints metrics internally
    trainer.train(train_data, val_data)

    return trainer
