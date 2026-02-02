import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from library.config import CACHE_DIR
from library.utils import save_pickle, seed_everything
from library.feature_engineering import FeaturePipeline
from library.dataset import create_dataloaders
from library.models import InteractionRandomForest, train_mlp_model


def train_rf(load_cached_data=True):
    """
    Trains the Interaction-Enhanced Random Forest (Stream A).

    Args:
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        model: The trained InteractionRandomForest instance.
    """
    seed_everything()

    print("Initializing Feature Pipeline for Random Forest...")
    pipeline = FeaturePipeline()
    rf_out, _ = pipeline.run(load_cached_data=load_cached_data)

    X_train = rf_out["train_X"]
    y_train = rf_out["train_y"]
    X_val = rf_out["val_X"]
    y_val = rf_out["val_y"]

    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    model = InteractionRandomForest()
    model.fit(X_train, y_train)

    # Validation
    print("Evaluating Random Forest...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"RF Validation AUC: {val_auc:.16f}")

    # Save model
    save_path = os.path.join(CACHE_DIR, "rf_model.pkl")
    save_pickle(model, save_path)
    print(f"Random Forest model saved to {save_path}")

    return model


def train_mlp(load_cached_data=True, batch_size=32):
    """
    Trains the FiLM-Conditioned MLP (Stream B).

    Args:
        load_cached_data (bool): Whether to load features from cache.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        model: The trained PizzaFiLMMLP instance.
    """
    seed_everything()

    print("Initializing DataLoaders for MLP...")
    train_loader, val_loader, _ = create_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # Determine metadata dimension from a sample batch
    # The dataset returns a dict, we access 'metadata' key
    sample_batch = next(iter(train_loader))
    metadata_dim = sample_batch["metadata"].shape[1]

    print(f"Detected Metadata Dimension: {metadata_dim}")

    # Delegate to the library's training loop
    model = train_mlp_model(train_loader, val_loader, metadata_dim)

    return model
