import os
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional

from library.config import Config
from library.utils import seed_everything
from library.model_rf import InteractionRandomForest
from library.model_mlp import MLPPipeline
from library.dataset import get_dataloaders


def train_rf_model(
    rf_data: Dict[str, np.ndarray], force_retrain: bool = True
) -> InteractionRandomForest:
    """
    Trains the Random Forest model using the provided data.

    Args:
        rf_data: Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val'.
        force_retrain: If True, forces training even if a saved model exists.

    Returns:
        Trained InteractionRandomForest instance.
    """
    seed_everything()

    model = InteractionRandomForest()

    # Check if model exists and we are not forcing retrain
    if not force_retrain and os.path.exists(model.model_path):
        print(f"RF model found at {model.model_path}. Skipping training.")
        # The predict method in InteractionRandomForest handles loading
        return model

    print("Starting Random Forest training...")
    model.train(
        X_train=rf_data["X_train"],
        y_train=rf_data["y_train"],
        X_val=rf_data.get("X_val"),
        y_val=rf_data.get("y_val"),
    )

    return model


def predict_rf(model: InteractionRandomForest, X: np.ndarray) -> np.ndarray:
    """
    Generates predictions using the Random Forest model.

    Args:
        model: InteractionRandomForest instance.
        X: Feature matrix.

    Returns:
        Array of probabilities for the positive class.
    """
    return model.predict(X)


def train_mlp_model(
    mlp_data: Dict[str, Any], force_retrain: bool = True
) -> MLPPipeline:
    """
    Trains the MLP model using the provided data.

    Args:
        mlp_data: Dictionary containing nested 'train', 'val', 'test' data.
        force_retrain: If True, forces training even if a saved model exists.

    Returns:
        Trained MLPPipeline instance.
    """
    seed_everything()

    # Create DataLoaders
    # We use a modest number of workers to avoid overhead on smaller datasets
    train_loader, val_loader, _ = get_dataloaders(
        mlp_data, batch_size=Config.BATCH_SIZE, num_workers=2
    )

    # Determine metadata dimension dynamically from the first batch
    # This ensures the model architecture matches the data engineering
    sample_batch = next(iter(train_loader))
    metadata_dim = sample_batch["metadata"].shape[1]

    pipeline = MLPPipeline(metadata_dim=metadata_dim)

    # Check if model exists and we are not forcing retrain
    if not force_retrain and os.path.exists(pipeline.model_path):
        print(f"MLP model found at {pipeline.model_path}. Loading weights...")
        pipeline.model.load_state_dict(
            torch.load(pipeline.model_path, map_location=pipeline.device)
        )
        return pipeline

    print("Starting MLP training...")
    pipeline.train(train_loader, val_loader)

    return pipeline


def predict_mlp(
    pipeline: MLPPipeline, mlp_data: Dict[str, Any], split: str = "test"
) -> np.ndarray:
    """
    Generates predictions using the MLP model for a specific data split.

    Args:
        pipeline: Trained MLPPipeline instance.
        mlp_data: Dictionary containing nested data.
        split: The data split to predict on ('val' or 'test').

    Returns:
        Array of probabilities for the positive class.
    """
    # We regenerate loaders here. Since data is in memory/numpy, this is fast.
    # It avoids passing loaders around which might have different states.
    _, val_loader, test_loader = get_dataloaders(
        mlp_data, batch_size=Config.BATCH_SIZE, num_workers=2
    )

    if split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'val' or 'test'.")

    print(f"Generating MLP predictions for {split} set...")
    probs = pipeline.predict(loader)

    return probs
