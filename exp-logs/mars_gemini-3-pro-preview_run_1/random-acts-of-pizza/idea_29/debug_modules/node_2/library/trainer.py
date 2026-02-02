import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from library.config import (
    NN_PARAMS,
    ENSEMBLE_WEIGHTS,
    SUBMISSION_FILE,
    RANDOM_SEED,
    TEST_PATH,
    CACHE_DIR,
)
from library.utils import set_seed, save_submission
from library.data_processing import get_rf_dataset, get_nn_dataset
from library.model_rf import PizzaRandomForest
from library.model_nn import PizzaNeuralNet


def train_rf_model(load_cached_data=True, max_samples=None):
    """
    Trains the Random Forest model (Stream A).

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        max_samples (int, optional): Number of samples to use for debugging.

    Returns:
        model: Trained PizzaRandomForest instance.
        val_auc: Validation ROC AUC score.
    """
    print("\n=== Training Random Forest (Stream A) ===")

    # Load Data
    X_train, y_train = get_rf_dataset(split="train", load_cached_data=load_cached_data)
    X_val, y_val = get_rf_dataset(split="val", load_cached_data=load_cached_data)

    # Subsample if requested
    if max_samples is not None:
        print(f"Subsampling RF data to {max_samples} samples.")
        X_train = X_train.iloc[:max_samples]
        y_train = y_train.iloc[:max_samples]
        # We also subsample validation for quick debugging
        X_val = X_val.iloc[:max_samples]
        y_val = y_val.iloc[:max_samples]

    # Initialize and Train
    model = PizzaRandomForest()
    val_auc = model.train(X_train, y_train, X_val, y_val)

    # Save Model
    model.save()

    return model, val_auc


def train_nn_model(load_cached_data=True, max_samples=None, epochs=None):
    """
    Trains the Neural Network model (Stream B).

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        max_samples (int, optional): Number of samples to use for debugging.
        epochs (int, optional): Override default number of epochs.

    Returns:
        model: Trained PizzaNeuralNet instance.
        val_auc: Validation ROC AUC score.
    """
    print("\n=== Training Neural Network (Stream B) ===")

    # Override epochs if provided
    if epochs is not None:
        NN_PARAMS["epochs"] = epochs
        print(f"Overriding epochs to: {epochs}")

    # Load Data
    train_ds = get_nn_dataset(split="train", load_cached_data=load_cached_data)
    val_ds = get_nn_dataset(split="val", load_cached_data=load_cached_data)

    # Subsample if requested
    if max_samples is not None:
        print(f"Subsampling NN data to {max_samples} samples.")
        train_ds = Subset(train_ds, range(min(len(train_ds), max_samples)))
        val_ds = Subset(val_ds, range(min(len(val_ds), max_samples)))

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=NN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in some envs
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=NN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Determine Metadata Dimension from the dataset
    # Access the first item to check metadata shape
    sample_item = (
        train_ds[0]
        if not isinstance(train_ds, Subset)
        else train_ds.dataset[train_ds.indices[0]]
    )
    metadata_dim = sample_item["metadata"].shape[0]
    print(f"Detected Metadata Dimension: {metadata_dim}")

    # Initialize and Train
    model = PizzaNeuralNet(metadata_dim=metadata_dim)
    val_auc = model.train(train_loader, val_loader)

    # Save Model
    model.save()

    return model, val_auc


def evaluate_ensemble(rf_model, nn_model, load_cached_data=True, max_samples=None):
    """
    Evaluates the ensemble on the validation set.
    """
    print("\n=== Evaluating Ensemble ===")

    # 1. RF Predictions
    X_val_rf, y_val_rf = get_rf_dataset(split="val", load_cached_data=load_cached_data)
    if max_samples is not None:
        X_val_rf = X_val_rf.iloc[:max_samples]
        y_val_rf = y_val_rf.iloc[:max_samples]

    rf_probs = rf_model.predict_proba(X_val_rf)

    # 2. NN Predictions
    val_ds_nn = get_nn_dataset(split="val", load_cached_data=load_cached_data)
    if max_samples is not None:
        val_ds_nn = Subset(val_ds_nn, range(min(len(val_ds_nn), max_samples)))

    val_loader_nn = DataLoader(
        val_ds_nn, batch_size=NN_PARAMS["batch_size"], shuffle=False
    )
    nn_probs = nn_model.predict_proba(val_loader_nn)

    # Ensure alignment (sanity check)
    if len(rf_probs) != len(nn_probs):
        print(
            f"Warning: Length mismatch in validation predictions. RF: {len(rf_probs)}, NN: {len(nn_probs)}"
        )
        min_len = min(len(rf_probs), len(nn_probs))
        rf_probs = rf_probs[:min_len]
        nn_probs = nn_probs[:min_len]
        y_val_rf = y_val_rf[:min_len]

    # 3. Weighted Average
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_nn = ENSEMBLE_WEIGHTS["nn"]

    ensemble_probs = (w_rf * rf_probs) + (w_nn * nn_probs)

    # 4. Score
    ensemble_auc = roc_auc_score(y_val_rf, ensemble_probs)
    print(f"Ensemble Validation ROC AUC: {ensemble_auc}")

    return ensemble_auc


def generate_submission(rf_model, nn_model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\n=== Generating Submission ===")

    # 1. RF Test Predictions
    X_test_rf, _ = get_rf_dataset(split="test", load_cached_data=load_cached_data)
    rf_probs = rf_model.predict_proba(X_test_rf)

    # 2. NN Test Predictions
    test_ds_nn = get_nn_dataset(split="test", load_cached_data=load_cached_data)
    test_loader_nn = DataLoader(
        test_ds_nn, batch_size=NN_PARAMS["batch_size"], shuffle=False
    )
    nn_probs = nn_model.predict_proba(test_loader_nn)

    # 3. Ensemble
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_nn = ENSEMBLE_WEIGHTS["nn"]
    ensemble_probs = (w_rf * rf_probs) + (w_nn * nn_probs)

    # 4. Load Request IDs
    # We read the raw test metadata to ensure we have the correct IDs in order
    test_metadata_df = pd.read_csv(TEST_PATH)
    request_ids = test_metadata_df["request_id"].tolist()

    if len(request_ids) != len(ensemble_probs):
        raise ValueError(
            f"Mismatch between ID count ({len(request_ids)}) and prediction count ({len(ensemble_probs)})"
        )

    # 5. Save
    save_submission(request_ids, ensemble_probs, SUBMISSION_FILE)
    print(f"Submission saved to {SUBMISSION_FILE}")


def run_pipeline(load_cached_data=True, max_samples=None, epochs=None):
    """
    Runs the full training and submission pipeline.
    """
    # Set global seed for reproducibility
    set_seed(RANDOM_SEED)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Train Models
    rf_model, rf_auc = train_rf_model(
        load_cached_data=load_cached_data, max_samples=max_samples
    )
    nn_model, nn_auc = train_nn_model(
        load_cached_data=load_cached_data, max_samples=max_samples, epochs=epochs
    )

    # Evaluate Ensemble
    evaluate_ensemble(
        rf_model, nn_model, load_cached_data=load_cached_data, max_samples=max_samples
    )

    # Generate Submission (only if not debugging with small samples, or if explicitly desired)
    # We always generate submission as per requirements, unless max_samples implies a dev run where test set might not match
    if max_samples is None:
        generate_submission(rf_model, nn_model, load_cached_data=load_cached_data)
    else:
        print("Skipping submission generation due to subsampling (debug mode).")
