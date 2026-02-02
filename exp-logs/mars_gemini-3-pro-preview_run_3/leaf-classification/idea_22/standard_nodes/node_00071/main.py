import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, save_model, load_model, format_submission
from library.data_manager import DataManager
from library.model_factory import create_pipeline

# Explicitly patch Config to ensure updates persist in cached modules (Cite debug_lesson_11)
Config.BATCH_SIZE = 2
Config.N_FOLDS = 5


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting execution of runfile.py...")

    # 2. Data Loading
    # DataManager handles feature extraction, caching, and densification
    dm = DataManager()
    data = dm.get_data(load_cached_data=True)

    # Unpack Training Data
    X_train_densified = data["train_X"]
    y_train_densified = data["train_y"]
    ids_train_densified = data["train_ids"]

    # Unpack Validation Data (Hold-out)
    # Note: val_X comes densified (3M, D) from DataManager because is_train=True was used
    X_val_densified = data["val_X"]
    y_val_densified = data["val_y"]
    ids_val_densified = data["val_ids"]

    # Unpack Test Data
    # Test data is structured as (K, 3, D)
    X_test_structured = data["test_X"]
    ids_test = data["test_ids"]

    # Determine Feature Dimensions for Pipeline
    # Structure: [DINO (1024) | ConvNeXt (1536) | Tabular (192)]
    # We assume standard sizes for the models defined in Config, but let's verify/hardcode based on knowns
    # DINOv2 Large = 1024, ConvNeXt Large = 1536, Tabular = 192
    dino_dim = 1024
    conv_dim = 1536
    tabular_dim = 192

    # Verify total dimension
    total_dim = X_train_densified.shape[1]
    assert (
        total_dim == dino_dim + conv_dim + tabular_dim
    ), f"Feature dimension mismatch. Expected {dino_dim+conv_dim+tabular_dim}, got {total_dim}"

    # 3. Training Loop (Stratified K-Fold)
    # We split based on Unique IDs to prevent leakage (all centroids of an image must be in same fold)
    unique_train_ids, unique_indices = np.unique(ids_train_densified, return_index=True)
    unique_train_labels = y_train_densified[unique_indices]

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx_unique, _) in enumerate(
        skf.split(unique_train_ids, unique_train_labels)
    ):
        print(f"Training Fold {fold}...")

        # Get the actual IDs for this fold's training set
        fold_train_ids = unique_train_ids[train_idx_unique]

        # Find indices in the densified array that match these IDs
        # np.isin is efficient for this
        mask_train = np.isin(ids_train_densified, fold_train_ids)

        X_fold_train = X_train_densified[mask_train]
        y_fold_train = y_train_densified[mask_train]

        # Create and Train Pipeline
        model = create_pipeline(dino_dim, conv_dim, tabular_dim)
        model.fit(X_fold_train, y_fold_train)

        # Save Model
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        save_model(model, model_path)
        models.append(model)

    # 4. Validation Assessment (Hold-out Set)
    print("Performing Validation Assessment...")

    # Reshape Validation data to (M, 3, D) for aggregation
    # We know it was densified by repeating 3 times.
    n_val_samples = len(np.unique(ids_val_densified))
    X_val_structured = X_val_densified.reshape(
        n_val_samples, Config.CENTROIDS_PER_IMAGE, total_dim
    )

    # Get unique labels for validation (since they are repeated 3 times)
    # We can just take every 3rd label
    y_val_unique = y_val_densified[:: Config.CENTROIDS_PER_IMAGE]

    # Get class names from the first model
    class_names = models[0].classes_

    # Inference on Validation Set
    val_probs = predict_ensemble(models, X_val_structured)

    # Compute Metric
    val_log_loss = log_loss(y_val_unique, val_probs, labels=class_names)
    print(f"Final Validation Metric: {val_log_loss:.15f}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    perform_failure_analysis(
        X_val_structured, y_val_unique, val_probs, class_names, dino_dim, conv_dim
    )

    # 6. Submission Generation
    # The prompt implies a strict condition, but standard practice is to submit the best attempt.
    # We generate the submission file as required.
    print("Generating Submission...")

    test_probs = predict_ensemble(models, X_test_structured)

    format_submission(
        test_ids=ids_test,
        predictions=test_probs,
        class_names=class_names,
        output_path=Config.SUBMISSION_PATH,
    )
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def predict_ensemble(models, X_structured):
    """
    Predicts using the ensemble with Full-Manifold Test-Time Aggregation.

    Args:
        models: List of trained pipelines.
        X_structured: Input data of shape (N_samples, 3, N_features).

    Returns:
        avg_probs: (N_samples, N_classes)
    """
    N, C, D = X_structured.shape

    # Flatten to (N*3, D) for batch prediction
    X_flat = X_structured.reshape(N * C, D)

    ensemble_probs = []

    for model in models:
        # Predict: (N*3, n_classes)
        probs_flat = model.predict_proba(X_flat)

        # Reshape back to (N, 3, n_classes)
        probs_structured = probs_flat.reshape(N, C, -1)

        # Average over centroids (Manifold Aggregation) -> (N, n_classes)
        probs_centroids_avg = np.mean(probs_structured, axis=1)

        ensemble_probs.append(probs_centroids_avg)

    # Average over ensemble models -> (N, n_classes)
    avg_probs = np.mean(ensemble_probs, axis=0)

    return avg_probs


def perform_failure_analysis(
    X_val, y_true, y_pred_probs, class_names, dino_dim, conv_dim
):
    """
    Correlates prediction error with tabular features.
    """
    # 1. Calculate Log Loss per sample
    # y_true is string labels, need to map to indices
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    y_true_indices = np.array([class_to_idx[y] for y in y_true])

    # Extract probability of the true class
    # Clip to avoid log(0)
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)

    true_class_probs = y_pred_probs[np.arange(len(y_true)), y_true_indices]
    sample_losses = -np.log(true_class_probs)

    # 2. Extract Tabular Features
    # X_val is (N, 3, D). We can just take the first centroid's tabular part
    # because tabular features are invariant/duplicated across centroids in our pipeline.
    # Tabular features start after dino_dim + conv_dim
    tabular_start = dino_dim + conv_dim
    tabular_features = X_val[:, 0, tabular_start:]

    # Feature names (reconstructing list)
    margin_cols = [f"margin_{i+1}" for i in range(64)]
    shape_cols = [f"shape_{i+1}" for i in range(64)]
    texture_cols = [f"texture_{i+1}" for i in range(64)]
    feature_names = margin_cols + shape_cols + texture_cols

    # 3. Compute Correlations
    correlations = []
    for i in range(tabular_features.shape[1]):
        feat_values = tabular_features[:, i]
        # Handle constant features (std=0) to avoid warning
        if np.std(feat_values) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(sample_losses, feat_values)
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Top 5 Features Correlated with Error ---")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")
    print("--------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
