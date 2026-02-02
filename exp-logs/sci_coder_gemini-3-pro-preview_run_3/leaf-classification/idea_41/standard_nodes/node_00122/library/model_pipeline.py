import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score

from library.config import Config
from library.utils import setup_logger

logger = setup_logger("model_pipeline")


def concat_features(dataset):
    """
    Concatenates the disparate feature sets in LeafDataset into a single matrix.

    Args:
        dataset: LeafDataset instance containing dino_features, conv_features, tab_features.

    Returns:
        X (np.ndarray): Concatenated feature matrix of shape (N, D_total).
        dims (tuple): Dimensions of each block (dino_dim, conv_dim, tab_dim).
    """
    dino = dataset.dino_features
    conv = dataset.conv_features
    tab = dataset.tab_features

    # Horizontal stack: [DINO | CONV | TAB]
    X = np.hstack([dino, conv, tab])
    dims = (dino.shape[1], conv.shape[1], tab.shape[1])

    return X, dims


def create_classifier(dims):
    """
    Constructs the Stratified Selective-Topology Orthogonal Manifold-Densified LDA pipeline.

    Args:
        dims (tuple): A tuple containing (dino_dim, conv_dim, tab_dim).

    Returns:
        sklearn.pipeline.Pipeline: The configured pipeline.
    """
    dino_dim, conv_dim, tab_dim = dims

    # Calculate indices for slicing
    idx_dino_end = dino_dim
    idx_conv_end = dino_dim + conv_dim

    # Generate column indices for ColumnTransformer
    dino_indices = list(range(0, idx_dino_end))
    conv_indices = list(range(idx_dino_end, idx_conv_end))
    tab_indices = list(range(idx_conv_end, idx_conv_end + tab_dim))

    # 1. Independent Subspace Reduction & Tabular Gaussianization
    # We use 'full' svd_solver for exact variance retention with float n_components
    preprocessor = ColumnTransformer(
        transformers=[
            # Global Geometry Stream: PCA (Linear Topology Preservation)
            (
                "dino_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                dino_indices,
            ),
            # Local Texture Stream: PCA (Linear Topology Preservation)
            (
                "conv_pca",
                PCA(
                    n_components=Config.PCA_VARIANCE,
                    svd_solver="full",
                    random_state=Config.SEED,
                ),
                conv_indices,
            ),
            # Tabular Stream: Quantile Transformer (Gaussianization)
            (
                "tab_qt",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
                tab_indices,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # 2. Global Variance Alignment & Classifier
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "global_scaler",
                StandardScaler(),
            ),  # Aligns variance for Ledoit-Wolf shrinkage
            (
                "classifier",
                LinearDiscriminantAnalysis(
                    solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                ),
            ),
        ]
    )

    return pipeline


def train_fold(dataset_train, dataset_val, fold_idx, save_dir=Config.MODELS_DIR):
    """
    Trains the pipeline on the densified training set and evaluates on the validation set.
    Saves the pipeline to disk.

    Args:
        dataset_train: LeafDataset for training (densified).
        dataset_val: LeafDataset for validation (densified).
        fold_idx (int): Current fold index.
        save_dir (str): Directory to save the model.

    Returns:
        pipeline: The fitted pipeline.
        metrics (dict): Dictionary containing 'loss' and 'accuracy'.
    """
    # Prepare Data
    X_train, dims = concat_features(dataset_train)
    y_train = dataset_train.labels

    X_val, _ = concat_features(dataset_val)
    y_val = dataset_val.labels

    logger.info(f"Fold {fold_idx}: Input Shape {X_train.shape}, Dims {dims}")

    # Create Pipeline
    pipeline = create_classifier(dims)

    # Fit
    logger.info(f"Fold {fold_idx}: Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    # Note: This evaluation is on the densified validation set (individual centroids),
    # which serves as a strict proxy for model health before aggregation.
    train_acc = pipeline.score(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)

    # Calculate Log Loss
    y_val_prob = pipeline.predict_proba(X_val)
    val_log_loss = log_loss(y_val, y_val_prob, labels=pipeline.classes_)

    logger.info(f"Fold {fold_idx} Results:")
    logger.info(f"  Train Accuracy: {train_acc:.6f}")
    logger.info(f"  Val Accuracy:   {val_acc:.6f}")
    logger.info(f"  Val Log Loss:   {val_log_loss:.6f}")

    # Save Pipeline
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"pipeline_fold_{fold_idx}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Fold {fold_idx}: Pipeline saved to {save_path}")

    return pipeline, {"loss": val_log_loss, "accuracy": val_acc}


def predict_and_aggregate(pipeline, dataset):
    """
    Performs Full-Manifold Test-Time Aggregation.
    Predicts on all 3 centroids for each image and averages the probabilities.

    Args:
        pipeline: Fitted sklearn pipeline.
        dataset: LeafDataset (densified, 3 centroids per image).

    Returns:
        unique_ids (np.ndarray): Array of unique image IDs.
        avg_probs (np.ndarray): Averaged probability matrix (N_images, N_classes).
    """
    X, _ = concat_features(dataset)

    # Predict on all densified samples (N * 3)
    # Shape: (N * 3, n_classes)
    probs_densified = pipeline.predict_proba(X)

    # Reshape to (N, 3, n_classes) to group by image
    # We assume the dataset is structured as [ID1_A, ID1_B, ID1_C, ID2_A, ...]
    # which is guaranteed by DataProcessor.densify_data
    n_samples = probs_densified.shape[0]
    n_classes = probs_densified.shape[1]
    n_images = n_samples // 3

    probs_reshaped = probs_densified.reshape(n_images, 3, n_classes)

    # Average across the 3 centroids (axis 1)
    avg_probs = np.mean(probs_reshaped, axis=1)

    # Extract unique IDs (every 3rd ID)
    unique_ids = dataset.ids[::3]

    return unique_ids, avg_probs


def generate_submission(models, test_dataset, output_path=None):
    """
    Generates the submission file by averaging predictions from an ensemble of models.

    Args:
        models (list): List of fitted pipelines.
        test_dataset: LeafDataset for testing.
        output_path (str): Path to save submission CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    logger.info(f"Generating submission with ensemble of {len(models)} models...")

    ensemble_probs = []
    unique_ids = None
    classes = models[0].classes_

    for i, model in enumerate(models):
        ids, probs = predict_and_aggregate(model, test_dataset)
        ensemble_probs.append(probs)

        if unique_ids is None:
            unique_ids = ids
        elif not np.array_equal(unique_ids, ids):
            raise ValueError(f"ID mismatch in model {i}")

    # Average across ensemble
    # Shape: (N_images, n_classes)
    final_probs = np.mean(ensemble_probs, axis=0)

    # Clip probabilities to avoid log loss extremes
    # range [10^-15, 1 - 10^-15]
    epsilon = 1e-15
    final_probs = np.clip(final_probs, epsilon, 1 - epsilon)

    # Create DataFrame
    df_submission = pd.DataFrame(final_probs, columns=classes)
    df_submission.insert(0, "id", unique_ids)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")

    return df_submission
