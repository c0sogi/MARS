import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import setup_logging, print_metric

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def create_pipeline(dino_dim, conv_dim, tab_dim):
    """
    Constructs the Selective Feature Topology pipeline.

    Args:
        dino_dim (int): Number of DINOv2 features.
        conv_dim (int): Number of ConvNeXt features.
        tab_dim (int): Number of tabular features.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Define column slices based on feature order in concatenation
    # Order: [DINO, CONV, TAB]
    dino_indices = list(range(0, dino_dim))
    conv_indices = list(range(dino_dim, dino_dim + conv_dim))
    tab_indices = list(range(dino_dim + conv_dim, dino_dim + conv_dim + tab_dim))

    # Selective Feature Topology
    # 1. Visual Streams: Linear PCA (Preserve Linearity)
    # 2. Tabular Stream: Gaussian Quantile Transform (Enforce Normality)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "pca_dino",
                PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                dino_indices,
            ),
            (
                "pca_conv",
                PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                conv_indices,
            ),
            (
                "trans_tab",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
                tab_indices,
            ),
        ],
        n_jobs=None,  # Run in main process to avoid overhead
    )

    # Classifier: LDA with Ledoit-Wolf Shrinkage
    clf = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    return pipeline


def train_models(data_dict, load_cached_models=True):
    """
    Trains the ensemble using Stratified K-Fold Cross-Validation.

    Args:
        data_dict (dict): Dictionary containing 'dino', 'conv', 'tab', 'y', 'ids'.
        load_cached_models (bool): If True, attempts to load trained pipelines from disk.

    Returns:
        tuple: (pipelines, label_encoder, oof_preds, oof_targets)
    """
    logger.info("Preparing data for training...")

    # Unpack data
    X_dino = data_dict["dino"]
    X_conv = data_dict["conv"]
    X_tab = data_dict["tab"]
    y_raw = data_dict["y"]
    ids = data_dict["ids"]

    # Validate alignment
    n_samples = len(y_raw)
    assert len(X_dino) == n_samples
    assert len(X_conv) == n_samples
    assert len(X_tab) == n_samples

    # Concatenate features: [DINO, CONV, TAB]
    X = np.hstack([X_dino, X_conv, X_tab])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Save LabelEncoder for inference
    le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
    if not os.path.exists(le_path) or not load_cached_models:
        joblib.dump(le, le_path)
    else:
        le = joblib.load(le_path)

    # Dimensions for pipeline construction
    dino_dim = X_dino.shape[1]
    conv_dim = X_conv.shape[1]
    tab_dim = X_tab.shape[1]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    pipelines = []
    oof_probs = np.zeros((n_samples, len(le.classes_)))
    oof_targets = np.zeros(n_samples)  # To store shuffled targets aligned with OOF

    # Check if all models exist for caching
    all_models_exist = True
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            all_models_exist = False
            break

    if load_cached_models and all_models_exist:
        logger.info("Loading cached models from disk...")
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")
            pipelines.append(joblib.load(model_path))

        # We cannot easily reconstruct OOF scores without re-running splitting logic,
        # but for inference purposes, we just need the pipelines.
        # We will return dummy OOFs if loading from cache to save time.
        return pipelines, le, None, None

    logger.info(
        f"Starting training on {n_samples} samples with {Config.N_FOLDS} folds..."
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pkl")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create new pipeline instance
        pipeline = create_pipeline(dino_dim, conv_dim, tab_dim)

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = pipeline.predict_proba(X_val)

        # Clip probabilities to avoid log loss extremes
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

        score = log_loss(y_val, y_pred_proba)
        fold_scores.append(score)

        print_metric(f"Fold {fold} Log Loss", score)

        # Store OOF
        oof_probs[val_idx] = y_pred_proba
        oof_targets[val_idx] = y_val

        # Save model
        joblib.dump(pipeline, model_path)
        pipelines.append(pipeline)

    avg_score = np.mean(fold_scores)
    print_metric("Average CV Log Loss", avg_score)

    return pipelines, le, oof_probs, oof_targets


def predict_and_submit(test_data_dict, pipelines, label_encoder):
    """
    Generates predictions for the test set and creates the submission file.

    Args:
        test_data_dict (dict): Dictionary containing 'dino', 'conv', 'tab', 'ids'.
        pipelines (list): List of trained sklearn pipelines.
        label_encoder (LabelEncoder): Fitted label encoder.

    Returns:
        None
    """
    logger.info("Generating predictions for test set...")

    X_dino = test_data_dict["dino"]
    X_conv = test_data_dict["conv"]
    X_tab = test_data_dict["tab"]
    ids = test_data_dict["ids"]

    # Concatenate features
    X_test = np.hstack([X_dino, X_conv, X_tab])

    # Accumulate probabilities
    avg_probs = np.zeros((len(X_test), len(label_encoder.classes_)))

    for i, pipeline in enumerate(pipelines):
        probs = pipeline.predict_proba(X_test)
        avg_probs += probs

    # Average over folds
    avg_probs /= len(pipelines)

    # Clip probabilities (Metric requirement)
    avg_probs = np.clip(avg_probs, 1e-15, 1 - 1e-15)

    # Create Submission DataFrame
    # Columns must be 'id' followed by species names
    submission_df = pd.DataFrame(avg_probs, columns=label_encoder.classes_)
    submission_df.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {submission_df.shape}")
