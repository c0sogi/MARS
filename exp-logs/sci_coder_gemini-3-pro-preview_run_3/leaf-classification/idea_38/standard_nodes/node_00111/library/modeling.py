import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import setup_logger, seed_everything
from library.feature_extraction import extract_dataset
from library.densification import prepare_training_data, prepare_inference_data

logger = setup_logger("modeling")


def build_pipeline():
    """
    Constructs the Selective-Topology LDA pipeline.

    Architecture:
    1. Independent Subspace Reduction (Linear): PCA on Visual Streams
    2. Tabular Gaussianization (Non-Linear): QuantileTransformer on Tabular Features
    3. Global Alignment: StandardScaler
    4. Classifier: LDA with Ledoit-Wolf Shrinkage
    """
    # Define column slices based on feature extraction concatenation order
    # DINOv2 Large: 1024 dimensions (Indices 0-1024)
    # ConvNeXt Large: 1536 dimensions (Indices 1024-2560)
    # Tabular: 192 dimensions (Indices 2560-2752)

    idx_dino = slice(0, 1024)
    idx_conv = slice(1024, 2560)
    idx_tab = slice(2560, 2752)

    # 1. Feature-Specific Transformations
    preprocessor = ColumnTransformer(
        transformers=[
            # Visual Streams: Preserve Linear Topology with PCA
            (
                "dino_pca",
                PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                idx_dino,
            ),
            (
                "conv_pca",
                PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                idx_conv,
            ),
            # Tabular Stream: Enforce Gaussian Assumption
            (
                "tab_qt",
                QuantileTransformer(
                    output_distribution=Config.TABULAR_OUTPUT_DIST,
                    random_state=Config.SEED,
                ),
                idx_tab,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # 2. Pipeline Assembly
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "scaler",
                StandardScaler(),
            ),  # Global standardization for uniform shrinkage
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                ),
            ),
        ]
    )

    return pipeline


def train_model(load_cached_features=True, load_cached_densification=True):
    """
    Executes the training pipeline with Stratified K-Fold Cross Validation.
    Uses Convex-Hull Densification for training and Canonical Aggregation for validation.
    """
    seed_everything(Config.SEED)

    logger.info("Starting training process...")

    # 1. Load Raw Extracted Features
    img_feats, tab_feats, ids, labels = extract_dataset(
        "train", load_cached_data=load_cached_features
    )

    # 2. Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    fold_scores = []

    # 3. K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(img_feats, labels)):
        logger.info(f"--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # Split Data (Raw Views)
        X_img_train, X_img_val = img_feats[train_idx], img_feats[val_idx]
        X_tab_train, X_tab_val = tab_feats[train_idx], tab_feats[val_idx]
        y_train_raw, y_val_raw = labels[train_idx], labels[val_idx]
        ids_train, ids_val = ids[train_idx], ids[val_idx]

        # 4. Densification (Train Set) - Convex Hull (6x Expansion)
        # We use a unique cache suffix per fold to prevent collisions
        X_img_train_dense, X_tab_train_dense, y_train_dense, _ = prepare_training_data(
            X_img_train,
            X_tab_train,
            ids_train,
            y_train_raw,
            cache_suffix=f"fold_{fold}_train",
            load_cached_data=load_cached_densification,
        )

        # 5. Inference Prep (Validation Set) - Canonical Centroids (3x Expansion)
        X_img_val_can, X_tab_val_can, _, _ = prepare_inference_data(
            X_img_val,
            X_tab_val,
            ids_val,
            y_val_raw,
            cache_suffix=f"fold_{fold}_val",
            load_cached_data=load_cached_densification,
        )

        # 6. Concatenate Modalities for Pipeline
        # Shape: (N_samples, 2752)
        X_train = np.hstack([X_img_train_dense, X_tab_train_dense])
        X_val = np.hstack([X_img_val_can, X_tab_val_can])

        # 7. Build and Train Pipeline
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train_dense)

        # 8. Save Model and Classes
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        joblib.dump(pipeline, model_path)

        if fold == 0:
            joblib.dump(pipeline.classes_, os.path.join(models_dir, "classes.pkl"))

        # 9. Evaluate (Full-Manifold Aggregation)
        # Predict on all 3 centroids per validation image
        probs_expanded = pipeline.predict_proba(X_val)  # (N_val * 3, n_classes)

        # Aggregate: Mean over 3 centroids per image
        n_val = len(val_idx)
        n_classes = len(pipeline.classes_)

        # Reshape to (N_val, 3, n_classes) and mean over axis 1
        probs_reshaped = probs_expanded.reshape(n_val, 3, n_classes)
        probs_mean = probs_reshaped.mean(axis=1)  # (N_val, n_classes)

        # Calculate Log Loss
        score = log_loss(y_val_raw, probs_mean, labels=pipeline.classes_)
        fold_scores.append(score)

        logger.info(f"Fold {fold+1} Log Loss: {score:.10f}")

    avg_score = np.mean(fold_scores)
    logger.info(f"Average Cross-Validation Log Loss: {avg_score:.10f}")

    return fold_scores


def generate_submission(load_cached_features=True, load_cached_inference=True):
    """
    Generates the submission file by aggregating predictions from the trained ensemble.
    """
    seed_everything(Config.SEED)
    models_dir = os.path.join(Config.WORKING_DIR, "models")

    logger.info("Starting submission generation...")

    # 1. Load Test Data
    img_feats, tab_feats, ids, _ = extract_dataset(
        "test", load_cached_data=load_cached_features
    )

    # 2. Prepare Inference Data (Canonical Centroids 3x)
    X_img_test, X_tab_test, ids_expanded, _ = prepare_inference_data(
        img_feats,
        tab_feats,
        ids,
        labels=None,
        cache_suffix="test",
        load_cached_data=load_cached_inference,
    )

    X_test = np.hstack([X_img_test, X_tab_test])

    # 3. Load Classes
    classes_path = os.path.join(models_dir, "classes.pkl")
    if not os.path.exists(classes_path):
        raise FileNotFoundError("Classes file not found. Please train the model first.")

    classes = joblib.load(classes_path)

    # 4. Ensemble Prediction
    n_samples = len(ids)
    n_classes = len(classes)
    avg_probs = np.zeros((n_samples, n_classes))

    model_files = [f for f in os.listdir(models_dir) if f.startswith("pipeline_fold_")]
    if not model_files:
        raise FileNotFoundError("No trained models found in working directory.")

    logger.info(f"Aggregating predictions from {len(model_files)} models...")

    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        pipeline = joblib.load(path)

        # Predict on expanded test set (N*3)
        probs_expanded = pipeline.predict_proba(X_test)

        # Aggregate Centroids: (N, 3, C) -> (N, C)
        probs_reshaped = probs_expanded.reshape(n_samples, 3, n_classes)
        probs_mean = probs_reshaped.mean(axis=1)

        # Add to ensemble sum
        avg_probs += probs_mean

    # Average over folds
    avg_probs /= len(model_files)

    # 5. Create and Save Submission
    df_sub = pd.DataFrame(avg_probs, columns=classes)
    df_sub.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
