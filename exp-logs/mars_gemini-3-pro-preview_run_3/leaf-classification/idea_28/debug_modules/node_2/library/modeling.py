import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from library.config import Config
from library.utils import get_logger, calculate_metric, save_submission

# Initialize logger
logger = get_logger("modeling")


def create_selective_pipeline(dino_dim=1024, conv_dim=1536, tab_dim=192):
    """
    Constructs the Selective-Topology pipeline.

    Args:
        dino_dim (int): Number of DINOv2 features (start of vector).
        conv_dim (int): Number of ConvNeXt features (middle of vector).
        tab_dim (int): Number of tabular features (end of vector).

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Calculate index ranges for slicing the concatenated feature vector
    # Structure: [DINO features | ConvNeXt features | Tabular features]
    dino_end = dino_dim
    conv_end = dino_dim + conv_dim
    tab_end = conv_end + tab_dim

    # Define indices
    dino_indices = list(range(0, dino_end))
    conv_indices = list(range(dino_end, conv_end))
    tab_indices = list(range(conv_end, tab_end))

    # 1. Visual Stream Transformations (Independent Subspace Reduction)
    # We strictly preserve linear topology (PCA) and avoid Gaussianization for deep features.
    visual_transformer = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

    # 2. Tabular Stream Transformation (Gaussianization)
    # We force handcrafted histograms into a Gaussian distribution for LDA.
    tabular_transformer = QuantileTransformer(
        output_distribution=Config.TABULAR_TRANSFORMER_OUTPUT, random_state=Config.SEED
    )

    # 3. Column Transformer
    # Applies specific transforms to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", visual_transformer, dino_indices),
            ("conv_pca", visual_transformer, conv_indices),
            ("tab_qt", tabular_transformer, tab_indices),
        ],
        n_jobs=1,
    )

    # 4. Classifier
    # LDA with Ledoit-Wolf shrinkage (ideal for HDLSS)
    classifier = LinearDiscriminantAnalysis(
        solver=Config.CLASSIFIER_SOLVER, shrinkage=Config.CLASSIFIER_SHRINKAGE
    )

    # Assemble Pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline


def aggregate_predictions(ids, probs):
    """
    Aggregates probabilities by Image ID (averaging across centroids).

    Args:
        ids (np.array): Array of image IDs corresponding to rows of probs.
        probs (np.array): Matrix of probabilities (N_samples, N_classes).

    Returns:
        tuple: (unique_ids, aggregated_probs)
    """
    df = pd.DataFrame(probs)
    df["id"] = ids

    # Group by ID and compute mean
    grouped = df.groupby("id").mean()

    unique_ids = grouped.index.values
    agg_probs = grouped.values

    return unique_ids, agg_probs


def run_cross_validation(train_data, test_data):
    """
    Executes the Stratified K-Fold training and inference loop.

    Args:
        train_data (dict): Dictionary containing densified training arrays
                           (X_dino, X_conv, X_tab, ids, y).
        test_data (dict): Dictionary containing densified test arrays
                          (X_dino, X_conv, X_tab, ids).
    """
    logger.info("Preparing data for Cross-Validation...")

    # 1. Concatenate Features
    # Train
    X_train_full = np.hstack(
        [train_data["X_dino"], train_data["X_conv"], train_data["X_tab"]]
    )
    y_train_full = train_data["y"]
    ids_train_full = train_data["ids"]

    # Test
    X_test_full = np.hstack(
        [test_data["X_dino"], test_data["X_conv"], test_data["X_tab"]]
    )
    ids_test_full = test_data["ids"]

    # Determine dimensions for the pipeline
    dino_dim = train_data["X_dino"].shape[1]
    conv_dim = train_data["X_conv"].shape[1]
    tab_dim = train_data["X_tab"].shape[1]

    # 2. Setup Stratified Splitting on UNIQUE IDs
    # We must split based on original images, not densified samples, to prevent leakage.
    # Create a DataFrame of unique (id, label) pairs
    df_train_map = (
        pd.DataFrame({"id": ids_train_full, "label": y_train_full})
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )

    unique_ids = df_train_map["id"].values
    unique_labels = df_train_map["label"].values

    kf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF (Out of Fold) and Test predictions
    # We accumulate test probabilities from each fold
    test_probs_sum = None
    oof_metrics = []

    logger.info(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx_unique, val_idx_unique) in enumerate(
        kf.split(unique_ids, unique_labels)
    ):
        # Identify IDs for this fold
        fold_train_ids = unique_ids[train_idx_unique]
        fold_val_ids = unique_ids[val_idx_unique]

        # Create masks for the densified dataset
        train_mask = np.isin(ids_train_full, fold_train_ids)
        val_mask = np.isin(ids_train_full, fold_val_ids)

        # Split densified data
        X_fold_train = X_train_full[train_mask]
        y_fold_train = y_train_full[train_mask]

        X_fold_val = X_train_full[val_mask]
        y_fold_val = y_train_full[val_mask]
        ids_fold_val = ids_train_full[val_mask]

        # Create and Fit Pipeline
        pipeline = create_selective_pipeline(dino_dim, conv_dim, tab_dim)
        pipeline.fit(X_fold_train, y_fold_train)

        # --- Validation ---
        # Predict on densified validation set
        val_probs_densified = pipeline.predict_proba(X_fold_val)

        # Aggregate predictions by ID (Manifold Aggregation)
        val_ids_agg, val_probs_agg = aggregate_predictions(
            ids_fold_val, val_probs_densified
        )

        # Get true labels for aggregated IDs
        # We can map back using the df_train_map
        val_labels_map = df_train_map.set_index("id")["label"]
        val_labels_agg = val_labels_map.loc[val_ids_agg].values

        # Calculate Metric
        score = calculate_metric(
            val_labels_agg, val_probs_agg, labels=pipeline.classes_
        )
        oof_metrics.append(score)

        logger.info(f"Fold {fold+1}/{Config.N_FOLDS} - Log Loss: {score:.6f}")

        # --- Test Inference ---
        test_probs = pipeline.predict_proba(X_test_full)

        if test_probs_sum is None:
            test_probs_sum = test_probs
            classes = pipeline.classes_
        else:
            test_probs_sum += test_probs

    # 3. Finalize Results
    mean_oof_score = np.mean(oof_metrics)
    std_oof_score = np.std(oof_metrics)
    logger.info(
        f"CV Complete. Mean Log Loss: {mean_oof_score:.6f} (+/- {std_oof_score:.6f})"
    )

    # Average test probabilities across folds
    test_probs_avg = test_probs_sum / Config.N_FOLDS

    # Aggregate test probabilities by ID (Manifold Aggregation)
    # This combines the 3 orthogonal centroids per test image
    final_ids, final_probs = aggregate_predictions(ids_test_full, test_probs_avg)

    # 4. Save Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(final_ids, final_probs, classes, filename=Config.SUBMISSION_PATH)
    logger.info("Submission saved successfully.")
