import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library import config, utils, data_manager, pipeline_builder

# Set up logging
logger = utils.setup_logger(
    "execution_engine", os.path.join(config.WORKING_DIR, "execution.log")
)


def train_ensemble(load_cached_data=True):
    """
    Executes the Stratified K-Fold Cross-Validation training loop.
    Merges train and val metadata to maximize training data.
    Trains 10 LDA pipelines on densified data and saves them.
    """
    logger.info("Starting ensemble training...")

    # 1. Load Data
    dm = data_manager.LeafDataManager()

    # Load train set for CV
    train_data = dm.get_dataset("train", load_cached_data=load_cached_data)

    # Features: [DINO, CONV, TABULAR]
    dino_all = train_data["dino"]
    conv_all = train_data["conv"]
    tab_all = train_data["tabular"]
    ids_all = train_data["ids"]
    y_all = train_data["y"]

    # Construct full feature matrix X
    # Order must match pipeline_builder expectations: DINO | CONV | TABULAR
    X_all = np.hstack([dino_all, conv_all, tab_all])

    # Dimensions for pipeline builder
    dino_dim = dino_all.shape[1]
    conv_dim = conv_all.shape[1]
    tab_dim = tab_all.shape[1]

    logger.info(f"Combined Data Shape: {X_all.shape}")
    logger.info(f"Feature Dims: DINO={dino_dim}, CONV={conv_dim}, TAB={tab_dim}")

    # 2. Prepare for Stratified K-Fold
    # We must split based on Unique IDs to avoid leakage (since we have 3 centroids per ID)
    # The data is ordered as [ID1_A, ID1_B, ID1_C, ID2_A, ...], so every 3 rows belong to one ID.

    # Extract unique IDs and their labels
    # Since data is structured in blocks of 3, we can just take every 3rd element
    unique_ids = ids_all[::3]
    unique_y = y_all[::3]

    # Verify integrity
    assert len(unique_ids) * 3 == len(
        ids_all
    ), "Data alignment error: rows not divisible by 3"

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=config.SHUFFLE_FOLDS, random_state=config.SEED
    )

    # Directory for models
    models_dir = os.path.join(config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    fold_scores = []

    # 3. Training Loop
    for fold, (train_idx_unique, val_idx_unique) in enumerate(
        skf.split(unique_ids, unique_y)
    ):
        logger.info(f"--- Fold {fold} ---")

        # Map unique indices back to densified indices
        # If index i is selected, then indices 3*i, 3*i+1, 3*i+2 are selected
        train_indices_densified = []
        for idx in train_idx_unique:
            base = idx * 3
            train_indices_densified.extend([base, base + 1, base + 2])

        val_indices_densified = []
        for idx in val_idx_unique:
            base = idx * 3
            val_indices_densified.extend([base, base + 1, base + 2])

        train_indices_densified = np.array(train_indices_densified)
        val_indices_densified = np.array(val_indices_densified)

        # Split Data
        X_train = X_all[train_indices_densified]
        y_train = y_all[train_indices_densified]
        X_val = X_all[val_indices_densified]
        y_val = y_all[val_indices_densified]

        # Build Pipeline
        pipeline = pipeline_builder.build_selective_pipeline(
            dino_dim=dino_dim, conv_dim=conv_dim, tab_dim=tab_dim
        )

        # Fit
        pipeline.fit(X_train, y_train)

        # Evaluate (on densified validation set)
        # Note: We evaluate on all centroids of validation data
        probs = pipeline.predict_proba(X_val)

        # Clip probabilities for stability
        probs = utils.clip_probabilities(probs)

        # Calculate Metric
        # We need to encode y_val to integers matching the classes in pipeline
        classes = pipeline.classes_
        class_map = {c: i for i, c in enumerate(classes)}
        y_val_indices = np.array([class_map[label] for label in y_val])

        score = log_loss(y_val_indices, probs, labels=list(range(len(classes))))
        logger.info(f"Fold {fold} Log Loss: {score}")
        fold_scores.append(score)

        # Save Model
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        utils.save_pickle(pipeline, model_path)

        # Save Classes (only need to do this once, but doing it every fold is safe)
        utils.save_pickle(classes, os.path.join(models_dir, "classes.pkl"))

    avg_score = np.mean(fold_scores)
    logger.info(f"Average CV Log Loss: {avg_score}")


def predict_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the trained ensemble.
    Performs double averaging:
    1. Average across 3 orthogonal centroids per image.
    2. Average across 10 ensemble models.
    """
    logger.info("Starting submission generation...")

    # 1. Load Test Data
    dm = data_manager.LeafDataManager()
    test_data = dm.get_dataset("test", load_cached_data=load_cached_data)

    # Construct X_test
    X_test = np.hstack([test_data["dino"], test_data["conv"], test_data["tabular"]])
    ids_test_densified = test_data["ids"]

    # Unique IDs for submission (every 3rd element)
    unique_test_ids = ids_test_densified[::3]

    # 2. Load Metadata
    models_dir = os.path.join(config.WORKING_DIR, "models")
    classes_path = os.path.join(models_dir, "classes.pkl")

    if not os.path.exists(classes_path):
        raise FileNotFoundError("Classes file not found. Run training first.")

    classes = utils.load_pickle(classes_path)
    n_classes = len(classes)
    n_samples = len(unique_test_ids)

    # Accumulator for ensemble probabilities
    ensemble_probs = np.zeros((n_samples, n_classes))

    # 3. Inference Loop
    for fold in range(config.N_FOLDS):
        model_path = os.path.join(models_dir, f"pipeline_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold} not found. Skipping.")
            continue

        logger.info(f"Predicting with model fold {fold}...")
        pipeline = utils.load_pickle(model_path)

        # Predict on densified test set (3N samples)
        # Shape: (3 * N_samples, N_classes)
        probs_densified = pipeline.predict_proba(X_test)

        # Reshape to (N_samples, 3, N_classes)
        # This groups the 3 centroids for each image
        probs_reshaped = probs_densified.reshape(n_samples, 3, n_classes)

        # Average across centroids (Manifold Densification Aggregation)
        # Shape: (N_samples, N_classes)
        probs_aggregated = np.mean(probs_reshaped, axis=1)

        # Add to ensemble accumulator
        ensemble_probs += probs_aggregated

    # 4. Final Aggregation
    # Average across models
    final_probs = ensemble_probs / config.N_FOLDS

    # Clip probabilities
    final_probs = utils.clip_probabilities(final_probs)

    # 5. Format Submission
    logger.info("Formatting submission...")

    # Load sample submission to ensure correct column order
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Create DataFrame
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", unique_test_ids)

    # Reorder columns to match sample submission
    # Ensure 'id' is first, then the species columns in correct order
    target_cols = [c for c in sample_sub.columns if c != "id"]

    # Check if we have all columns
    missing_cols = set(target_cols) - set(submission_df.columns)
    if missing_cols:
        logger.warning(
            f"Missing columns in prediction: {missing_cols}. Filling with epsilon."
        )
        for c in missing_cols:
            submission_df[c] = config.PROB_CLIP_MIN

    # Select and reorder
    final_cols = ["id"] + target_cols
    submission_df = submission_df[final_cols]

    # Save
    save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
