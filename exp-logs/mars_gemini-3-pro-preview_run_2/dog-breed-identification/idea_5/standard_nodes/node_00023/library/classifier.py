import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import library.config as config
import library.feature_engine as feature_engine

# Ensure reproducibility
np.random.seed(config.SEED)


def get_breed_list():
    """
    Helper to get the sorted list of breeds to map model outputs to columns.
    Matches the logic in library.dataset.get_class_mapping.
    """
    labels_path = os.path.join(config.INPUT_DIR, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    df = pd.read_csv(labels_path)
    unique_breeds = sorted(df["breed"].unique())
    return unique_breeds


def train_logreg(X_train, y_train, X_val, y_val, model_name):
    """
    Trains a Logistic Regression model with Cross-Validation and evaluates on validation set.
    Cite solution_lesson_node_00021: Prioritize Hyperparameter Tuning of Linear Probes.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        model_name (str): Name identifier for saving the model.

    Returns:
        clf: Trained LogisticRegressionCV model.
        val_probs: Probability predictions on validation set.
    """
    print(f"Training Logistic Regression CV for {model_name}...")

    # Initialize model with CV
    clf = LogisticRegressionCV(
        Cs=10,  # Grid of 10 values for C
        cv=5,  # 5-fold CV
        solver=config.LOGREG_SOLVER,
        multi_class=config.LOGREG_MULTI_CLASS,
        max_iter=config.LOGREG_MAX_ITER,
        random_state=config.SEED,
        verbose=0,
        n_jobs=-1,  # Use all available cores
        refit=True,  # Refit on full training set with best C
    )

    # Fit model
    clf.fit(X_train, y_train)
    print(f"  Best C found: {clf.C_[0]}")

    # Predict on validation
    val_probs_raw = clf.predict_proba(X_val)

    # Ensure output shape matches total classes (Cite debug_lesson_2)
    # In debug mode or small datasets, not all classes may be present in training.
    if val_probs_raw.shape[1] < config.NUM_CLASSES:
        val_probs = np.zeros(
            (val_probs_raw.shape[0], config.NUM_CLASSES), dtype=val_probs_raw.dtype
        )
        val_probs[:, clf.classes_.astype(int)] = val_probs_raw
    else:
        val_probs = val_probs_raw

    # Calculate metric
    # Cite debug_lesson_2: Filter validation samples to ensure they belong to known classes
    known_classes = set(clf.classes_)
    mask = np.array([y in known_classes for y in y_val])

    if mask.sum() > 0:
        # Use raw probs for log_loss calculation with specific labels to avoid dimension mismatch errors in metric
        loss = log_loss(y_val[mask], val_probs_raw[mask], labels=clf.classes_)
        print(f"  {model_name} Validation Log Loss: {loss}")
    else:
        print(f"  {model_name} Validation Log Loss: N/A (No overlapping classes)")

    # Save model
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    model_path = os.path.join(config.WORKING_DIR, f"{model_name}_logreg.joblib")
    joblib.dump(clf, model_path)
    print(f"  Model saved to {model_path}")

    return clf, val_probs


def generate_submission(model, debug=False, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained model.
        debug (bool): Debug flag.
        load_cached_data (bool): Cache flag.
    """
    print("Generating submission...")

    # 1. Extract Test Features
    X_test, _, ids = feature_engine.extract_features(
        dataset_key="test",
        model_name=config.MODEL_NAME,
        weights_name=config.MODEL_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 2. Predict
    probs_raw = model.predict_proba(X_test)

    # Cite debug_lesson_3: Enforce Global Output Schemas
    # Ensure probabilities cover all 120 classes, even if model only saw a subset
    if probs_raw.shape[1] < config.NUM_CLASSES:
        probs = np.zeros(
            (probs_raw.shape[0], config.NUM_CLASSES), dtype=probs_raw.dtype
        )
        probs[:, model.classes_.astype(int)] = probs_raw
    else:
        probs = probs_raw

    # 3. Create Submission DataFrame
    breeds = get_breed_list()

    # Check if number of classes matches
    if probs.shape[1] != len(breeds):
        print(
            f"Warning: Predicted classes ({probs.shape[1]}) != Total breeds ({len(breeds)})"
        )

    df_sub = pd.DataFrame(probs, columns=breeds)
    df_sub.insert(0, "id", ids)

    # 4. Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run_classifier_pipeline(debug=False, load_cached_data=True):
    """
    Orchestrates the feature loading, training, ensemble optimization, and submission.

    Args:
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, attempts to load features from disk.
    """
    print("Starting Classifier Pipeline...")

    # --- Stream A (ConvNeXt) ---
    X_train_a, y_train_a, _ = feature_engine.extract_features(
        dataset_key="train",
        model_name=config.MODEL_A_NAME,
        weights_name=config.MODEL_A_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )
    X_val_a, y_val_a, _ = feature_engine.extract_features(
        dataset_key="val",
        model_name=config.MODEL_A_NAME,
        weights_name=config.MODEL_A_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    model_a, probs_val_a = train_logreg(
        X_train_a, y_train_a, X_val_a, y_val_a, "Stream_A_ConvNeXt"
    )

    # --- Stream B (ViT) ---
    X_train_b, y_train_b, _ = feature_engine.extract_features(
        dataset_key="train",
        model_name=config.MODEL_B_NAME,
        weights_name=config.MODEL_B_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )
    X_val_b, y_val_b, _ = feature_engine.extract_features(
        dataset_key="val",
        model_name=config.MODEL_B_NAME,
        weights_name=config.MODEL_B_WEIGHTS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # Sanity check labels
    if not np.array_equal(y_val_a, y_val_b):
        raise ValueError("Mismatch in validation labels between streams.")

    model_b, probs_val_b = train_logreg(
        X_train_b, y_train_b, X_val_b, y_val_b, "Stream_B_ViT"
    )

    # --- Ensemble ---
    best_weight_a = optimize_ensemble_weights(
        probs_val_a, probs_val_b, y_val_a, labels=model_a.classes_
    )

    # --- Submission ---
    generate_submission(
        model_a, model_b, best_weight_a, debug=debug, load_cached_data=load_cached_data
    )
