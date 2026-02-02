import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.base import clone

# Import provided library modules
from library.utils import ensure_float64, clipped_log_loss, save_submission, set_seed
from library.image_processing import process_dataset
from library.transformers import (
    MarginalBasis,
    RotationalBasis,
    RobustBasis,
    FactorizedInteractionProjector,
)

# Constants
METADATA_DIR = "./metadata"
SUBMISSION_PATH = "./submission/submission.csv"
SHRINKAGE_VALUES = [0.001, 0.01]


def get_ordered_feature_columns():
    """
    Returns the list of 192 feature columns strictly ordered by group:
    Margin (1-64), Shape (1-64), Texture (1-64).
    This order is critical for the FactorizedInteractionProjector.
    """
    margin_cols = [f"margin_{i+1}" for i in range(64)]
    shape_cols = [f"shape_{i+1}" for i in range(64)]
    texture_cols = [f"texture_{i+1}" for i in range(64)]
    return margin_cols + shape_cols + texture_cols


def load_dataset(split, scope, load_cached_data=True):
    """
    Loads data for a specific split and feature scope.

    Args:
        split (str): 'train', 'val', or 'test'.
        scope (str): 'Global', 'Physical', or 'Factorized'.
        load_cached_data (bool): Whether to use cached morphometrics.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): Feature matrix (float64).
            y (np.ndarray or None): Target labels (None for test).
            ids (np.ndarray): Image IDs.
    """
    metadata_path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    ids = df_meta["id"].values

    # Extract Targets (if available)
    y = None
    if "species" in df_meta.columns:
        y = df_meta["species"].values

    # Extract Features based on Scope
    if scope == "Physical":
        # Load morphometrics (id, hu_0..6, aspect_ratio...)
        df_morph = process_dataset(
            metadata_path, split, load_cached_data=load_cached_data
        )
        # Drop ID to get features
        # process_dataset guarantees row order matches metadata iteration
        X = df_morph.drop(columns=["id"]).values

    elif scope in ["Global", "Factorized"]:
        # Load pre-extracted features
        cols = get_ordered_feature_columns()
        X = df_meta[cols].values

    else:
        raise ValueError(f"Unknown scope: {scope}")

    return ensure_float64(X), y, ids


def build_expert_library():
    """
    Constructs the library of candidate experts.
    Cartesian Product: Scopes x Bases x Shrinkage

    Returns:
        dict: {expert_name: sklearn_pipeline}
    """
    scopes = ["Global", "Physical", "Factorized"]
    bases = {
        "Marginal": MarginalBasis(),
        "Rotational": RotationalBasis(),
        "Robust": RobustBasis(n_quantiles=50),
    }

    library = {}

    for scope in scopes:
        for basis_name, basis_transformer in bases.items():
            for shrinkage in SHRINKAGE_VALUES:
                name = f"{scope}_{basis_name}_LDA_{shrinkage}"
                steps = []

                # 1. Feature Engineering / Projection
                if scope == "Physical":
                    # Physical Scope: Polynomial Expansion of Morphometrics
                    steps.append(
                        (
                            "poly",
                            PolynomialFeatures(degree=2, include_bias=False),
                        )
                    )
                elif scope == "Factorized":
                    # Factorized Scope: Interaction Projector
                    steps.append(
                        (
                            "interact",
                            FactorizedInteractionProjector(n_components=10),
                        )
                    )
                # Global Scope uses raw features directly

                # 2. Basis Transformation (Preprocessing)
                steps.append(("basis", clone(basis_transformer)))

                # 3. Estimator (LDA with Fixed Shrinkage)
                # solver='lsqr' is required for shrinkage
                steps.append(
                    (
                        "clf",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage),
                    )
                )

                library[name] = Pipeline(steps)

    return library


def train_experts(library, X_dict, y):
    """
    Trains all experts in the library.

    Args:
        library (dict): Dictionary of pipelines.
        X_dict (dict): {scope: X_matrix}
        y (array-like): Target labels.

    Returns:
        dict: Trained library.
    """
    trained_library = {}
    print(f"Training {len(library)} experts...")

    for name, pipeline in library.items():
        # Determine scope from name
        scope = name.split("_")[0]
        X = X_dict[scope]

        # Fit
        pipeline.fit(X, y)
        trained_library[name] = pipeline

    return trained_library


def predict_experts(library, X_dict):
    """
    Generates probabilities for all experts.

    Args:
        library (dict): Trained pipelines.
        X_dict (dict): {scope: X_matrix}

    Returns:
        dict: {name: probability_matrix}
    """
    predictions = {}
    for name, pipeline in library.items():
        scope = name.split("_")[0]
        X = X_dict[scope]
        predictions[name] = ensure_float64(pipeline.predict_proba(X))
    return predictions


def greedy_forward_selection(predictions_dict, y_true, max_iter=20):
    """
    Performs Greedy Forward Selection to optimize ensemble weights.

    Args:
        predictions_dict (dict): {name: prob_matrix} for validation set.
        y_true (array-like): encoded validation labels.
        max_iter (int): Maximum number of experts to add.

    Returns:
        list: selected_experts (names)
        list: weights (counts of each expert)
    """
    expert_names = list(predictions_dict.keys())
    n_samples, n_classes = list(predictions_dict.values())[0].shape

    # Initialize
    selected = []
    current_ensemble_prob = np.zeros((n_samples, n_classes), dtype=np.float64)
    best_loss = float("inf")

    print(f"Starting Greedy Forward Selection (Max Iter: {max_iter})...")

    for i in range(max_iter):
        iteration_best_loss = float("inf")
        iteration_best_expert = None

        # Try adding each expert to the current ensemble
        for name in expert_names:
            prob = predictions_dict[name]

            # Calculate candidate ensemble probability
            # New Average = (CurrentSum + NewProb) / (Count + 1)
            # We maintain the sum for efficiency
            if i == 0:
                candidate_prob = prob
            else:
                candidate_prob = (current_ensemble_prob * i + prob) / (i + 1)

            loss = clipped_log_loss(y_true, candidate_prob)

            if loss < iteration_best_loss:
                iteration_best_loss = loss
                iteration_best_expert = name

        # Check for improvement
        if iteration_best_loss < best_loss:
            best_loss = iteration_best_loss
            selected.append(iteration_best_expert)

            # Update current ensemble probability (store the running average)
            prob_best = predictions_dict[iteration_best_expert]
            if i == 0:
                current_ensemble_prob = prob_best
            else:
                current_ensemble_prob = (current_ensemble_prob * i + prob_best) / (
                    i + 1
                )

            print(
                f"Iter {i+1}: Added {iteration_best_expert}, Val Loss: {best_loss:.15f}"
            )
        else:
            print(f"Iter {i+1}: No improvement. Stopping.")
            break

    # Calculate weights
    from collections import Counter

    counts = Counter(selected)
    unique_experts = list(counts.keys())
    weights = [counts[e] for e in unique_experts]

    return unique_experts, weights


def generate_submission():
    """
    Main execution pipeline:
    1. Load Data (Train, Val, Test).
    2. Build Library.
    3. Phase 1: Train on Train, Select on Val.
    4. Phase 2: Retrain Selected on Train+Val, Predict on Test.
    5. Save Submission.
    """
    set_seed(42)

    # --- 1. Load Data ---
    print("Loading datasets...")
    scopes = ["Global", "Physical", "Factorized"]

    # Dictionaries to hold data by scope
    X_train_dict = {}
    X_val_dict = {}
    X_test_dict = {}

    y_train = None
    y_val = None
    test_ids = None

    # Load Train
    for scope in scopes:
        X, y, _ = load_dataset("train", scope)
        X_train_dict[scope] = X
        if scope == "Global":
            y_train = y  # y is same for all

    # Load Val
    for scope in scopes:
        X, y, _ = load_dataset("val", scope)
        X_val_dict[scope] = X
        if scope == "Global":
            y_val = y

    # Load Test
    for scope in scopes:
        X, _, ids = load_dataset("test", scope)
        X_test_dict[scope] = X
        if scope == "Global":
            test_ids = ids

    # Encode Labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = le.classes_

    # --- 2. Build Library ---
    print("Building expert library...")
    library = build_expert_library()

    # --- 3. Phase 1: Selection ---
    print("\n--- Phase 1: Selection ---")
    # Train on Train split
    trained_lib_p1 = train_experts(library, X_train_dict, y_train_enc)

    # Predict on Val split
    val_preds = predict_experts(trained_lib_p1, X_val_dict)

    # Run Selection
    selected_names, weights = greedy_forward_selection(val_preds, y_val_enc)

    print(f"\nSelected {len(selected_names)} experts:")
    for name, w in zip(selected_names, weights):
        print(f"  - {name} (Weight: {w})")

    # --- 4. Phase 2: Retraining & Inference ---
    print("\n--- Phase 2: Retraining & Inference ---")

    # Prepare Combined Data (Train + Val)
    X_full_dict = {}
    for scope in scopes:
        X_full_dict[scope] = np.vstack([X_train_dict[scope], X_val_dict[scope]])

    y_full_enc = np.concatenate([y_train_enc, y_val_enc])

    # Filter library for selected experts
    selected_library = {name: library[name] for name in selected_names}

    # Retrain on Full Data
    trained_lib_p2 = train_experts(selected_library, X_full_dict, y_full_enc)

    # Predict on Test
    test_preds_dict = predict_experts(trained_lib_p2, X_test_dict)

    # Aggregate Predictions (Weighted Average)
    n_test = len(test_ids)
    n_classes = len(classes)
    final_probs = np.zeros((n_test, n_classes), dtype=np.float64)
    total_weight = sum(weights)

    for name, weight in zip(selected_names, weights):
        probs = test_preds_dict[name]
        final_probs += probs * weight

    final_probs /= total_weight

    # --- 5. Save Submission ---
    save_submission(test_ids, classes, final_probs, SUBMISSION_PATH)
