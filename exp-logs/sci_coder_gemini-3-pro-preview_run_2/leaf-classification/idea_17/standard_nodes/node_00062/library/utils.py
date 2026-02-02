import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.base import clone

# Constants
CACHE_DIR = "./working/idea_17"
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clip_and_score(y_true, y_pred_probs):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    as required by the competition metric.
    """
    # Rescale rows to sum to 1 (each row is divided by the row sum)
    row_sums = y_pred_probs.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    y_pred_probs = y_pred_probs / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of the log function
    eps = 1e-15
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)

    # Calculate Log Loss
    return log_loss(y_true, y_pred_probs)


def load_data(load_cached_data=True):
    """
    Loads data from metadata or cache.
    Returns: X_train, y_train, X_val, y_val, X_test, test_ids, classes
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Processing data from metadata...")
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Identify feature columns
    feature_cols = [
        c for c in train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Encode targets
    le = LabelEncoder()
    # Fit on all possible species (train + val) to ensure consistency
    all_species = pd.concat([train_df["species"], val_df["species"]]).unique()
    le.fit(all_species)
    classes = le.classes_

    # Extract arrays
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = le.transform(train_df["species"])

    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = le.transform(val_df["species"])

    X_test = test_df[feature_cols].values.astype(np.float64)
    test_ids = test_df["id"].values

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def get_experts_pool(random_state=42):
    """
    Constructs the pool of experts: Linear LDA, Kernel LDA, and Linear LR.
    """
    # Expert A: Linear Generative Anchor
    expert_a = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    # Expert B: Kernel Generative Expert
    # PCA to densify -> Nystroem for non-linear mapping -> LDA
    expert_b = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            (
                "nystroem",
                Nystroem(kernel="rbf", n_components=300, random_state=random_state),
            ),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    # Expert C: Discriminative Linear Expert
    # Dense logarithmic grid for C
    cs_grid = np.logspace(-4, 4, 20)
    expert_c = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr_cv",
                LogisticRegressionCV(
                    Cs=cs_grid,
                    cv=5,
                    scoring="neg_log_loss",
                    max_iter=2000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {"Linear_LDA": expert_a, "Kernel_LDA": expert_b, "Linear_LR": expert_c}


def ensemble_selection(predictions_dict, y_true, iterations=100):
    """
    Performs Greedy Forward Selection (Caruana et al.) to find optimal ensemble weights.
    """
    model_names = list(predictions_dict.keys())
    n_samples, n_classes = list(predictions_dict.values())[0].shape

    # Initialize ensemble predictions (sum)
    ensemble_sum = np.zeros((n_samples, n_classes))
    selected_models = []

    best_loss = float("inf")

    # Iteratively add the model that minimizes the loss of the ensemble mean
    for i in range(1, iterations + 1):
        best_model_to_add = None
        current_best_loss_iter = float("inf")

        for name in model_names:
            preds = predictions_dict[name]
            # Calculate what the ensemble average would be if we added this model
            temp_ensemble = (ensemble_sum + preds) / i
            loss = clip_and_score(y_true, temp_ensemble)

            if loss < current_best_loss_iter:
                current_best_loss_iter = loss
                best_model_to_add = name

        # Update ensemble
        selected_models.append(best_model_to_add)
        ensemble_sum += predictions_dict[best_model_to_add]
        best_loss = current_best_loss_iter

    # Calculate final weights based on selection counts
    total_selected = len(selected_models)
    weights = {
        name: selected_models.count(name) / total_selected for name in model_names
    }

    return weights, best_loss


def run_training_pipeline(load_cached_data=True, random_state=42):
    """
    Main driver function for the Selective Kernel-Generative Hybrid Ensemble.
    """
    set_seed(random_state)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        load_cached_data
    )
    print(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 2. Phase 1: Expert Training & Selection
    print("\n--- Phase 1: Expert Training & Selection ---")
    experts = get_experts_pool(random_state)
    val_predictions = {}

    # Train each expert on X_train and predict on X_val
    for name, model in experts.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predict Probabilities
        probs = model.predict_proba(X_val)
        val_predictions[name] = probs

        score = clip_and_score(y_val, probs)
        print(f"  {name} Val LogLoss: {score:.15f}")

    # Select optimal weights
    print("Selecting Ensemble Weights...")
    weights, ensemble_score = ensemble_selection(val_predictions, y_val)

    print(f"Ensemble Selection Complete.")
    print(f"  Best Ensemble Val LogLoss: {ensemble_score:.15f}")
    print("  Weights:")
    for name, w in weights.items():
        if w > 0:
            print(f"    {name}: {w:.4f}")

    # 3. Phase 2: Final Retraining on Full Data
    print("\n--- Phase 2: Final Retraining ---")
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    final_experts = {}

    for name, w in weights.items():
        if w == 0:
            continue

        print(f"Retraining {name} on full data...")
        # Clone to reset the model
        refit_model = clone(experts[name])
        refit_model.fit(X_full, y_full)
        final_experts[name] = refit_model

    # 4. Inference on Test Set
    print("\nGenerating Test Predictions...")
    test_probs_sum = np.zeros((len(X_test), len(classes)))

    for name, w in weights.items():
        if w > 0:
            model = final_experts[name]
            probs = model.predict_proba(X_test)
            test_probs_sum += w * probs

    # Normalize final probabilities
    final_probs = test_probs_sum / test_probs_sum.sum(axis=1, keepdims=True)

    # 5. Save Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
