import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and python environment.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(ids, classes, probs, output_path):
    """
    Saves the prediction probabilities to a CSV file in the required format.

    Args:
        ids: Array of image IDs.
        classes: Array of class names (strings).
        probs: Matrix of probabilities (n_samples, n_classes).
        output_path: Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame with class columns
    df_sub = pd.DataFrame(probs, columns=classes)

    # Insert ID column at the beginning
    df_sub.insert(0, "id", ids)

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_and_process_data(
    metadata_dir="./metadata", cache_dir="./working/idea_3", load_cached_data=True
):
    """
    Loads data from metadata CSVs, processes it (scaling), and handles caching.
    Combines Train and Validation sets for full training as per the strategy.

    Args:
        metadata_dir: Directory containing train.csv, val.csv, test.csv.
        cache_dir: Directory to store/load cached numpy arrays.
        load_cached_data: Boolean to enable/disable loading from cache.

    Returns:
        X_train (scaled), y_train, X_test (scaled), test_ids, classes
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        X_train = np.load(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"], allow_pickle=True)
        X_test = np.load(cache_paths["X_test"])
        test_ids = np.load(cache_paths["test_ids"])
        classes = np.load(cache_paths["classes"], allow_pickle=True)
        return X_train, y_train, X_test, test_ids, classes

    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Combine Train and Val for maximum sample efficiency
    full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # Identify feature columns (margin, shape, texture)
    feature_cols = [
        c for c in full_train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Extract raw data and labels
    X_train_raw = full_train_df[feature_cols].values
    y_train = full_train_df["species"].values
    X_test_raw = test_df[feature_cols].values
    test_ids = test_df["id"].values
    classes = np.sort(full_train_df["species"].unique())

    # Preprocessing: Global Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Save to cache
    np.save(cache_paths["X_train"], X_train)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["X_test"], X_test)
    np.save(cache_paths["test_ids"], test_ids)
    np.save(cache_paths["classes"], classes)

    return X_train, y_train, X_test, test_ids, classes


def train_and_predict(X_train, y_train, X_test, classes, random_state=42):
    """
    Trains the Bayesian-Linear Hybrid Ensemble and generates predictions.

    Components:
    1. Logistic Regression (Discriminative Linear)
    2. Linear Discriminant Analysis (Generative Linear)
    3. Gaussian Process Classifier (Probabilistic Non-Linear) with PCA

    Returns:
        probs_ensemble: Averaged probability matrix.
    """
    print("Starting Ensemble Training...")

    # --- Model 1: Logistic Regression ---
    print("Training Logistic Regression (Discriminative Linear)...")
    # Search grid focusing on high C (weak regularization) for high SNR
    Cs = np.logspace(-2, 6, 10)
    lr = LogisticRegressionCV(
        Cs=Cs,
        cv=3,
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=5000,
        random_state=random_state,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    print(f"LR Training Accuracy: {lr.score(X_train, y_train)}")
    probs_lr = lr.predict_proba(X_test)

    # --- Model 2: Linear Discriminant Analysis ---
    print("Training LDA (Generative Linear)...")
    # Use lsqr solver with auto shrinkage (Ledoit-Wolf)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train, y_train)
    print(f"LDA Training Accuracy: {lda.score(X_train, y_train)}")
    probs_lda = lda.predict_proba(X_test)

    # --- Model 3: Gaussian Process Classifier ---
    print("Training GPC (Probabilistic Non-Linear)...")

    # Step 3a: PCA Projection for GPC branch
    # Reduce dimensionality to avoid curse of dimensionality in kernel space
    n_components = 40
    print(f"Applying PCA (n_components={n_components}) for GPC branch...")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Step 3b: GPC Training
    # RBF kernel with internal hyperparameter optimization
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(
        kernel=kernel, random_state=random_state, n_jobs=-1, copy_X_train=False
    )
    gpc.fit(X_train_pca, y_train)
    print(f"GPC Log Marginal Likelihood: {gpc.log_marginal_likelihood_value_}")
    print(f"GPC Training Accuracy: {gpc.score(X_train_pca, y_train)}")
    probs_gpc = gpc.predict_proba(X_test_pca)

    # --- Ensemble: Soft Voting ---
    print("Calculating Ensemble Predictions (Soft Vote)...")
    probs_ensemble = (probs_lr + probs_lda + probs_gpc) / 3.0

    return probs_ensemble
