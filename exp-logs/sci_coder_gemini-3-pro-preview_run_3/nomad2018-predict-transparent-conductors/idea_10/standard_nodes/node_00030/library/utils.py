import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from ase.io import read
from ase.neighborlist import neighbor_list
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error.
    """
    # Ensure non-negative predictions for log
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def save_submission(ids, formation_energy, bandgap_energy, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_energy,
            "bandgap_energy_ev": bandgap_energy,
        }
    )
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def compute_rdf_features(atoms, cutoff=8.0, n_bins=80):
    """
    Computes element-resolved Radial Distribution Function (RDF) features.
    High resolution (0.1A) and longer cutoff (8.0A) to capture medium-range order.
    Cite Lesson 29.
    """
    # Define elements of interest
    elements = ["Al", "Ga", "In", "O"]
    pairs = []
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            pairs.append(tuple(sorted((elements[i], elements[j]))))

    # Initialize histograms
    histograms = {pair: np.zeros(n_bins) for pair in pairs}
    bin_edges = np.linspace(0, cutoff, n_bins + 1)

    # Compute neighbors
    # 'i' and 'j' are indices of atoms, 'd' is distance
    try:
        i_indices, j_indices, dists = neighbor_list("ijd", atoms, cutoff)
    except Exception:
        # Fallback for empty or problematic structures
        return np.zeros(len(pairs) * n_bins)

    if len(dists) == 0:
        return np.zeros(len(pairs) * n_bins)

    chemical_symbols = np.array(atoms.get_chemical_symbols())

    # Create a DataFrame for easier grouping
    df_neighbors = pd.DataFrame(
        {
            "elem_i": chemical_symbols[i_indices],
            "elem_j": chemical_symbols[j_indices],
            "dist": dists,
        }
    )

    # Sort elements in each pair to match dictionary keys
    # We can swap elem_i and elem_j where elem_i > elem_j to enforce alphabetical order
    mask = df_neighbors["elem_i"] > df_neighbors["elem_j"]
    df_neighbors.loc[mask, ["elem_i", "elem_j"]] = df_neighbors.loc[
        mask, ["elem_j", "elem_i"]
    ].values

    # Group by pairs
    grouped = df_neighbors.groupby(["elem_i", "elem_j"])

    for (e1, e2), group in grouped:
        pair = (e1, e2)
        if pair in histograms:
            hist, _ = np.histogram(group["dist"], bins=bin_edges)
            # Normalize by number of atoms to make it intensive
            histograms[pair] = hist / len(atoms)

    # Flatten features
    features = []
    for pair in pairs:
        features.extend(histograms[pair])

    return np.array(features)


def get_physical_features(atoms):
    """
    Extracts basic physical descriptors from the ASE Atoms object.
    """
    vol = atoms.get_volume()
    mass = sum(atoms.get_masses())
    density = mass / vol if vol > 0 else 0.0
    return [vol, density]


def process_dataset(
    metadata_path, cache_path, input_dir="./input", load_cached_data=True
):
    """
    Loads metadata, extracts geometry features (Physical + RDF), and returns a DataFrame.
    Implements caching using parquet.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Lists to store extracted features
    rdf_data = []
    phys_data = []

    # Define column names for RDF features
    elements = ["Al", "Ga", "In", "O"]
    pairs = []
    for i in range(len(elements)):
        for j in range(i, len(elements)):
            pairs.append(f"{elements[i]}-{elements[j]}")

    # Updated RDF parameters for higher resolution
    n_bins = 80
    cutoff = 8.0

    rdf_cols = [f"rdf_{p}_{b}" for p in pairs for b in range(n_bins)]
    phys_cols = ["vol", "density"]

    for idx, row in df.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])

        try:
            atoms = read(file_path, format="aims")

            # Physical features
            phys = get_physical_features(atoms)
            phys_data.append(phys)

            # RDF features
            rdf = compute_rdf_features(atoms, cutoff=cutoff, n_bins=n_bins)
            rdf_data.append(rdf)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Append zeros in case of error to maintain shape
            phys_data.append([0.0, 0.0])
            rdf_data.append(np.zeros(len(rdf_cols)))

    # Create DataFrames for new features
    df_phys = pd.DataFrame(phys_data, columns=phys_cols)
    df_rdf = pd.DataFrame(rdf_data, columns=rdf_cols)

    # Concatenate with original metadata
    # We drop file_path to keep it clean, but keep ID for merging/tracking
    # Reset indices to ensure alignment
    df.reset_index(drop=True, inplace=True)
    df_phys.reset_index(drop=True, inplace=True)
    df_rdf.reset_index(drop=True, inplace=True)

    df_final = pd.concat([df, df_phys, df_rdf], axis=1)

    # Save to cache
    print(f"Saving features to {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final


def train_xgb_model(X_train, y_train, X_val, y_val, random_state=42):
    """
    Trains an XGBoost regressor with early stopping.
    y_train and y_val should be log-transformed targets.
    Hyperparameters optimized for high-dimensional RDF features (Shrinkage).
    Cite Lesson 4.
    """
    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.005,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.5,
        n_jobs=-1,
        random_state=random_state,
        objective="reg:squarederror",
        tree_method="hist",  # Faster training
        early_stopping_rounds=100,
        verbose=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,  # Silent training
    )

    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Metrics on log-transformed data (which approximates RMSLE on original data)
    train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
    val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

    print(f"  Train RMSE (log-space): {train_rmse:.6f}")
    print(f"  Val RMSE (log-space):   {val_rmse:.6f}")

    return model


def run_training_pipeline(train_meta_path, val_meta_path, cache_dir, random_state=42):
    """
    Orchestrates the feature extraction and training process.
    """
    # 1. Process Data
    train_cache = os.path.join(cache_dir, "train_features_rdf.parquet")
    val_cache = os.path.join(cache_dir, "val_features_rdf.parquet")

    df_train = process_dataset(train_meta_path, train_cache)
    df_val = process_dataset(val_meta_path, val_cache)

    # 2. Prepare Features and Targets
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    drop_cols = ["id", "file_path"] + target_cols

    # Ensure columns match
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]

    models = {}

    # 3. Train Models
    for target in target_cols:
        print(f"\nTraining model for {target}...")

        # Log transform targets
        y_train = np.log1p(df_train[target])
        y_val = np.log1p(df_val[target])

        model = train_xgb_model(X_train, y_train, X_val, y_val, random_state)
        models[target] = model

    return models, feature_cols


def generate_predictions(models, feature_cols, test_meta_path, cache_dir, output_path):
    """
    Generates predictions for the test set and saves the submission.
    """
    test_cache = os.path.join(cache_dir, "test_features_rdf.parquet")
    df_test = process_dataset(test_meta_path, test_cache)

    X_test = df_test[feature_cols]
    ids = df_test["id"]

    preds = {}

    for target, model in models.items():
        # Predict in log space
        log_preds = model.predict(X_test)
        # Inverse transform
        preds[target] = np.expm1(log_preds)

    save_submission(
        ids, preds["formation_energy_ev_natom"], preds["bandgap_energy_ev"], output_path
    )
