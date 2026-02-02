import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import ase

# Import from the provided library
from library.config import Config
from library.utils import log_transform, inverse_log_transform, rmsle_score
from library.data_processing import load_metadata, extract_physical_descriptors
from library.gnn_features import StructureEmbedder
from library.model_training import train_xgboost_model, cross_validate_model

# --- 1. Setup and Configuration ---
# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Override Config parameters for speed in this demonstration
print("--- Configuration ---")
Config.XGB_PARAMS["n_estimators"] = 5  # Reduce boosting rounds for speed
Config.XGB_PARAMS["n_jobs"] = 1  # Avoid multiprocessing overhead for tiny data
print(f"Modified XGB_PARAMS: {Config.XGB_PARAMS}")


# --- 2. Utilities Demonstration ---
print("\n--- Testing Utilities ---")
# Test data
y_true = np.array([1.0, 10.0, 100.0])
y_pred = np.array([1.1, 9.5, 102.0])

# Log transform
y_log = log_transform(y_true)
print(f"Log transform of {y_true}: {y_log}")
assert np.all(y_log >= 0), "Log transformed values should be non-negative for y >= 0"

# Inverse log transform
y_inv = inverse_log_transform(y_log)
print(f"Inverse transform: {y_inv}")
assert np.allclose(y_true, y_inv), "Inverse transform should recover original values"

# RMSLE Score
score = rmsle_score(y_true, y_pred)
print(f"RMSLE Score: {score:.4f}")
# Manual calculation check: sqrt(mean((log(1+y) - log(1+p))^2))
expected_score = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
assert np.isclose(score, expected_score), "RMSLE calculation mismatch"
print("Utilities verified.")


# --- 3. Data Processing Demonstration ---
print("\n--- Testing Data Processing ---")
# Load validation metadata (smaller than train)
df_meta = load_metadata(split="val")
print(f"Loaded validation metadata: {df_meta.shape}")

# Sample a small subset for speed (e.g., 5 samples)
df_sample = df_meta.head(5).copy()
print(f"Processing sample of {len(df_sample)} items...")

# Extract physical descriptors (Volume, Density)
# This reads the geometry files referenced in the metadata
df_features, atoms_list = extract_physical_descriptors(df_sample)

print(f"Extracted features shape: {df_features.shape}")
print(f"Number of atoms objects loaded: {len(atoms_list)}")

# Verify features exist
assert "volume" in df_features.columns, "Volume feature missing"
assert "density" in df_features.columns, "Density feature missing"
assert len(atoms_list) == len(df_sample), "Mismatch in atoms list length"

# Check values are reasonable
print(f"Sample Volume: {df_features.iloc[0]['volume']:.2f}")
print(f"Sample Density: {df_features.iloc[0]['density']:.2f}")
assert df_features.iloc[0]["volume"] > 0, "Volume should be positive"
print("Data processing verified.")


# --- 4. GNN Features Demonstration ---
print("\n--- Testing GNN Feature Extraction ---")
# Initialize the embedder (loads MatGL model)
# Note: This might download the model if not cached, which is expected behavior.
# We use cpu to ensure compatibility in this demo environment if GPU is busy/unavailable,
# though the class defaults to cuda if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing StructureEmbedder on {device}...")
embedder = StructureEmbedder(device=device)

# Generate embeddings for the sample atoms
print("Generating embeddings...")
df_emb = embedder.generate_embeddings(atoms_list, batch_size=2)

print(f"Embeddings shape: {df_emb.shape}")
# Verify output
assert len(df_emb) == len(df_sample), "Embedding count mismatch"
assert df_emb.shape[1] > 0, "No embedding features generated"
print("GNN features verified.")


# --- 5. Model Training Demonstration ---
print("\n--- Testing Model Training ---")

# Combine physical features and embeddings for training
# Drop non-numeric columns for XGBoost
numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
# Exclude targets and ID from features
targets = Config.TARGET_COLS
exclude = targets + [Config.ID_COL]
feature_cols = [c for c in numeric_cols if c not in exclude]

X_phys = df_features[feature_cols].reset_index(drop=True)
X_emb = df_emb.reset_index(drop=True)
X = pd.concat([X_phys, X_emb], axis=1)

# Targets for the sample
y_formation = df_features["formation_energy_ev_natom"].values
y_bandgap = df_features["bandgap_energy_ev"].values

print(f"Training data shape: {X.shape}")

# Test 1: Train a single model on Formation Energy
print("Training XGBoost on Formation Energy...")
model = train_xgboost_model(
    X,
    y_formation,
    X_val=X,
    y_val=y_formation,  # Using train as val just for demo mechanics
    early_stopping_rounds=2,
    verbose=True,
)
print("Model training complete.")

# Test prediction
preds_log = model.predict(X)
preds = inverse_log_transform(preds_log)
print(f"Predictions (first 3): {preds[:3]}")
print(f"Actuals (first 3): {y_formation[:3]}")

# Test 2: Cross-validation on Bandgap Energy
# We need at least n_splits samples. We have 5 samples, so 2 splits is fine.
print("\nRunning Cross-Validation on Bandgap Energy...")
cv_scores = cross_validate_model(X, pd.Series(y_bandgap), n_splits=2, random_state=42)

print(f"CV Scores: {cv_scores}")
assert len(cv_scores) == 2, "CV should return 2 scores"
print("Model training verified.")

print("\nAll demonstrations completed successfully.")
