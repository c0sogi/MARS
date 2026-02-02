import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- Import Library Modules ---
# We assume the file structure provided in the prompt exists.
from library.config import Config
from library.utils import seed_everything, get_logger
from library.features import preprocess_data
from library.knn_embedding import KNNFeatureExtractor
from library.model import XGBoostTrainer, inject_knn_features


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> 1. Configuring environment for demo...")

    # Override Config for a fast, deterministic demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small subset for speed
    Config.NUM_BOOST_ROUND = 5  # Minimal rounds for training demo
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = False  # Silent XGBoost
    Config.KNN_K = 5  # Small K for demo
    Config.KNN_BATCH_SIZE = 512  # Safe batch size

    # Set a temporary cache directory in working
    Config.IDEA_DIR = "./working/demo_cache"
    if os.path.exists(Config.IDEA_DIR):
        shutil.rmtree(Config.IDEA_DIR)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Ensure reproducibility
    seed_everything(Config.SEED)
    logger = get_logger("demo")
    logger.info("Configuration complete. Starting demo pipeline.")

    # ==========================================
    # 2. Feature Engineering Demo
    # ==========================================
    print("\n>>> 2. Running Feature Engineering (preprocess_data)...")

    # This function loads raw data, subsamples it (since debug=True),
    # computes physics features, dense indices, and scales columns for KNN.
    train_df, val_df, test_df = preprocess_data(
        load_cached_data=False,  # Force fresh processing
        debug=True,
        debug_samples=Config.DEBUG_SAMPLES,
    )

    # --- Verification ---
    print("   Verifying Feature Engineering outputs...")
    assert (
        len(train_df) == Config.DEBUG_SAMPLES
    ), f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == Config.DEBUG_SAMPLES, f"Val size mismatch: {len(val_df)}"

    # Check for Physics Features
    expected_physics = [
        "Euclidean_Distance_To_Hydrology",
        "Hydrology_Elevation",
        "Aspect_Sin",
    ]
    for col in expected_physics:
        assert col in train_df.columns, f"Missing physics feature: {col}"

    # Check for Dense Indices
    if "Soil_Type1" in train_df.columns:  # Assuming raw data had OHE
        assert "Soil_Type_Index" in train_df.columns, "Missing Soil_Type_Index"

    # Check for Scaled Columns (required for KNN)
    expected_scaled = [f"{c}_scaled" for c in Config.KNN_FEATURES]
    for col in expected_scaled:
        assert col in train_df.columns, f"Missing scaled feature: {col}"

    print("   Feature Engineering verification passed.")

    # ==========================================
    # 3. KNN Feature Extractor Demo (Unit Test)
    # ==========================================
    print("\n>>> 3. Testing KNNFeatureExtractor (Unit Test)...")

    # Create synthetic data
    # Reference: 100 samples, 5 features, 3 classes
    n_ref = 100
    n_query = 20
    n_features = 5
    n_classes = 3

    X_ref_syn = np.random.randn(n_ref, n_features).astype(np.float32)
    y_ref_syn = np.random.randint(0, n_classes, size=n_ref)
    X_query_syn = np.random.randn(n_query, n_features).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    knn_engine = KNNFeatureExtractor(k=10, device=device)

    # Fit
    knn_engine.fit(X_ref_syn, y_ref_syn)

    # Transform
    knn_feats_syn = knn_engine.transform(X_query_syn, batch_size=32, exclude_self=False)

    # --- Verification ---
    print("   Verifying KNN Extractor outputs...")
    # Expected columns: 1 density + n_classes probability columns
    expected_cols = 1 + n_classes
    assert knn_feats_syn.shape == (
        n_query,
        expected_cols,
    ), f"KNN output shape mismatch. Got {knn_feats_syn.shape}, expected {(n_query, expected_cols)}"

    # Check Density is non-negative
    assert (knn_feats_syn["KNN_Density"] >= 0).all(), "Negative density values found."

    # Check Probabilities sum to 1 (approx)
    prob_cols = [c for c in knn_feats_syn.columns if "Prob" in c]
    prob_sums = knn_feats_syn[prob_cols].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    knn_engine.clear_memory()
    print("   KNN Unit Test verification passed.")

    # ==========================================
    # 4. KNN Injection Integration Demo
    # ==========================================
    print("\n>>> 4. Testing KNN Injection on Real Data...")

    # We will use the processed train_df as both reference and query for this demo
    # In a real scenario, we use exclude_self=True for train-on-train

    # Ensure target is present
    assert Config.TARGET_COL in train_df.columns

    # Inject
    train_df_aug = inject_knn_features(
        ref_df=train_df,
        query_df=train_df,
        knn_cols=Config.KNN_FEATURES,
        exclude_self=True,
    )

    # --- Verification ---
    print("   Verifying KNN Injection...")
    # Check if new columns are added
    new_cols = [c for c in train_df_aug.columns if c not in train_df.columns]
    knn_cols_added = [c for c in new_cols if "KNN_" in c]

    assert len(knn_cols_added) > 0, "No KNN columns were injected."
    assert "KNN_Density" in knn_cols_added, "KNN_Density missing."

    # Check row count preservation
    assert len(train_df_aug) == len(train_df), "Row count changed after injection."

    print(f"   Injected {len(knn_cols_added)} features successfully.")

    # ==========================================
    # 5. Model Training Demo
    # ==========================================
    print("\n>>> 5. Testing XGBoost Training...")

    # Prepare Data
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_df_aug[Config.TARGET_COL])
    num_classes = len(le.classes_)

    # Select Features (exclude ID, Target, and Scaled intermediate cols)
    ignore_cols = [Config.ID_COL, Config.TARGET_COL] + [
        f"{c}_scaled" for c in Config.KNN_FEATURES
    ]
    feature_cols = [c for c in train_df_aug.columns if c not in ignore_cols]

    X = train_df_aug[feature_cols]

    # Split into Train/Val for the trainer
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]

    print(f"   Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
    print(f"   Features: {len(feature_cols)}, Classes: {num_classes}")

    # Initialize Trainer
    trainer = XGBoostTrainer(num_class=num_classes, device=device)

    # Fit Model
    model = trainer.fit(X_train, y_train, X_val, y_val)

    # Predict
    preds = trainer.predict_proba(model, X_val)

    # --- Verification ---
    print("   Verifying Model Predictions...")
    assert preds.shape == (
        len(X_val),
        num_classes,
    ), f"Prediction shape mismatch. Got {preds.shape}, expected {(len(X_val), num_classes)}"

    # Check probability range
    assert (preds >= 0).all() and (
        preds <= 1.00001
    ).all(), "Predictions out of probability range."

    # Check accuracy is not random (simple heuristic check, though on 5 rounds it might be poor)
    # We just check if it runs without error.
    acc = (np.argmax(preds, axis=1) == y_val).mean()
    print(f"   Demo Model Accuracy (5 rounds): {acc:.4f}")

    print("   Model Training verification passed.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
