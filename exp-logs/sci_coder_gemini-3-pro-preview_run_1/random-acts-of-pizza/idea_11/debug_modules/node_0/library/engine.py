import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_dataset
from library.feature_engineering import generate_features
from library.semantic_processing import SemanticEngine
from library.dataset import get_dataloaders
from library.models import TopicAugmentedRF, AttentionGatedMLP, train_mlp_model


def get_rf_features(df, semantic_data_subset, numeric_cols):
    """
    Constructs the feature matrix for the Random Forest model.
    Combines:
    1. Tabular Metadata (Numerical)
    2. TF-IDF Vectors
    3. Topic Ratios
    4. Consistency Scores
    """
    # 1. Tabular Metadata
    # Ensure we strictly follow the column order
    tabular = df[numeric_cols].values.astype(np.float32)

    # 2. Semantic Features
    tfidf = semantic_data_subset["tfidf"]
    topic_ratios = semantic_data_subset["topic_ratios"]
    consistency = semantic_data_subset["consistency"].reshape(-1, 1)

    # Concatenate all features
    # Shape: (N, n_tabular + n_tfidf + n_topics + 1)
    X = np.hstack([tabular, tfidf, topic_ratios, consistency])
    return X


def train_rf_stream(train_df, val_df, semantic_data):
    """
    Trains and evaluates the Topic-Augmented Random Forest.
    """
    print("\n=== Stream A: Training Random Forest ===")

    # Identify numerical columns (excluding drop cols)
    drop_cols = set(Config.DROP_COLS)
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in drop_cols
    ]

    # Prepare Feature Matrices
    print("Constructing RF feature matrices...")
    X_train = get_rf_features(train_df, semantic_data["train"], numeric_cols)
    y_train = train_df[Config.TARGET_COL].values.astype(int)

    X_val = get_rf_features(val_df, semantic_data["val"], numeric_cols)
    y_val = val_df[Config.TARGET_COL].values.astype(int)

    # Initialize and Train
    rf_model = TopicAugmentedRF()
    print(f"Training RF with input shape: {X_train.shape}...")
    rf_model.fit(X_train, y_train)

    # Evaluate
    val_probs = rf_model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Random Forest Validation AUC: {val_auc}")

    return rf_model, numeric_cols


def train_mlp_stream(load_cached_data=True):
    """
    Trains and evaluates the Attention-Gated MLP.
    Uses get_dataloaders to handle specific MLP preprocessing.
    """
    print("\n=== Stream B: Training Attention-Gated MLP ===")

    # Get DataLoaders
    # Note: get_dataloaders handles loading, feat eng, semantic processing internally
    # and returns loaders with scaled/transformed data.
    train_loader, val_loader, test_loader, metadata_dim = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    model = AttentionGatedMLP(metadata_dim=metadata_dim)

    # Train
    trained_model, history = train_mlp_model(model, train_loader, val_loader)

    # Best Validation Score
    best_val_auc = max(history["val_auc"])
    print(f"MLP Best Validation AUC: {best_val_auc}")

    return trained_model, test_loader


def predict_rf(model, test_df, semantic_data, numeric_cols):
    """
    Generates predictions using the Random Forest model.
    """
    print("Generating RF predictions...")
    X_test = get_rf_features(test_df, semantic_data["test"], numeric_cols)
    probs = model.predict_proba(X_test)
    return probs


def predict_mlp(model, test_loader):
    """
    Generates predictions using the MLP model.
    """
    print("Generating MLP predictions...")
    device = Config.DEVICE
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            meta = batch["metadata"].to(device)
            req = batch["request_emb"].to(device)
            hist = batch["history_emb"].to(device)
            mask = batch["history_mask"].to(device)

            logits = model(meta, req, hist, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    return np.array(all_probs)


def run_pipeline(load_cached_data=True):
    """
    Main execution function.
    1. Loads and processes data.
    2. Trains RF and MLP.
    3. Ensembles predictions.
    4. Generates submission file.
    """
    set_seed(Config.SEED)

    # --- 1. Data Preparation (Shared) ---
    # We call these explicitly to have the raw objects for RF
    # get_dataloaders will call them again but hit the cache, so it's efficient.
    print("Initializing Data Pipeline...")
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)
    train_df, val_df, test_df = generate_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    semantic_engine = SemanticEngine()
    semantic_data = semantic_engine.process(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # --- 2. Train Stream A (Random Forest) ---
    rf_model, numeric_cols = train_rf_stream(train_df, val_df, semantic_data)

    # --- 3. Train Stream B (MLP) ---
    # We pass load_cached_data=True so it picks up the files generated/loaded in step 1
    mlp_model, test_loader = train_mlp_stream(load_cached_data=True)

    # --- 4. Inference ---
    print("\n=== Running Inference ===")

    # RF Predictions
    rf_probs = predict_rf(rf_model, test_df, semantic_data, numeric_cols)

    # MLP Predictions
    mlp_probs = predict_mlp(mlp_model, test_loader)

    # --- 5. Ensemble & Submission ---
    print("\n=== Generating Submission ===")

    # Weighted Average
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    final_probs = (w_rf * rf_probs) + (w_mlp * mlp_probs)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": test_df[Config.ID_COL], Config.TARGET_COL: final_probs}
    )

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Head of submission:\n{submission.head()}")
