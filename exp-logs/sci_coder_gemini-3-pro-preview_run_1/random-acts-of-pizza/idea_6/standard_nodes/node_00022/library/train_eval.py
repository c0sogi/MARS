import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, utils, feature_engineering, neural_net


def train_rf(train_data, val_data):
    """
    Trains the Lexical-Ratio Random Forest model.
    Combines Dense Ratio features and Sparse TF-IDF features.
    """
    print("Preparing data for Random Forest...")

    # Combine Dense and TF-IDF features
    X_train = sparse.hstack([train_data["dense"], train_data["tfidf"]])
    y_train = train_data["y"]

    X_val = sparse.hstack([val_data["dense"], val_data["tfidf"]])
    y_val = val_data["y"]

    print(f"Training Random Forest with config: {config.RF_CONFIG}...")
    rf_model = RandomForestClassifier(**config.RF_CONFIG)
    rf_model.fit(X_train, y_train)

    # Validation
    val_probs = rf_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Random Forest Validation AUC: {val_auc}")

    return rf_model


def train_mlp_wrapper(train_data, val_data):
    """
    Wrapper to train the Stabilized Dual-Branch MLP.
    Delegates to library.neural_net.train_model.
    """
    print("Initializing Dual-Branch MLP training...")

    # Determine dense input dimension from training data
    dense_input_dim = train_data["dense"].shape[1]

    # Train using the neural_net module
    mlp_model = neural_net.train_model(train_data, val_data, dense_input_dim)

    return mlp_model


def predict_ensemble(rf_model, mlp_model, test_data):
    """
    Generates predictions using the weighted ensemble of RF and MLP.
    """
    print("Generating ensemble predictions...")

    # --- 1. Random Forest Predictions ---
    # Construct input: Dense + TF-IDF
    X_test_rf = sparse.hstack([test_data["dense"], test_data["tfidf"]])
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]

    # --- 2. MLP Predictions ---
    # Uses embedding and dense inputs
    mlp_probs = neural_net.predict_model(mlp_model, test_data)

    # --- 3. Weighted Average ---
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    ensemble_probs = (w_rf * rf_probs) + (w_mlp * mlp_probs)

    print(f"Ensemble Weights -> RF: {w_rf}, MLP: {w_mlp}")

    return ensemble_probs


def run_training_pipeline(load_cached_data=True):
    """
    Main execution function.
    1. Loads/Processes Data
    2. Trains RF
    3. Trains MLP
    4. Ensembles and Predicts on Test
    5. Saves Submission
    """
    # Set global seed for reproducibility
    utils.set_seed()

    # Initialize Feature Processor
    processor = feature_engineering.FeatureProcessor()

    # Load Data (Cached or Processed from scratch)
    data = processor.process_data(load_cached_data=load_cached_data)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # --- Train Models ---
    rf_model = train_rf(train_data, val_data)
    mlp_model = train_mlp_wrapper(train_data, val_data)

    # --- Validation Ensemble Check ---
    print("Evaluating Ensemble on Validation Set...")
    # RF Val Probs
    X_val_rf = sparse.hstack([val_data["dense"], val_data["tfidf"]])
    rf_val_probs = rf_model.predict_proba(X_val_rf)[:, 1]

    # MLP Val Probs
    # We need to re-run prediction on val using the predict_model utility for consistency
    # (though train_model prints metrics, we need the actual array here)
    mlp_val_probs = neural_net.predict_model(mlp_model, val_data)

    # Ensemble
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]
    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)

    ensemble_val_auc = roc_auc_score(val_data["y"], ensemble_val_probs)
    print(f"Final Ensemble Validation AUC: {ensemble_val_auc}")

    # --- Test Inference & Submission ---
    test_probs = predict_ensemble(rf_model, mlp_model, test_data)
    test_ids = test_data["ids"]

    utils.save_submission(test_probs, test_ids)
