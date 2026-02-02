import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, feature_engine, neural_net, data_loader


def train_rf(X_train, y_train):
    """
    Trains the Random Forest model (Stream A).
    """
    print("Training Random Forest (Stream A)...")
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight=config.RF_CLASS_WEIGHT,
        max_depth=config.RF_MAX_DEPTH,
        n_jobs=config.RF_N_JOBS,
        random_state=config.RANDOM_STATE,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    return rf


def train_nn(train_data, val_data, device):
    """
    Trains the Dual-Query MLP (Stream B) using the library function.
    """
    print(f"Training Neural Network (Stream B) on {device}...")
    model, best_auc = neural_net.train_model(train_data, val_data, device=device)
    return model, best_auc


def evaluate_ensemble(rf_model, nn_model, X_rf, data_nn, y_true, device):
    """
    Evaluates the ensemble on a given dataset (usually validation).
    """
    # RF Predictions
    rf_probs = rf_model.predict_proba(X_rf)[:, 1]

    # NN Predictions
    nn_probs = neural_net.predict(nn_model, data_nn, device=device)

    # Ensemble
    w_rf, w_nn = config.ENSEMBLE_WEIGHTS
    ensemble_probs = (w_rf * rf_probs) + (w_nn * nn_probs)

    # Metrics
    rf_auc = roc_auc_score(y_true, rf_probs)
    nn_auc = roc_auc_score(y_true, nn_probs)
    ensemble_auc = roc_auc_score(y_true, ensemble_probs)

    print("-" * 40)
    print(f"Random Forest AUC: {rf_auc}")
    print(f"Neural Network AUC: {nn_auc}")
    print(f"Ensemble AUC:      {ensemble_auc}")
    print("-" * 40)

    return ensemble_auc


def generate_submission(rf_model, nn_model, X_rf_test, data_nn_test, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating test predictions...")

    # RF Predictions
    rf_probs = rf_model.predict_proba(X_rf_test)[:, 1]

    # NN Predictions
    nn_probs = neural_net.predict(nn_model, data_nn_test, device=device)

    # Ensemble
    w_rf, w_nn = config.ENSEMBLE_WEIGHTS
    final_probs = (w_rf * rf_probs) + (w_nn * nn_probs)

    # Load Test Metadata for IDs
    df_test = data_loader.load_dataset("test", load_cached_data=True)
    request_ids = df_test["request_id"].values

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": final_probs}
    )

    # Save
    save_path = config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training, evaluation, and submission pipeline.
    """
    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Features
    # Returns tuples of (X_rf, data_nn)
    train_res, val_res, test_res = feature_engine.run_feature_pipeline(
        load_cached_data=load_cached_data
    )

    X_rf_train, data_nn_train = train_res
    X_rf_val, data_nn_val = val_res
    X_rf_test, data_nn_test = test_res

    # Extract Labels (Shared between streams)
    # data_nn dictionaries contain 'labels' key for train/val
    y_train = data_nn_train["labels"]
    y_val = data_nn_val["labels"]

    # 2. Train Random Forest
    rf_model = train_rf(X_rf_train, y_train)

    # 3. Train Neural Network
    nn_model, _ = train_nn(data_nn_train, data_nn_val, device)

    # 4. Evaluate on Validation
    print("\nEvaluating on Validation Set:")
    evaluate_ensemble(rf_model, nn_model, X_rf_val, data_nn_val, y_val, device)

    # 5. Generate Submission
    generate_submission(rf_model, nn_model, X_rf_test, data_nn_test, device)
