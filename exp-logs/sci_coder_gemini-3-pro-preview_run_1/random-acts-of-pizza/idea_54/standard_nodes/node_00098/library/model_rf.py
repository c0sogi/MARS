import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from library.config import RF_PARAMS, WORKING_DIR
from library.utils import compute_auc


def train_rf_model(rf_features, targets):
    """
    Trains the Random Forest model (Stream A) using the provided features and targets.

    Args:
        rf_features (dict): Dictionary containing 'train', 'val', and 'test' feature arrays (numpy).
        targets (dict): Dictionary containing 'train' and 'val' target arrays (numpy).

    Returns:
        tuple: (val_preds, test_preds, model)
            val_preds (np.ndarray): Probability predictions for the validation set (class 1).
            test_preds (np.ndarray): Probability predictions for the test set (class 1).
            model (RandomForestClassifier): The trained Random Forest model.
    """

    # 1. Unpack Data
    # rf_features contains the concatenated Interaction-Projected features (TF-IDF + Metadata + Interactions)
    X_train = rf_features["train"]
    y_train = targets["train"]

    X_val = rf_features["val"]
    y_val = targets["val"]

    X_test = rf_features["test"]

    # 2. Initialize Model
    # RF_PARAMS are loaded from config.py. Expected keys:
    # n_estimators, min_samples_leaf, class_weight, random_state, n_jobs
    print("Initializing Random Forest (Stream A)...")
    model = RandomForestClassifier(**RF_PARAMS)

    # 3. Train Model
    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    model.fit(X_train, y_train)

    # 4. Evaluation (Validation)
    print("Generating validation predictions...")
    # predict_proba returns [prob_class_0, prob_class_1]
    val_preds = model.predict_proba(X_val)[:, 1]

    auc_score = compute_auc(y_val, val_preds)
    # Print full precision as requested
    print(f"Random Forest Validation AUC: {auc_score}")

    # 5. Inference (Test)
    print("Generating test predictions...")
    test_preds = model.predict_proba(X_test)[:, 1]

    # 6. Save Artifacts
    # Save model and predictions to WORKING_DIR for the ensemble step or caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    model_path = os.path.join(WORKING_DIR, "best_rf.joblib")
    joblib.dump(model, model_path)

    preds_path = os.path.join(WORKING_DIR, "rf_preds.npz")
    np.savez(preds_path, val=val_preds, test=test_preds)

    return val_preds, test_preds, model
