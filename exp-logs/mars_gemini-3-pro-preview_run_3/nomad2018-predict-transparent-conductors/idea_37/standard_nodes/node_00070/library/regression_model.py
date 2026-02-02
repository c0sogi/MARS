import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.data_handler import load_metadata
from library.feature_manager import generate_feature_matrix, clean_features

# Constants
SUBMISSION_DIR = "./submission"
RANDOM_SEED = 42


class LogTransformedXGBoost:
    """
    XGBoost regressor wrapper that handles log-transformation of targets
    to directly optimize for RMSLE (Root Mean Squared Logarithmic Error).
    """

    def __init__(
        self,
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        early_stopping_rounds=100,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="reg:squarederror",  # Optimizing MSE on log(1+y) is equivalent to RMSLE on y
            early_stopping_rounds=early_stopping_rounds,
        )
        self.target_name = ""

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        target_name="target",
    ):
        """
        Trains the model using log-transformed targets.
        """
        self.target_name = target_name

        # Log transformation: z = log(1 + y)
        z_train = np.log1p(y_train)
        z_val = np.log1p(y_val)

        print(f"Training XGBoost for {self.target_name}...")
        self.model.fit(
            X_train,
            z_train,
            eval_set=[(X_val, z_val)],
            verbose=False,
        )

        # Evaluate on validation set
        z_pred = self.model.predict(X_val)
        rmsle = np.sqrt(mean_squared_error(z_val, z_pred))
        print(f"Validation RMSLE for {self.target_name}: {rmsle}")

        return rmsle

    def predict(self, X):
        """
        Predicts and applies inverse log transformation.
        """
        # Predict in log space
        z_pred = self.model.predict(X)
        # Inverse transform: y = exp(z) - 1
        y_pred = np.expm1(z_pred)
        # Ensure no negative predictions (physical energy constraints)
        y_pred = np.maximum(y_pred, 0.0)
        return y_pred


def prepare_data(split, load_cached_data=True):
    """
    Loads metadata and features, merges them, and cleans the feature set.
    """
    # Load metadata
    df_meta = load_metadata(split)

    # Generate features (cached)
    df_features = generate_feature_matrix(split, load_cached_data=load_cached_data)

    # Ensure ID types match for merging
    df_meta["id"] = df_meta["id"].astype(int)
    df_features["id"] = df_features["id"].astype(int)

    # Merge to align rows
    df_merged = pd.merge(df_meta, df_features, on="id", how="inner")

    # Extract IDs
    ids = df_merged["id"]

    # Extract Targets if available
    targets = {}
    if "formation_energy_ev_natom" in df_merged.columns:
        targets["formation_energy_ev_natom"] = df_merged["formation_energy_ev_natom"]
    if "bandgap_energy_ev" in df_merged.columns:
        targets["bandgap_energy_ev"] = df_merged["bandgap_energy_ev"]

    # Extract Features (all columns from feature matrix except 'id')
    feature_cols = [c for c in df_features.columns if c != "id"]
    X = df_merged[feature_cols]

    # Clean features (handle NaNs, Infs, constant columns)
    X = clean_features(X)

    return ids, X, targets


def run_training_and_inference(load_cached_data=True):
    """
    Main pipeline function:
    1. Prepares Train, Val, and Test data.
    2. Aligns feature columns across splits.
    3. Trains models for both targets.
    4. Generates and saves predictions.
    """
    print("Preparing Training Data...")
    _, X_train, y_train_dict = prepare_data("train", load_cached_data)

    print("Preparing Validation Data...")
    _, X_val, y_val_dict = prepare_data("val", load_cached_data)

    # Align Validation columns to Training columns
    # (clean_features might drop different constant columns in different splits)
    train_cols = X_train.columns.tolist()

    # Add missing columns to Val with 0
    for col in train_cols:
        if col not in X_val.columns:
            X_val[col] = 0.0
    # Keep only training columns in correct order
    X_val = X_val[train_cols]

    # Initialize Models with optimized hyperparameters
    # Using slightly lower subsample to encourage using diverse features
    model_formation = LogTransformedXGBoost(
        n_estimators=3000,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        max_depth=6,
    )

    model_bandgap = LogTransformedXGBoost(
        n_estimators=3000,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        max_depth=6,
    )

    # Train Formation Energy Model
    rmsle_formation = model_formation.train(
        X_train,
        y_train_dict["formation_energy_ev_natom"],
        X_val,
        y_val_dict["formation_energy_ev_natom"],
        target_name="Formation Energy",
    )

    # Train Bandgap Energy Model
    rmsle_bandgap = model_bandgap.train(
        X_train,
        y_train_dict["bandgap_energy_ev"],
        X_val,
        y_val_dict["bandgap_energy_ev"],
        target_name="Bandgap Energy",
    )

    print(f"Average RMSLE: {(rmsle_formation + rmsle_bandgap) / 2}")

    # Inference
    print("Preparing Test Data...")
    test_ids, X_test, _ = prepare_data("test", load_cached_data)

    # Align Test columns to Training columns
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[train_cols]

    print("Generating Predictions...")
    pred_formation = model_formation.predict(X_test)
    pred_bandgap = model_bandgap.predict(X_test)

    # Create submission dataframe
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "formation_energy_ev_natom": pred_formation,
            "bandgap_energy_ev": pred_bandgap,
        }
    )

    # Save submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission.head())
