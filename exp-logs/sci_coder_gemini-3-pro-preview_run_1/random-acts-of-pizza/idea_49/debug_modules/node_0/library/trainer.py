import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.models_mlp import train_model as train_mlp_model, predict as predict_mlp
from library.models_rf import train_rf_model, predict_rf


class Trainer:
    """
    Orchestrates the training and inference pipeline for the Hybrid Ensemble.
    Manages data loading, feature engineering, model training (MLP + RF),
    ensemble validation, and submission generation.
    """

    def __init__(self):
        pass

    def train(self, debug=False, max_samples=None, epochs=None):
        """
        Executes the full training pipeline.

        Args:
            debug (bool): If True, enables debug mode in Config.
            max_samples (int): Optional limit on dataset size for quick testing.
            epochs (int): Optional override for MLP training epochs.
        """
        # 1. Apply Configuration Overrides
        if debug:
            Config.DEBUG = True
        if max_samples is not None:
            Config.MAX_SAMPLES = max_samples
        if epochs is not None:
            Config.MLP_EPOCHS = epochs

        Config.ensure_dirs()

        # 2. Load Data
        print("Loading data...")
        dl = DataLoader()
        train_df, val_df, test_df = dl.load_dataset(load_cached_data=True)

        # Extract labels and IDs
        y_train = train_df["requester_received_pizza"].astype(int).values
        y_val = val_df["requester_received_pizza"].astype(int).values
        test_ids = test_df["request_id"].values

        # 3. Feature Engineering
        print("Engineering features...")
        fe = FeatureEngineer()
        features = fe.create_features(load_cached_data=True)

        # 4. Train MLP Model
        print("Training MLP Model...")
        # Determine metadata dimension dynamically from the engineered features
        # features['train']['mlp']['metadata'] has shape (N_samples, N_features)
        mlp_meta_dim = features["train"]["mlp"]["metadata"].shape[1]

        mlp_model = train_mlp_model(
            features["train"]["mlp"],
            y_train,
            features["val"]["mlp"],
            y_val,
            mlp_meta_dim,
        )

        # 5. Train Random Forest Model
        print("Training Random Forest Model...")
        rf_model = train_rf_model(
            features["train"]["rf"], y_train, features["val"]["rf"], y_val
        )

        # 6. Ensemble Validation
        print("Validating Ensemble...")
        # Get predictions for validation set
        val_preds_mlp = predict_mlp(mlp_model, features["val"]["mlp"])
        val_preds_rf = predict_rf(rf_model, features["val"]["rf"])

        # Weighted Average
        w_rf = Config.ENSEMBLE_WEIGHT_RF
        w_mlp = Config.ENSEMBLE_WEIGHT_MLP

        val_preds_ensemble = (w_rf * val_preds_rf) + (w_mlp * val_preds_mlp)

        # Calculate and print metric
        val_auc = roc_auc_score(y_val, val_preds_ensemble)
        print(f"Ensemble Validation AUC: {val_auc}")

        # 7. Test Inference & Submission
        print("Generating Test Predictions...")
        test_preds_mlp = predict_mlp(mlp_model, features["test"]["mlp"])
        test_preds_rf = predict_rf(rf_model, features["test"]["rf"])

        test_preds_ensemble = (w_rf * test_preds_rf) + (w_mlp * test_preds_mlp)

        submission = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_preds_ensemble}
        )

        submission_path = Config.SUBMISSION_PATH
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
