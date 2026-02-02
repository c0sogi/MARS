import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.utils import enu_to_ecef, ecef_to_geodetic


class ResidualLGBMEnsemble:
    """
    Ensemble of LightGBM models to predict ENU residuals from physics-based features.
    Uses GroupKFold cross-validation and robust median aggregation for inference.
    """

    def __init__(self, output_dir="./working/idea_15/models", n_folds=5, seed=42):
        """
        Initialize the ensemble.

        Args:
            output_dir (str): Directory to save trained models.
            n_folds (int): Number of cross-validation folds.
            seed (int): Random seed for reproducibility.
        """
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.seed = seed
        self.models = {}  # Structure: {target_col: {fold_idx: model_object}}
        self.feature_cols = []
        self.target_cols = []

        os.makedirs(self.output_dir, exist_ok=True)

    def train_group_kfold(
        self,
        train_df,
        feature_cols,
        target_cols,
        params=None,
        n_estimators=1000,
        early_stopping_rounds=50,
    ):
        """
        Train LightGBM models using GroupKFold cross-validation.

        Args:
            train_df (pd.DataFrame): Training data containing features, targets, and 'drive_id'.
            feature_cols (list): List of feature column names.
            target_cols (list): List of target column names (e.g., ['target_E', 'target_N', 'target_U']).
            params (dict, optional): LightGBM hyperparameters.
            n_estimators (int): Maximum number of boosting iterations.
            early_stopping_rounds (int): Rounds for early stopping.

        Returns:
            dict: Dictionary of validation scores (MAE) per target.
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        # Default parameters if none provided
        if params is None:
            params = {
                "objective": "regression_l1",  # MAE loss
                "metric": "mae",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
                "n_jobs": -1,
                "verbose": -1,
                "seed": self.seed,
            }
        else:
            # Ensure seed is set
            params["seed"] = self.seed
            params["verbose"] = -1

        # Initialize model storage
        for target in target_cols:
            self.models[target] = {}

        gkf = GroupKFold(n_splits=self.n_folds)
        groups = train_df["drive_id"]

        scores = {target: [] for target in target_cols}

        print(f"Starting training with {self.n_folds} folds...")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")

            X_train = train_df.iloc[train_idx][feature_cols]
            X_val = train_df.iloc[val_idx][feature_cols]

            for target in target_cols:
                y_train = train_df.iloc[train_idx][target]
                y_val = train_df.iloc[val_idx][target]

                # Create dataset
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

                # Train
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Suppress log output
                ]

                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=n_estimators,
                    valid_sets=[lgb_train, lgb_val],
                    valid_names=["train", "valid"],
                    callbacks=callbacks,
                )

                # Evaluate
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                mae = mean_absolute_error(y_val, y_pred)
                scores[target].append(mae)

                # Store model
                self.models[target][fold] = model

                # Save model to disk
                model_path = os.path.join(
                    self.output_dir, f"lgbm_{target}_fold_{fold}.txt"
                )
                model.save_model(model_path)

                print(
                    f"Target: {target} | Val MAE: {mae:.8f} | Best Iter: {model.best_iteration}"
                )

        # Aggregate scores
        final_scores = {}
        print("\n--- Cross-Validation Results ---")
        for target, mae_list in scores.items():
            mean_mae = np.mean(mae_list)
            std_mae = np.std(mae_list)
            final_scores[target] = mean_mae
            print(f"{target}: Mean MAE = {mean_mae:.8f} (+/- {std_mae:.8f})")

        return final_scores

    def predict_ensemble(
        self,
        test_df,
        feature_cols,
        target_cols,
        submission_path="./submission/submission.csv",
    ):
        """
        Generate predictions using the trained ensemble with robust median aggregation.
        Converts predicted ENU residuals back to Geodetic coordinates.

        Args:
            test_df (pd.DataFrame): Test data containing features and WLS baseline.
            feature_cols (list): List of feature column names.
            target_cols (list): List of target column names (E, N, U).
            submission_path (str): Path to save the submission file.

        Returns:
            pd.DataFrame: DataFrame with predictions formatted for submission.
        """
        if not self.models:
            raise ValueError("Models not trained or loaded.")

        print("\nStarting ensemble prediction...")

        X_test = test_df[feature_cols]

        # Dictionary to store aggregated predictions for each target (E, N, U)
        final_preds_enu = {}

        for target in target_cols:
            if target not in self.models:
                raise ValueError(f"No models found for target {target}")

            # Collect predictions from all folds
            fold_preds = []
            for fold in range(self.n_folds):
                if fold not in self.models[target]:
                    continue

                model = self.models[target][fold]
                preds = model.predict(X_test, num_iteration=model.best_iteration)
                fold_preds.append(preds)

            # Stack predictions: (n_samples, n_folds)
            fold_preds = np.column_stack(fold_preds)

            # Robust Aggregation: Pixel-wise Median
            # This ignores outliers from specific folds
            median_pred = np.median(fold_preds, axis=1)
            final_preds_enu[target] = median_pred

        # Reconstruct Absolute Positions
        # Target definition: Target = GT - WLS (in ENU)
        # Prediction: Pred_Delta = Model(Features)
        # Reconstructed GT (ENU) = Pred_Delta (relative to WLS)
        # We need to convert this ENU offset back to ECEF, then Geodetic

        # WLS Baseline
        wls_x = test_df["WlsPositionXEcefMeters"].values
        wls_y = test_df["WlsPositionYEcefMeters"].values
        wls_z = test_df["WlsPositionZEcefMeters"].values

        # Predicted ENU offsets
        # Assuming target_cols are ['target_E', 'target_N', 'target_U']
        # We map them correctly based on names
        pred_e = final_preds_enu.get("target_E", np.zeros_like(wls_x))
        pred_n = final_preds_enu.get("target_N", np.zeros_like(wls_x))
        pred_u = final_preds_enu.get("target_U", np.zeros_like(wls_x))

        # Convert ENU offsets + WLS Reference -> ECEF
        pred_x, pred_y, pred_z = enu_to_ecef(
            pred_e, pred_n, pred_u, wls_x, wls_y, wls_z
        )

        # Convert ECEF -> Geodetic (Lat, Lon)
        pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "LatitudeDegrees": pred_lat,
                "LongitudeDegrees": pred_lon,
            }
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        # Save
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        return submission_df
