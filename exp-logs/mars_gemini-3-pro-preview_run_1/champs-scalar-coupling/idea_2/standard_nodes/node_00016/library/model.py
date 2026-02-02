import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from library.config import XGB_PARAMS, COUPLING_TYPES, WORKING_DIR
from library.metrics import calculate_log_mae

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class StratifiedEnsemble:
    """
    A stratified ensemble of XGBoost models, where a separate regressor is trained
    for each scalar coupling type (e.g., 1JHC, 2JHH, etc.).
    """

    def __init__(self, params=None):
        """
        Initialize the ensemble with XGBoost parameters.

        Args:
            params (dict, optional): Hyperparameters for XGBoost.
                                     Defaults to library.config.XGB_PARAMS.
        """
        self.params = params if params else XGB_PARAMS.copy()
        self.models = {}
        self.feature_cols = {}  # Stores the list of feature names used for each type
        self.type_metrics = {}

        # Columns that must be excluded from the feature set
        self.exclude_cols = {
            "id",
            "molecule_name",
            "type",
            "atom_0",
            "atom_1",
            "scalar_coupling_constant",
            "file_path",
            "fc",
            "sd",
            "pso",
            "dso",  # Contribution terms if present
        }

    def _get_features(self, df):
        """
        Selects numeric features and drops excluded columns.

        Args:
            df (pd.DataFrame): The dataframe containing all columns.

        Returns:
            pd.DataFrame: Dataframe containing only the numeric feature columns.
        """
        # Identify candidate columns (exclude metadata/targets)
        candidates = [c for c in df.columns if c not in self.exclude_cols]

        # Select only numeric columns from candidates
        # This automatically drops string columns like atom types if they weren't in exclude_cols
        X = df[candidates].select_dtypes(include=[np.number])
        return X

    def fit(
        self,
        X_train,
        y_train,
        groups_train,
        X_val=None,
        y_val=None,
        groups_val=None,
        verbose=False,
    ):
        """
        Trains a separate XGBoost model for each coupling type found in groups_train.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            groups_train (pd.Series): Coupling types for training data.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.Series, optional): Validation targets.
            groups_val (pd.Series, optional): Validation coupling types.
            verbose (bool): Whether to print XGBoost training logs.
        """
        unique_types = sorted(groups_train.unique())

        print(f"Training Stratified Ensemble on {len(unique_types)} coupling types...")

        overall_preds = []
        overall_truth = []
        overall_types = []

        for coupling_type in unique_types:
            # Skip types not defined in our configuration (sanity check)
            if coupling_type not in COUPLING_TYPES:
                continue

            print(f"\n--- Training Model for Type: {coupling_type} ---")

            # 1. Slice Training Data for this type
            mask_train = groups_train == coupling_type
            X_t = self._get_features(X_train[mask_train])
            y_t = y_train[mask_train]

            # Save the feature names used for this model to ensure consistency during inference
            self.feature_cols[coupling_type] = X_t.columns.tolist()

            # 2. Slice Validation Data (if available)
            eval_set = []
            X_v = None
            y_v = None

            if X_val is not None and y_val is not None and groups_val is not None:
                mask_val = groups_val == coupling_type
                if mask_val.sum() > 0:
                    # Select rows and ensure columns match training features exactly
                    X_v = X_val[mask_val][self.feature_cols[coupling_type]]
                    y_v = y_val[mask_val]
                    eval_set = [(X_v, y_v)]

            # 3. Initialize and Train Model
            # Update params with early_stopping_rounds for XGBoost 2.0+ compatibility
            train_params = self.params.copy()
            if eval_set:
                # Increased patience for lower learning rate (Cite solution_lesson_node_00008)
                train_params["early_stopping_rounds"] = 500

            model = xgb.XGBRegressor(**train_params)

            model.fit(X_t, y_t, eval_set=eval_set, verbose=verbose)

            self.models[coupling_type] = model

            # 4. Evaluate on Validation Slice
            if eval_set and X_v is not None:
                y_pred_val = model.predict(X_v)

                # Calculate metrics for this type
                mae = np.mean(np.abs(y_v - y_pred_val))
                log_mae = np.log(mae)
                self.type_metrics[coupling_type] = log_mae

                print(
                    f"Type {coupling_type} - Validation MAE: {mae:.9f}, LogMAE: {log_mae:.9f}"
                )
                print(f"Best Iteration: {model.best_iteration}")

                # Accumulate for global metric calculation
                overall_preds.extend(y_pred_val)
                overall_truth.extend(y_v)
                overall_types.extend([coupling_type] * len(y_v))

        # 5. Calculate and Print Global Metric
        if overall_preds:
            final_score = calculate_log_mae(overall_truth, overall_preds, overall_types)
            print("\n" + "=" * 40)
            print(f"Overall Validation LogMAE: {final_score:.9f}")
            print("=" * 40 + "\n")

    def predict(self, X, groups):
        """
        Predicts target values using the appropriate model for each sample's coupling type.

        Args:
            X (pd.DataFrame): Input features.
            groups (pd.Series): Coupling types corresponding to X.

        Returns:
            pd.Series: Predicted values aligned with the input index.
        """
        # Initialize result series with NaNs, using the index from X
        predictions = pd.Series(index=X.index, dtype=float)
        predictions[:] = np.nan

        unique_types = groups.unique()

        for coupling_type in unique_types:
            if coupling_type not in self.models:
                print(f"Warning: No model found for type {coupling_type}. Skipping.")
                continue

            mask = groups == coupling_type
            if mask.sum() == 0:
                continue

            # Retrieve the specific feature columns expected by this type's model
            features_needed = self.feature_cols[coupling_type]

            # Filter input rows and columns
            # This ensures the model receives exactly the features it was trained on
            X_subset = X.loc[mask]

            # Check for missing columns (should not happen if pipeline is consistent)
            missing_cols = [c for c in features_needed if c not in X_subset.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing features for type {coupling_type}: {missing_cols}"
                )

            X_filtered = X_subset[features_needed]

            # Predict
            model = self.models[coupling_type]
            preds = model.predict(X_filtered)

            # Assign predictions to the correct indices
            predictions.loc[mask] = preds

        return predictions

    def save_models(self, directory=WORKING_DIR):
        """
        Saves the ensemble models and metadata to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "stratified_ensemble.pkl")

        payload = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "type_metrics": self.type_metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Ensemble saved to {path}")

    def load_models(self, directory=WORKING_DIR):
        """
        Loads the ensemble models from the specified directory.
        """
        path = os.path.join(directory, "stratified_ensemble.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models = data["models"]
            self.feature_cols = data["feature_cols"]
            self.type_metrics = data.get("type_metrics", {})
        print(f"Ensemble loaded from {path}")
