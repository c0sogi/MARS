import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library.config import GASEConfig
from library.feature_eng import FeatureAssembler, prepare_stratified_data


class StratifiedEnsemble:
    """
    Manages a stratified ensemble of XGBoost models, one for each scalar coupling type.
    Integrates geometric features and learned embeddings for prediction.
    """

    def __init__(self):
        self.models_dir = GASEConfig.XGB_MODELS_DIR
        self.submission_path = GASEConfig.SUBMISSION_PATH
        self.coupling_types = GASEConfig.COUPLING_TYPES

        # Columns to exclude from features (Metadata, Targets, Raw Coordinates, Strings)
        self.exclude_cols = {
            "id",
            "molecule_name",
            "type",
            "scalar_coupling_constant",
            "file_path",
            "atom_0",
            "atom_1",
            "x0",
            "y0",
            "z0",
            "x1",
            "y1",
            "z1",
            "dx",
            "dy",
            "dz",  # Exclude raw vector components to enforce rotation invariance, relying on dist/angles
        }

    def _get_feature_cols(self, df):
        """
        Identifies feature columns by excluding non-feature columns.
        """
        return [c for c in df.columns if c not in self.exclude_cols]

    def _get_model_path(self, coupling_type):
        """Returns the file path for a saved model of a specific type."""
        return os.path.join(self.models_dir, f"xgb_{coupling_type}.pkl")

    def train_ensemble(self, load_cached_data=True):
        """
        Trains separate XGBoost models for each coupling type.

        Args:
            load_cached_data (bool): Whether to use cached processed data.
        """
        print("Initializing FeatureAssembler for Training...")
        assembler = FeatureAssembler()

        # 1. Load and Assemble Data
        print("Loading Train Data...")
        df_train = assembler.assemble_data("train", load_cached_data=load_cached_data)

        print("Loading Validation Data...")
        df_val = assembler.assemble_data("val", load_cached_data=load_cached_data)

        # 2. Stratify Data
        print("Stratifying data by coupling type...")
        train_groups = prepare_stratified_data(df_train)
        val_groups = prepare_stratified_data(df_val)

        # 3. Training Loop
        metrics = {}

        print(f"Starting Stratified Training for {len(self.coupling_types)} types...")

        for c_type in self.coupling_types:
            if c_type not in train_groups:
                print(f"Warning: No training data for type {c_type}. Skipping.")
                continue

            print(f"\n[{c_type}] Preparing data...")
            train_subset = train_groups[c_type]
            val_subset = val_groups.get(c_type, pd.DataFrame())

            if val_subset.empty:
                print(f"Warning: No validation data for type {c_type}.")

            # Select Features
            features = self._get_feature_cols(train_subset)
            X_train = train_subset[features]
            y_train = train_subset["scalar_coupling_constant"]

            X_val = val_subset[features]
            y_val = val_subset["scalar_coupling_constant"]

            # Configure Model
            params = GASEConfig.get_xgb_params(c_type)
            model = xgb.XGBRegressor(**params)

            # Train
            print(f"[{c_type}] Training XGBoost (Features: {len(features)})...")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Evaluate
            # Best iteration is automatically used if early stopping is enabled
            preds_val = model.predict(X_val)
            mae = np.mean(np.abs(y_val - preds_val))
            metrics[c_type] = mae

            print(f"[{c_type}] Validation MAE: {mae}")

            # Save Model
            save_path = self._get_model_path(c_type)
            joblib.dump(model, save_path)
            print(f"[{c_type}] Model saved to {save_path}")

        # 4. Overall Metric Calculation
        print("\n" + "=" * 40)
        print("FINAL VALIDATION METRICS")
        print("=" * 40)

        log_maes = []
        for c_type, mae in metrics.items():
            print(f"Type {c_type}: MAE = {mae}")
            # Metric is Log of MAE. We use natural log as is standard unless specified otherwise.
            # However, competition metric is often log(MAE).
            # Note: log(0) is undefined, but MAE shouldn't be 0.
            log_maes.append(np.log(mae))

        if log_maes:
            final_score = np.mean(log_maes)
            print("-" * 40)
            print(f"Competition Metric (Mean Log MAE): {final_score}")
        else:
            print("No metrics computed.")
        print("=" * 40)

    def predict_ensemble(self, load_cached_data=True):
        """
        Generates predictions for the test set using the trained ensemble.
        Saves predictions to submission.csv.

        Args:
            load_cached_data (bool): Whether to use cached processed data.
        """
        print("Initializing FeatureAssembler for Inference...")
        assembler = FeatureAssembler()

        # 1. Load Test Data
        print("Loading Test Data...")
        df_test = assembler.assemble_data("test", load_cached_data=load_cached_data)

        # 2. Stratify
        test_groups = prepare_stratified_data(df_test)

        # 3. Prediction Loop
        results = []

        print("Starting Inference...")
        for c_type in self.coupling_types:
            if c_type not in test_groups:
                continue

            subset = test_groups[c_type]
            features = self._get_feature_cols(subset)
            X_test = subset[features]
            ids = subset["id"].values

            # Load Model
            model_path = self._get_model_path(c_type)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model for {c_type} not found at {model_path}. Train first."
                )

            model = joblib.load(model_path)

            # Predict
            preds = model.predict(X_test)

            # Store results
            chunk_df = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})
            results.append(chunk_df)
            print(f"[{c_type}] Predicted {len(chunk_df)} samples.")

        # 4. Aggregate and Save
        if not results:
            raise RuntimeError("No predictions generated.")

        submission_df = pd.concat(results, axis=0)

        # Sort by ID as per sample submission
        submission_df = submission_df.sort_values("id")

        # Save
        print(f"Saving submission to {self.submission_path}...")
        submission_df.to_csv(self.submission_path, index=False)
        print("Submission generation complete.")
