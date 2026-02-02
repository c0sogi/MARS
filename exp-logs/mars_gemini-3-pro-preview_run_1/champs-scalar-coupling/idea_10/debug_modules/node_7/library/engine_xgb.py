import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from library import config


class StratifiedEnsemble:
    """
    Neuro-Symbolic Stratified Ensemble.
    Trains and predicts using separate XGBoost models for each coupling type,
    applying specific feature pruning per stratum.
    """

    def __init__(self):
        self.models = {}
        self.feature_masks = {}
        self.types = config.COUPLING_TYPES
        self.model_dir = config.MODEL_DIR

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_candidate_features(self, df):
        """
        Returns a list of all potential feature columns, excluding metadata and targets.
        """
        exclude_cols = {
            "id",
            "molecule_name",
            "type",
            "scalar_coupling_constant",
            "file_path",
        }
        return [c for c in df.columns if c not in exclude_cols]

    def _prune_features(self, df, type_name):
        """
        Identifies features with variance higher than the threshold for this specific group.
        Saves the mask to disk.
        """
        candidate_features = self._get_candidate_features(df)

        # Calculate variance for candidates
        # numeric_only=True ensures we don't crash on accidental object columns,
        # though pipeline should be clean.
        variances = df[candidate_features].var(numeric_only=True)

        # Select features exceeding threshold
        active_features = variances[
            variances > config.PRUNING_VARIANCE_THRESHOLD
        ].index.tolist()

        # Save to file
        mask_path = os.path.join(self.model_dir, f"features_{type_name}.json")
        with open(mask_path, "w") as f:
            json.dump(active_features, f)

        self.feature_masks[type_name] = active_features

        return active_features, candidate_features

    def fit(self, df_train, df_val):
        """
        Trains the stratified ensemble.

        Args:
            df_train (pd.DataFrame): Training data containing features and target.
            df_val (pd.DataFrame): Validation data containing features and target.
        """
        print(
            f"Initializing Stratified XGBoost Training on {config.GNN_PARAMS['device']}..."
        )
        print(f"Global Configuration: {config.XGB_PARAMS}")

        maes = []

        for t in self.types:
            print(f"\n{'='*10} Processing Coupling Type: {t} {'='*10}")

            # 1. Stratify Data
            train_subset = df_train[df_train["type"] == t].reset_index(drop=True)
            val_subset = df_val[df_val["type"] == t].reset_index(drop=True)

            if len(train_subset) == 0:
                print(f"Warning: No training samples found for type {t}. Skipping.")
                continue

            # 2. Stratified Feature Pruning
            active_feats, all_candidates = self._prune_features(train_subset, t)
            n_dropped = len(all_candidates) - len(active_feats)
            print(
                f"Feature Pruning: Kept {len(active_feats)} / {len(all_candidates)} features (Dropped {n_dropped})"
            )

            if len(active_feats) == 0:
                print(
                    f"Error: No active features remaining for type {t} after pruning."
                )
                continue

            # 3. Prepare Matrices
            X_train = train_subset[active_feats]
            y_train = train_subset["scalar_coupling_constant"]

            X_val = val_subset[active_feats]
            y_val = val_subset["scalar_coupling_constant"]

            # 4. Initialize and Train Model
            model = xgb.XGBRegressor(**config.XGB_PARAMS)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
                verbose=False,  # Silent training
            )

            # 5. Evaluation and Logging
            # Calculate metric manually to ensure full precision display
            if hasattr(model, "best_iteration"):
                # Use best iteration for prediction if early stopping occurred
                # Note: predict() uses best_iteration by default if early stopping was used
                preds_val = model.predict(X_val)
            else:
                preds_val = model.predict(X_val)

            mae = np.mean(np.abs(y_val - preds_val))
            log_mae = np.log(mae)

            print(f"Type {t} Results:")
            print(f"  Best Iteration: {model.best_iteration}")
            print(f"  Validation MAE: {mae}")
            print(f"  Log MAE: {log_mae}")

            maes.append(log_mae)

            # 6. Save Model
            model_path = os.path.join(self.model_dir, f"xgb_{t}.json")
            model.save_model(model_path)
            self.models[t] = model

        # Global Metric
        if maes:
            avg_log_mae = np.mean(maes)
            print(f"\n{'='*30}")
            print(f"Training Complete.")
            print(f"Mean Log MAE across all types: {avg_log_mae}")
            print(f"{'='*30}")
        else:
            print("\nWarning: No models were trained.")

    def predict(self, df_test):
        """
        Generates predictions for the test set using the stratified models.

        Args:
            df_test (pd.DataFrame): Test data features.

        Returns:
            pd.DataFrame: DataFrame with 'id' and 'scalar_coupling_constant'.
        """
        print("\nStarting Stratified Prediction...")
        results = []

        for t in self.types:
            # 1. Stratify
            test_subset = df_test[df_test["type"] == t].copy()

            if len(test_subset) == 0:
                continue

            # 2. Load Resources (Model and Mask)
            # Load Feature Mask
            if t not in self.feature_masks:
                mask_path = os.path.join(self.model_dir, f"features_{t}.json")
                if os.path.exists(mask_path):
                    with open(mask_path, "r") as f:
                        self.feature_masks[t] = json.load(f)
                else:
                    print(
                        f"Warning: Feature mask for {t} not found. Skipping prediction for this type."
                    )
                    continue

            active_feats = self.feature_masks[t]

            # Load Model
            if t not in self.models:
                model_path = os.path.join(self.model_dir, f"xgb_{t}.json")
                if os.path.exists(model_path):
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                    # Re-apply params to ensure GPU usage if applicable during inference
                    model.set_params(**config.XGB_PARAMS)
                    self.models[t] = model
                else:
                    print(
                        f"Warning: Model for {t} not found. Skipping prediction for this type."
                    )
                    continue

            # 3. Predict
            # Ensure columns align
            try:
                X_test = test_subset[active_feats]
            except KeyError as e:
                print(f"Error: Missing features for type {t} in test set: {e}")
                continue

            preds = self.models[t].predict(X_test)

            # 4. Store Results
            subset_res = pd.DataFrame(
                {"id": test_subset["id"], "scalar_coupling_constant": preds}
            )
            results.append(subset_res)

        # 5. Aggregate
        if not results:
            print("Warning: No predictions generated.")
            return pd.DataFrame(columns=["id", "scalar_coupling_constant"])

        final_df = pd.concat(results, axis=0)

        # Sort by ID to match submission format requirements
        final_df = final_df.sort_values("id").reset_index(drop=True)

        return final_df
