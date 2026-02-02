import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import copy
from library import config


class EnsembleTrainer:
    """
    Manages the training and validation of an XGBoost ensemble.
    """

    def __init__(self, params=None):
        self.params = (
            copy.deepcopy(params) if params else copy.deepcopy(config.XGB_PARAMS)
        )
        self.le = LabelEncoder()
        self.models = []
        self.oof_preds = None
        self.feature_names = None

    def train_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """
        Trains a single XGBoost model for a specific fold.
        """
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Watchlist for monitoring
        watchlist = [(dtrain, "train"), (dval, "val")]

        print(f"Fold {fold_idx + 1}: Training started...")

        # Train model
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 3000),
            evals=watchlist,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,  # Silent as requested, print metrics manually if needed or rely on final score
        )

        # Predict on validation set (best iteration is used automatically by xgb.train if early stopping occurred)
        # However, to be safe with DMatrix slicing, we predict on dval
        val_preds_prob = model.predict(dval)
        val_preds_cls = np.argmax(val_preds_prob, axis=1)

        acc = accuracy_score(y_val, val_preds_cls)
        print(f"Fold {fold_idx + 1} Accuracy: {acc}")

        return model, val_preds_prob

    def run_stratified_cv(self, df_train):
        """
        Executes Stratified K-Fold Cross-Validation.
        """
        # Prepare data
        # Drop Id and Target
        cols_to_drop = [config.ID_COL, config.TARGET_COL]
        X = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
        y = df_train[config.TARGET_COL]

        # Store feature names for consistency during inference
        self.feature_names = X.columns.tolist()

        # Encode Target
        # The dataset has classes like [1, 2, 3, 4, 6, 7]. XGBoost needs 0..N-1.
        y_encoded = self.le.fit_transform(y)

        # Update num_class in params based on actual unique classes found
        num_classes = len(self.le.classes_)
        self.params["num_class"] = num_classes

        # Initialize storage
        self.models = []
        self.oof_preds = np.zeros((len(df_train), num_classes), dtype=np.float32)

        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            # Train
            model, val_probs = self.train_fold(X_train, y_train, X_val, y_val, fold)

            # Store model
            self.models.append(model)

            # Store OOF predictions
            self.oof_preds[val_idx] = val_probs

            # Calculate fold score
            val_preds = np.argmax(val_probs, axis=1)
            score = accuracy_score(y_val, val_preds)
            fold_scores.append(score)

        # Calculate overall accuracy
        oof_class_preds = np.argmax(self.oof_preds, axis=1)
        overall_acc = accuracy_score(y_encoded, oof_class_preds)

        print(f"Overall CV Accuracy: {overall_acc}")

        return self.models, overall_acc

    def predict(self, df_test):
        """
        Generates predictions for the test set using the trained ensemble.
        Returns the original class labels (inverse transformed).
        """
        if not self.models:
            raise ValueError("No models trained. Run run_stratified_cv first.")

        # Prepare Test Data
        # Ensure ID column is dropped and columns match training features
        X_test = df_test.drop(columns=[config.ID_COL], errors="ignore")

        # Align columns just in case (though pipeline ensures consistency)
        if self.feature_names:
            X_test = X_test[self.feature_names]

        dtest = xgb.DMatrix(X_test)

        # Soft Voting
        avg_preds = np.zeros((len(df_test), len(self.le.classes_)), dtype=np.float32)

        for i, model in enumerate(self.models):
            preds = model.predict(dtest)
            avg_preds += preds

        avg_preds /= len(self.models)

        # Get Class Indices
        class_indices = np.argmax(avg_preds, axis=1)

        # Inverse Transform to original labels
        final_predictions = self.le.inverse_transform(class_indices)

        return final_predictions

    def generate_submission_file(self, df_test, predictions):
        """
        Creates the submission CSV file.
        """
        if config.ID_COL not in df_test.columns:
            raise ValueError(f"Test dataframe missing '{config.ID_COL}' column.")

        submission = pd.DataFrame(
            {config.ID_COL: df_test[config.ID_COL], config.TARGET_COL: predictions}
        )

        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        return submission
