import numpy as np
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from library.utils import set_seed, score_predictions


class LGBMWrapper:
    """
    Wrapper for LightGBM Classifier with Early Stopping.
    """

    def __init__(
        self,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        **kwargs,
    ):
        set_seed(random_state)
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        # Update with any additional kwargs passed to init
        self.params.update(kwargs)
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        Fits the LightGBM model. Uses early stopping if validation data is provided.
        """
        if X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Suppress printing
            ]
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )
            # Calculate and print validation score
            val_preds = self.model.predict_proba(X_val)
            score = score_predictions(y_val, val_preds)
            print(f"LGBM Validation Log Loss: {score}")
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LDAWrapper:
    """
    Wrapper for Linear Discriminant Analysis with Automatic Shrinkage (Ledoit-Wolf).
    """

    def __init__(self, solver="lsqr", shrinkage="auto", random_state=42):
        set_seed(random_state)
        # LDA with lsqr solver supports shrinkage
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict_proba(X_val)
            score = score_predictions(y_val, val_preds)
            print(f"LDA Validation Log Loss: {score}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class SVMWrapper:
    """
    Wrapper for Support Vector Machine (RBF Kernel) with Probability Calibration.
    Uses CalibratedClassifierCV (Platt Scaling) to generate probabilities.
    """

    def __init__(self, C=10.0, gamma="scale", cv=5, random_state=42):
        set_seed(random_state)
        # Base estimator: SVC with RBF kernel
        # C=10.0 is often better for high-dimensional features than default 1.0
        # probability=False because calibration handles it.
        base_svc = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            random_state=random_state,
            class_weight="balanced",
        )

        # Calibrated Classifier
        # method='sigmoid' implements Platt Scaling
        # cv determines the cross-validation strategy for calibration
        self.model = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=cv)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict_proba(X_val)
            score = score_predictions(y_val, val_preds)
            print(f"SVM (Calibrated) Validation Log Loss: {score}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)
