import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from library import config


def build_linear_branch():
    """
    Constructs the Discriminative Linear Branch.
    Architecture: StandardScaler -> LogisticRegressionCV
    """
    # Logistic Regression with broad grid search, L2 regularization, and log-loss optimization
    clf = LogisticRegressionCV(
        Cs=config.LR_CS_GRID,
        cv=config.LR_CV_FOLDS,
        scoring=config.LR_SCORING,
        solver=config.LR_SOLVER,
        max_iter=config.LR_MAX_ITER,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        multi_class="multinomial",
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    return pipeline


def build_generative_branch():
    """
    Constructs the Generative Linear Branch.
    Architecture: StandardScaler -> LDA (with Ledoit-Wolf shrinkage)
    """
    # LDA with Ledoit-Wolf shrinkage for robustness in high dimensions
    clf = LinearDiscriminantAnalysis(
        solver=config.LDA_SOLVER, shrinkage=config.LDA_SHRINKAGE
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    return pipeline


def build_kernel_branch():
    """
    Constructs the Discriminative Kernel Branch.
    Architecture: StandardScaler -> PCA -> Nystroem -> LogisticRegressionCV
    """
    # 1. PCA to densify features and reduce noise before kernel mapping
    pca = PCA(n_components=config.PCA_VARIANCE, random_state=config.RANDOM_STATE)

    # 2. Nystroem approximation of RBF kernel
    nystroem = Nystroem(
        n_components=config.NYSTROEM_COMPONENTS,
        gamma=config.NYSTROEM_GAMMA,
        random_state=config.RANDOM_STATE,
    )

    # 3. Logistic Regression solver (same robust config as linear branch)
    clf = LogisticRegressionCV(
        Cs=config.LR_CS_GRID,
        cv=config.LR_CV_FOLDS,
        scoring=config.LR_SCORING,
        solver=config.LR_SOLVER,
        max_iter=config.LR_MAX_ITER,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        multi_class="multinomial",
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("nystroem", nystroem),
            ("classifier", clf),
        ]
    )

    return pipeline


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    A Soft-Voting Ensemble that aggregates predictions from Linear, Generative,
    and Kernel branches.
    """

    def __init__(self):
        self.estimators_ = [
            ("linear", build_linear_branch()),
            ("generative", build_generative_branch()),
            ("kernel", build_kernel_branch()),
        ]
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits all constituent branches on the provided training data.
        """
        # Record classes from the target vector
        self.classes_ = np.unique(y)

        for name, estimator in self.estimators_:
            print(f"Training {name} branch...")
            estimator.fit(X, y)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities by averaging the probabilities
        from all branches (Soft Voting).
        """
        probas = []
        for name, estimator in self.estimators_:
            p = estimator.predict_proba(X)
            probas.append(p)

        # Average the probabilities across estimators
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X):
        """
        Predicts class labels based on the averaged probabilities.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]
