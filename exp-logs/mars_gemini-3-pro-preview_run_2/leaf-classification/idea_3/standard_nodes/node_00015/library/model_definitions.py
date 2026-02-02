import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def get_logistic_regression(random_state=42, n_jobs=-1):
    """
    Returns a Logistic Regression model configured with Cross-Validation
    to select the best regularization parameter C.

    Configuration:
    - Penalty: L2 (Ridge)
    - Solver: lbfgs (Efficient for multiclass)
    - C Grid: Logspace from 1e-2 to 1e6 (Focus on weak regularization due to high SNR)
    - Multiclass: Multinomial
    """
    # Define a grid focusing on higher C values (weaker regularization)
    # as previous experiments indicated high signal quality.
    Cs = np.logspace(-2, 6, 10)

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=3,
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=5000,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return clf


def get_lda():
    """
    Returns a Linear Discriminant Analysis model configured with Ledoit-Wolf shrinkage.

    Configuration:
    - Solver: lsqr (Least squares solution, supports shrinkage)
    - Shrinkage: auto (Uses Ledoit-Wolf lemma for optimal covariance estimation)
    """
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return clf


def get_gpc(random_state=42, n_jobs=-1):
    """
    Returns a Gaussian Process Classifier configured with an RBF kernel.

    Configuration:
    - Kernel: 1.0 * RBF(1.0) (Initial hyperparameters, optimized during fit)
    - Optimizer: fmin_l_bfgs_b (Default, maximizes Log-Marginal-Likelihood)
    - copy_X_train: False (Memory efficiency)
    """
    # Initial kernel: ConstantKernel * RBF
    # The optimizer will tune the constant value and the length scale.
    kernel = 1.0 * RBF(length_scale=1.0)

    clf = GaussianProcessClassifier(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",  # Default optimizer for kernel hyperparameters
        n_restarts_optimizer=0,  # 0 restarts is usually sufficient if initialization is reasonable
        max_iter_predict=100,
        random_state=random_state,
        n_jobs=n_jobs,
        copy_X_train=False,  # Save memory, input is already processed/copied
    )
    return clf
