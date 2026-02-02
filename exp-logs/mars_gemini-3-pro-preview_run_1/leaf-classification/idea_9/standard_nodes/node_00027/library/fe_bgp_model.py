import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import log_loss

from library import config
from library import data_loader


class FisherEmbeddedBGP:
    """
    Fisher-Embedded Bayesian Gaussian Process (FE-BGP).

    Architecture:
    1. Filter: Supervised Manifold Learning via Linear Discriminant Analysis (LDA).
       - Preprocessing: Yeo-Johnson Power Transform + Standard Scaling.
       - Projection: Projects data into the (C-1) dimensional Fisher Subspace.
    2. Refine: Bayesian Calibration via Gaussian Process Classifier (GPC).
       - Kernel: Linear (DotProduct) + Noise (WhiteKernel).
       - Inference: Probabilistic classification with intrinsic uncertainty modeling.
    """

    def __init__(self, random_state=config.RANDOM_SEED):
        self.random_state = random_state
        self.classes_ = None
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Kernel Definition:
        # DotProduct captures the linear trends preserved by LDA.
        # WhiteKernel explicitly models global noise/aleatoric uncertainty.
        # Hyperparameters (sigma_0, noise_level) are optimized during fit.
        kernel = DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=1.0)

        steps = [
            # Step 1: Enforce Multivariate Gaussianity
            # Required for optimal LDA separation.
            ("pt", PowerTransformer(method="yeo-johnson", standardize=False)),
            # Step 2: Numerical Stability
            ("scaler", StandardScaler()),
            # Step 3: Supervised Dimensionality Reduction (The "Filter")
            # Projects 192 features -> 98 discriminative components.
            # Solver 'lsqr' with 'auto' shrinkage (Ledoit-Wolf) handles multicollinearity.
            (
                "lda",
                LinearDiscriminantAnalysis(
                    n_components=config.N_LDA_COMPONENTS,
                    solver="eigen",
                    shrinkage="auto",
                ),
            ),
            # Step 4: Probabilistic Classification (The "Refine")
            # GPC provides calibrated probabilities.
            (
                "gpc",
                GaussianProcessClassifier(
                    kernel=kernel,
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=1,  # Restart to avoid local optima
                    random_state=self.random_state,
                    n_jobs=-1,  # Parallelize prediction
                    copy_X_train=False,  # Save memory
                    multi_class="one_vs_rest",  # Fits one binary GP per class
                ),
            ),
        ]

        return Pipeline(steps)

    def fit(self, X, y):
        """
        Fits the FE-BGP pipeline to the training data.
        """
        self.pipeline.fit(X, y)
        # Capture classes from the pipeline (GPC step)
        if hasattr(self.pipeline.named_steps["gpc"], "classes_"):
            self.classes_ = self.pipeline.named_steps["gpc"].classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """
        Returns probability estimates for the test data.
        """
        return self.pipeline.predict_proba(X)

    def get_log_marginal_likelihood(self):
        """
        Attempts to retrieve the Log Marginal Likelihood from the GPC.
        Note: For multi-class (one-vs-rest), this might not be a single scalar
        exposed directly on the main object in all sklearn versions.
        """
        gpc = self.pipeline.named_steps["gpc"]
        if hasattr(gpc, "log_marginal_likelihood_value_"):
            return gpc.log_marginal_likelihood_value_
        return None


def run_fe_bgp_pipeline(load_cached_data=True, sample_size=None):
    """
    Executes the full FE-BGP workflow:
    1. Data Loading
    2. Validation (Hold-out evaluation)
    3. Global Training (Full dataset)
    4. Submission Generation

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        sample_size (int, optional): Limit dataset size for debugging.
    """
    print("=" * 40)
    print("STARTING FE-BGP PIPELINE")
    print("=" * 40)

    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading Data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = data_loader.load_data(
        load_cached_data=load_cached_data, sample_size=sample_size
    )
    print(
        f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}"
    )

    # 2. Validation Step
    # -------------------------------------------------------------------------
    print("\n[2/4] Validation Step (Train on Train, Evaluate on Val)...")
    val_model = FisherEmbeddedBGP(random_state=config.RANDOM_SEED)
    val_model.fit(X_train, y_train)

    val_probs = val_model.predict_proba(X_val)
    val_log_loss = log_loss(y_val, val_probs, labels=val_model.classes_)

    print(f"Validation Multi-class Log Loss: {val_log_loss}")

    # 3. Global Training Step
    # -------------------------------------------------------------------------
    print("\n[3/4] Global Training Step (Train on Train + Val)...")
    # Merge datasets to maximize signal for covariance estimation
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    global_model = FisherEmbeddedBGP(random_state=config.RANDOM_SEED)
    global_model.fit(X_full, y_full)

    lml = global_model.get_log_marginal_likelihood()
    if lml is not None:
        print(f"Final Model Log Marginal Likelihood: {lml}")
    else:
        print("Final Model Fitted (LML not directly exposed for OVR GPC).")

    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Submission...")
    test_probs = global_model.predict_proba(X_test)

    # Construct submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=global_model.classes_)
    submission_df.insert(0, config.ID_COL, test_ids)

    # Save
    print(f"Saving submission to: {config.SUBMISSION_FILE_PATH}")
    submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print("Submission saved successfully.")
    print("=" * 40)
