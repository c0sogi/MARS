import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor, ElasticNet

from library.config import Config
from library.utils import score_function, save_results, log_metrics
from library.data_manager import DataManager


class DecoupledQuantileModel:
    """
    Implements the MIP-Enhanced Zonal-Quantile Pipeline.

    Components:
    1. Preprocessing: StandardScaler + PCA
    2. FVC Predictor: Linear Quantile Regressor (Median) with Time Interactions
    3. Uncertainty Predictor: ElasticNet Regressor on Absolute Residuals
    """

    def __init__(self):
        # Hyperparameters from Config
        self.n_components = Config.N_COMPONENTS
        self.qreg_alpha = Config.QREG_ALPHA
        self.qreg_solver = Config.QREG_SOLVER
        self.enet_alpha = Config.ENET_ALPHA
        self.enet_l1_ratio = Config.ENET_L1_RATIO
        self.seed = Config.SEED

        # Preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.seed)

        # Models
        self.fvc_model = QuantileRegressor(
            quantile=0.5, alpha=self.qreg_alpha, solver=self.qreg_solver
        )

        self.unc_model = ElasticNet(
            alpha=self.enet_alpha,
            l1_ratio=self.enet_l1_ratio,
            random_state=self.seed,
            max_iter=Config.MAX_ITER,
        )

    def _compute_relative_time(self, weeks, base_weeks):
        """
        Computes relative time (dt) from baseline.
        dt = current_week - baseline_week
        """
        return weeks - base_weeks

    def _build_fvc_features(self, X_pca, dt):
        """
        Constructs features for FVC prediction:
        [PCA_Components, dt, PCA_Components * dt]
        """
        # Ensure dt is (N, 1)
        dt = dt.reshape(-1, 1)

        # Interaction terms: Element-wise multiplication of each PCA comp with time
        # Broadcasting: (N, n_comps) * (N, 1) -> (N, n_comps)
        interactions = X_pca * dt

        # Concatenate: (N, n_comps) + (N, 1) + (N, n_comps)
        return np.hstack([X_pca, dt, interactions])

    def _build_unc_features(self, X_pca, dt):
        """
        Constructs features for Uncertainty prediction:
        [PCA_Components, |dt|]
        """
        dt = dt.reshape(-1, 1)
        horizon = np.abs(dt)

        return np.hstack([X_pca, horizon])

    def fit(self, X_static, weeks, y, base_weeks):
        """
        Trains the pipeline.
        """
        print("Fitting Preprocessors (Scaler + PCA)...")
        # 1. Preprocessing
        X_scaled = self.scaler.fit_transform(X_static)

        # Dynamically adjust PCA components if samples < n_components
        # Cite {debug_lesson_7}: Adapt Static Hyperparameters to Runtime Data Dimensions
        n_samples, n_features = X_scaled.shape
        max_components = min(n_samples, n_features)

        if self.n_components > max_components:
            print(
                f"Warning: Reducing PCA components from {self.n_components} to {max_components} due to data shape {X_scaled.shape}."
            )
            self.n_components = max_components
            self.pca = PCA(n_components=self.n_components, random_state=self.seed)

        X_pca = self.pca.fit_transform(X_scaled)

        # Calculate relative time
        dt = self._compute_relative_time(weeks, base_weeks)

        # 2. Train FVC Model
        print("Training FVC Quantile Regressor...")
        X_fvc = self._build_fvc_features(X_pca, dt)
        self.fvc_model.fit(X_fvc, y)

        # 3. Train Uncertainty Model
        print("Training Uncertainty ElasticNet...")
        # Generate predictions on training set to get residuals
        y_pred_train = self.fvc_model.predict(X_fvc)
        residuals = np.abs(y - y_pred_train)

        # Build uncertainty features
        X_unc = self._build_unc_features(X_pca, dt)
        self.unc_model.fit(X_unc, residuals)

        print("Training complete.")

    def predict(self, X_static, weeks, base_weeks):
        """
        Inference pipeline.
        Returns:
            y_pred (np.array): Predicted Median FVC
            sigma (np.array): Predicted Confidence (clipped)
        """
        # 1. Preprocessing
        X_scaled = self.scaler.transform(X_static)
        X_pca = self.pca.transform(X_scaled)

        dt = self._compute_relative_time(weeks, base_weeks)

        # 2. Predict FVC
        X_fvc = self._build_fvc_features(X_pca, dt)
        y_pred = self.fvc_model.predict(X_fvc)

        # 3. Predict Uncertainty (MAD)
        X_unc = self._build_unc_features(X_pca, dt)
        pred_mad = self.unc_model.predict(X_unc)

        # Analytically scale MAD to Sigma for Laplace Metric
        # Sigma = MAD * sqrt(2)
        sigma = pred_mad * np.sqrt(2)

        # Clip Sigma
        sigma = np.maximum(sigma, Config.MIN_CONFIDENCE)

        return y_pred, sigma


def run_training_and_inference():
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Validation -> Inference -> Submission.
    """
    # 1. Initialize Data Manager
    dm = DataManager()

    # 2. Load Data
    print("\n=== Loading Data ===")
    train_data = dm.prepare_dataset("train")
    val_data = dm.prepare_dataset("val")

    # 3. Train Model
    print("\n=== Training Model ===")
    model = DecoupledQuantileModel()
    model.fit(
        train_data["X_static"],
        train_data["weeks"],
        train_data["y"],
        train_data["base_weeks"],
    )

    # 4. Validation
    print("\n=== Validating ===")
    val_preds, val_sigma = model.predict(
        val_data["X_static"], val_data["weeks"], val_data["base_weeks"]
    )

    metric_score = score_function(val_data["y"], val_preds, val_sigma)
    log_metrics({"Validation Laplace Metric": metric_score})

    # 5. Inference on Test Set
    print("\n=== Generating Submission ===")
    test_data = dm.prepare_dataset("test")

    test_preds, test_sigma = model.predict(
        test_data["X_static"], test_data["weeks"], test_data["base_weeks"]
    )

    # 6. Format Submission
    # Construct Patient_Week IDs
    # DataManager returns aligned arrays, so we can iterate or vectorise
    patient_ids = test_data["patient_ids"]
    target_weeks = test_data["weeks"].astype(int)

    submission_ids = [f"{pid}_{week}" for pid, week in zip(patient_ids, target_weeks)]

    df_sub = pd.DataFrame(
        {"Patient_Week": submission_ids, "FVC": test_preds, "Confidence": test_sigma}
    )

    # Save
    save_results(df_sub, Config.SUBMISSION_FILE)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(df_sub.head())
