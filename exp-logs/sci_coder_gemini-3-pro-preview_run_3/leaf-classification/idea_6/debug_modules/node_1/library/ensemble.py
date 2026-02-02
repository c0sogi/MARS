import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import resample

from library.config import Config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders
from library.feature_extraction import process_split


class BaggedLDAPipeline(BaseEstimator, ClassifierMixin):
    """
    Homogeneous Bagged Ensemble of Linear Discriminant Analysis.

    Architecture:
    - Ensemble of N_ESTIMATORS independent pipelines.
    - Each pipeline is trained on a bootstrap sample of the training data.
    - Pipeline components:
        1. PCA for DINOv2 features (Global Geometry).
        2. PCA for ConvNeXt features (Local Texture).
        3. QuantileTransformer for Tabular features.
        4. Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.
    """

    def __init__(self):
        self.n_estimators = Config.N_ESTIMATORS
        self.pca_variance = Config.PCA_VARIANCE
        self.estimators = []
        self.classes_ = None

    def fit(self, X_dino, X_conv, X_tab, y):
        """
        Fits the ensemble on the provided data.

        Args:
            X_dino (np.ndarray): DINOv2 embeddings.
            X_conv (np.ndarray): ConvNeXt embeddings.
            X_tab (np.ndarray): Tabular features.
            y (np.ndarray): Target labels.
        """
        seed_everything()
        self.classes_ = np.unique(y)
        n_samples = X_dino.shape[0]

        # Reset estimators
        self.estimators = []

        print(f"Fitting BaggedLDAPipeline with {self.n_estimators} estimators...")

        for i in range(self.n_estimators):
            # Ensure reproducibility for each estimator
            iter_seed = Config.SEED + i

            # Bootstrap sampling
            indices = resample(
                np.arange(n_samples), replace=True, random_state=iter_seed
            )

            X_dino_boot = X_dino[indices]
            X_conv_boot = X_conv[indices]
            X_tab_boot = X_tab[indices]
            y_boot = y[indices]

            # --- Transformers ---

            # 1. PCA for DINOv2
            # Retain 99% variance
            pca_dino = PCA(
                n_components=self.pca_variance,
                svd_solver="full",
                random_state=iter_seed,
            )
            feat_dino = pca_dino.fit_transform(X_dino_boot)

            # 2. PCA for ConvNeXt
            # Retain 99% variance
            pca_conv = PCA(
                n_components=self.pca_variance,
                svd_solver="full",
                random_state=iter_seed,
            )
            feat_conv = pca_conv.fit_transform(X_conv_boot)

            # 3. Quantile Transformer for Tabular
            # Output normal distribution for LDA compatibility
            # n_quantiles must be <= n_samples
            n_quantiles = min(len(indices), 1000)
            qt_tab = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=iter_seed,
            )
            feat_tab = qt_tab.fit_transform(X_tab_boot)

            # --- Concatenation ---
            X_final = np.hstack([feat_dino, feat_conv, feat_tab])

            # --- Classifier ---
            # LDA with Ledoit-Wolf shrinkage for high-dimensional data
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(X_final, y_boot)

            # Store pipeline components
            self.estimators.append(
                {
                    "pca_dino": pca_dino,
                    "pca_conv": pca_conv,
                    "qt_tab": qt_tab,
                    "lda": lda,
                    "classes": lda.classes_,
                }
            )

        print(f"Fitting complete. Total estimators: {len(self.estimators)}")
        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        """
        Predicts class probabilities for new data.

        Args:
            X_dino (np.ndarray): DINOv2 embeddings.
            X_conv (np.ndarray): ConvNeXt embeddings.
            X_tab (np.ndarray): Tabular features.

        Returns:
            np.ndarray: Averaged probability matrix (n_samples, n_classes).
        """
        if not self.estimators:
            raise RuntimeError("Model not fitted yet.")

        n_samples = X_dino.shape[0]
        n_classes = len(self.classes_)

        # Map class label to column index in the final output
        # This ensures alignment if bootstrap samples missed some classes
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}

        total_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for est in self.estimators:
            # --- Transform ---
            feat_dino = est["pca_dino"].transform(X_dino)
            feat_conv = est["pca_conv"].transform(X_conv)
            feat_tab = est["qt_tab"].transform(X_tab)

            X_final = np.hstack([feat_dino, feat_conv, feat_tab])

            # --- Predict ---
            # probs shape: (n_samples, n_classes_in_estimator)
            probs = est["lda"].predict_proba(X_final)

            # --- Map to Global Class Space ---
            est_classes = est["classes"]
            full_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

            # Identify which columns in global space correspond to the estimator's classes
            col_indices = [class_to_idx[c] for c in est_classes]
            full_probs[:, col_indices] = probs

            total_probs += full_probs

        # --- Aggregate ---
        avg_probs = total_probs / self.n_estimators

        # --- Clip ---
        # As per metric requirements: max(min(p, 1-10^-15), 10^-15)
        epsilon = Config.PROB_EPSILON
        avg_probs = np.clip(avg_probs, epsilon, 1.0 - epsilon)

        return avg_probs


def run_pipeline():
    """
    Orchestrates the full pipeline:
    1. Load Data
    2. Extract Features
    3. Train Ensemble
    4. Predict on Test
    5. Save Submission
    """
    print("Starting Bagged LDA Pipeline...")
    seed_everything()

    # 1. Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    # 2. Extract Features (or load from cache)
    print("Processing Training Data...")
    X_train_dino, X_train_conv, X_train_tab, y_train, _ = process_split(
        train_loader, "train", load_cached_data=True
    )

    # We can use validation data to check performance if desired,
    # but for final submission we often just train.
    # Here we will just process test data for submission.
    print("Processing Test Data...")
    X_test_dino, X_test_conv, X_test_tab, _, test_ids = process_split(
        test_loader, "test", load_cached_data=True
    )

    # 3. Train Model
    model = BaggedLDAPipeline()
    model.fit(X_train_dino, X_train_conv, X_train_tab, y_train)

    # 4. Predict
    print("Generating predictions on Test set...")
    predictions = model.predict_proba(X_test_dino, X_test_conv, X_test_tab)

    # 5. Save Submission
    save_submission(predictions, test_ids, classes)
    print("Pipeline completed successfully.")
