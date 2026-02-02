import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders
from library.feature_extraction import process_split


class SingleLDAPipeline(BaseEstimator, ClassifierMixin):
    """
    Single Linear Discriminant Analysis Pipeline.

    Architecture:
    1. PCA for DINOv2 features (Global Geometry).
    2. PCA for ConvNeXt features (Local Texture).
    3. QuantileTransformer for Tabular features.
    4. Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.
    """

    def __init__(self):
        self.pca_variance = Config.PCA_VARIANCE
        self.pca_dino = None
        self.pca_conv = None
        self.qt_tab = None
        self.lda = None
        self.classes_ = None

    def fit(self, X_dino, X_conv, X_tab, y):
        """
        Fits the pipeline on the provided data.
        """
        seed_everything()
        self.classes_ = np.unique(y)

        print("Fitting SingleLDAPipeline...")

        # 1. PCA for DINOv2
        self.pca_dino = PCA(
            n_components=self.pca_variance,
            svd_solver="full",
            random_state=Config.SEED,
        )
        feat_dino = self.pca_dino.fit_transform(X_dino)

        # 2. PCA for ConvNeXt
        self.pca_conv = PCA(
            n_components=self.pca_variance,
            svd_solver="full",
            random_state=Config.SEED,
        )
        feat_conv = self.pca_conv.fit_transform(X_conv)

        # 3. Quantile Transformer for Tabular
        n_quantiles = min(len(X_tab), 1000)
        self.qt_tab = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=Config.SEED,
        )
        feat_tab = self.qt_tab.fit_transform(X_tab)

        # --- Concatenation ---
        X_final = np.hstack([feat_dino, feat_conv, feat_tab])

        # --- Classifier ---
        # LDA with Ledoit-Wolf shrinkage for high-dimensional data
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda.fit(X_final, y)

        print("Fitting complete.")
        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        """
        Predicts class probabilities for new data.
        """
        if self.lda is None:
            raise RuntimeError("Model not fitted yet.")

        # --- Transform ---
        feat_dino = self.pca_dino.transform(X_dino)
        feat_conv = self.pca_conv.transform(X_conv)
        feat_tab = self.qt_tab.transform(X_tab)

        X_final = np.hstack([feat_dino, feat_conv, feat_tab])

        # --- Predict ---
        probs = self.lda.predict_proba(X_final)

        # --- Clip ---
        epsilon = Config.PROB_EPSILON
        probs = np.clip(probs, epsilon, 1.0 - epsilon)

        return probs


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
