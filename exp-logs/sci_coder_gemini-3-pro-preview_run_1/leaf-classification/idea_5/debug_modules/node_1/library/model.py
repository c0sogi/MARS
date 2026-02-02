import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import log_loss
from library.config import Config


class FisherGaussianEnsemble(BaseEstimator, ClassifierMixin):
    """
    A Hybrid Generative-Discriminative Ensemble.

    Architecture:
    1. Backbone: Linear Discriminant Analysis (LDA)
       - Projects high-dimensional data (192 features) into a compact Fisher Subspace.
       - Acts as a 'Safety Net' providing robust linear baseline probabilities.

    2. Head: Gaussian Process Classifier (GPC)
       - Operates on the Fisher features.
       - Uses an RBF + WhiteKernel to model non-linear decision boundaries with
         Bayesian uncertainty calibration.

    3. Aggregation:
       - Final probabilities are a weighted average of LDA and GPC outputs.
    """

    def __init__(self):
        # Load hyperparameters from Config
        self.lda_params = Config.LDA_PARAMS
        self.gpc_params = Config.GPC_PARAMS
        self.kernel_params = Config.KERNEL_PARAMS
        self.weights = Config.ENSEMBLE_WEIGHTS

        # Initialize the Backbone (LDA)
        self.lda = LinearDiscriminantAnalysis(**self.lda_params)

        # Initialize the Kernel for GPC
        # Compound kernel: RBF for smoothness + WhiteKernel for noise handling
        self.kernel = RBF(
            length_scale=self.kernel_params["rbf_length_scale"],
            length_scale_bounds=self.kernel_params["rbf_length_scale_bounds"],
        ) + WhiteKernel(
            noise_level=self.kernel_params["white_noise_level"],
            noise_level_bounds=self.kernel_params["white_noise_level_bounds"],
        )

        # Initialize the Head (GPC)
        self.gpc = GaussianProcessClassifier(kernel=self.kernel, **self.gpc_params)

        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the pipeline to the training data.
        """
        self.classes_ = np.unique(y)

        # 1. Fit LDA Backbone
        # This computes the projection matrix based on class separability
        print("Fitting LDA Backbone...")
        self.lda.fit(X, y)

        # 2. Project Data into Fisher Subspace
        # Transform 192 dims -> (N_classes - 1) dims
        print("Projecting training data into Fisher Subspace...")
        X_fisher = self.lda.transform(X)

        # 3. Fit GPC Head
        # Optimizes kernel hyperparameters (length_scale, noise_level)
        print("Fitting GPC Head (Bayesian optimization)...")
        self.gpc.fit(X_fisher, y)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the hybrid ensemble.
        """
        # 1. LDA Path (Safety Net)
        # Provides robust baseline, prevents failure if GPC overfits
        prob_lda = self.lda.predict_proba(X)

        # 2. GPC Path (Refinement)
        # Projects data and applies Bayesian classification
        X_fisher = self.lda.transform(X)
        prob_gpc = self.gpc.predict_proba(X_fisher)

        # 3. Ensemble Aggregation
        w_lda = self.weights.get("lda", 0.5)
        w_gpc = self.weights.get("gpc", 0.5)

        # Weighted Average
        prob_final = (w_lda * prob_lda) + (w_gpc * prob_gpc)

        # Ensure numerical stability (renormalize rows)
        # Clip to avoid log(0) issues downstream, though log_loss handles it usually
        prob_final = np.clip(prob_final, 1e-15, 1 - 1e-15)
        prob_final = prob_final / prob_final.sum(axis=1, keepdims=True)

        return prob_final

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the FisherGaussianEnsemble model and evaluates it.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray, optional): Validation features.
        y_val (np.ndarray, optional): Validation labels.

    Returns:
        model: The trained FisherGaussianEnsemble instance.
    """
    print(f"Initializing Fisher-Gaussian Process Pipeline...")
    model = FisherGaussianEnsemble()

    # Train
    model.fit(X_train, y_train)

    # Evaluate if validation set is provided
    if X_val is not None and y_val is not None:
        print("Evaluating on Validation Set...")

        # Get probabilities
        y_pred_proba = model.predict_proba(X_val)

        # Calculate Log Loss
        # We pass model.classes_ to ensure correct column mapping
        loss = log_loss(y_val, y_pred_proba, labels=model.classes_)

        print(f"Validation Multi-class Log Loss: {loss}")

    return model


def predict_and_submit(model, X_test, test_ids):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained FisherGaussianEnsemble model.
        X_test (np.ndarray): Test features.
        test_ids (np.ndarray): Test image IDs.
    """
    print("Generating predictions for test set...")

    # Generate probabilities
    y_pred_proba = model.predict_proba(X_test)

    # Create DataFrame
    # Columns must be the species names
    df_submission = pd.DataFrame(y_pred_proba, columns=model.classes_)

    # Insert ID column at the beginning
    df_submission.insert(0, "id", test_ids)

    # Ensure IDs are integers
    df_submission["id"] = df_submission["id"].astype(int)

    # Save to CSV
    output_path = Config.SUBMISSION_FILE
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)

    print("Submission generated successfully.")
