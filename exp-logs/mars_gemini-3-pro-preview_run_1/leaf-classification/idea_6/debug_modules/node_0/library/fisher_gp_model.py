import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from library.config import Config
from library.utils import calculate_log_loss, save_submission, set_seed


class FisherBayesianEnsemble:
    """
    A hybrid classifier that combines Linear Discriminant Analysis (LDA) for
    supervised dimensionality reduction (Fisher Embedding) with a Gaussian
    Process Classifier (GPC) for Bayesian inference on the projected subspace.

    The final prediction is an ensemble of the LDA's generative probability
    and the GPC's discriminative probability.
    """

    def __init__(
        self,
        lda_n_components=Config.LDA_N_COMPONENTS,
        lda_solver=Config.LDA_SOLVER,
        gpc_length_scale=Config.GPC_RBF_LENGTH_SCALE,
        gpc_length_scale_bounds=Config.GPC_RBF_LENGTH_SCALE_BOUNDS,
        gpc_noise_level=Config.GPC_NOISE_LEVEL,
        gpc_noise_level_bounds=Config.GPC_NOISE_LEVEL_BOUNDS,
        gpc_optimizer=Config.GPC_OPTIMIZER,
        gpc_n_restarts=Config.GPC_N_RESTARTS_OPTIMIZER,
        gpc_max_iter_predict=Config.GPC_MAX_ITER_PREDICT,
        gpc_multi_class=Config.GPC_MULTI_CLASS,
        weight_lda=Config.WEIGHT_LDA,
        weight_gpc=Config.WEIGHT_GPC,
        random_state=Config.SEED,
    ):

        self.lda_n_components = lda_n_components
        self.lda_solver = lda_solver
        self.weight_lda = weight_lda
        self.weight_gpc = weight_gpc
        self.random_state = random_state

        # Initialize LDA
        # Note: LDA solver 'svd' does not use random_state, but we accept it for consistency
        self.lda = LinearDiscriminantAnalysis(
            n_components=lda_n_components, solver=lda_solver
        )

        # Construct Kernel: Constant * RBF + White
        # ConstantKernel allows the GP to scale the covariance function magnitude
        k_rbf = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=gpc_length_scale, length_scale_bounds=gpc_length_scale_bounds
        )
        k_noise = WhiteKernel(
            noise_level=gpc_noise_level, noise_level_bounds=gpc_noise_level_bounds
        )
        kernel = k_rbf + k_noise

        # Initialize GPC
        self.gpc = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=gpc_optimizer,
            n_restarts_optimizer=gpc_n_restarts,
            max_iter_predict=gpc_max_iter_predict,
            multi_class=gpc_multi_class,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            copy_X_train=False,  # Save memory
        )

    def fit(self, X, y):
        """
        Fits the LDA backbone and then the GPC head on the projected features.
        """
        print(f"Training FisherBayesianEnsemble (LDA -> GPC)...")

        # 1. Fit LDA Backbone
        start_time = time.time()
        self.lda.fit(X, y)
        lda_time = time.time() - start_time
        print(f"  LDA fitted in {lda_time:.2f}s")

        # 2. Project Data to Fisher Subspace
        X_fisher = self.lda.transform(X)
        print(f"  Data projected from {X.shape[1]} to {X_fisher.shape[1]} dimensions.")

        # 3. Fit GPC Head
        start_time = time.time()
        self.gpc.fit(X_fisher, y)
        gpc_time = time.time() - start_time
        print(f"  GPC fitted in {gpc_time:.2f}s")

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the weighted ensemble of LDA and GPC.
        """
        # 1. LDA Probabilities (Generative Baseline)
        proba_lda = self.lda.predict_proba(X)

        # 2. GPC Probabilities (Discriminative Refinement)
        X_fisher = self.lda.transform(X)
        proba_gpc = self.gpc.predict_proba(X_fisher)

        # 3. Weighted Ensemble
        proba_final = (self.weight_lda * proba_lda) + (self.weight_gpc * proba_gpc)

        return proba_final


def train_fisher_gp(X_train, y_train, X_val, y_val, verbose=True):
    """
    Orchestrates the training and evaluation of the Fisher-GP model.
    """
    set_seed(Config.SEED)

    # Instantiate Model
    model = FisherBayesianEnsemble()

    # Fit Model
    model.fit(X_train, y_train)

    # Evaluate on Train
    train_probs = model.predict_proba(X_train)
    # y_train are integer labels, so we pass labels=range(n_classes) to ensure correct mapping
    labels = np.arange(Config.N_CLASSES)
    train_loss = calculate_log_loss(y_train, train_probs, labels=labels)

    # Evaluate on Validation
    val_probs = model.predict_proba(X_val)
    val_loss = calculate_log_loss(y_val, val_probs, labels=labels)

    if verbose:
        print("\n=== Model Evaluation ===")
        print(f"Train Log Loss: {train_loss:.15f}")
        print(f"Val Log Loss:   {val_loss:.15f}")

    return model, val_loss


def predict_and_submit(model, X_test, test_ids, classes):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating predictions for test set...")

    # Generate probabilities
    probs = model.predict_proba(X_test)

    # Save submission
    save_submission(test_ids, classes, probs)

    return probs
