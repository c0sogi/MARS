import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("models")


class MultiViewEnsemble:
    """
    A Multi-View Hybrid Ensemble that aggregates predictions from two distinct
    modeling paradigms, having removed the underperforming linear branch:
    1. Lexical-Tree: Random Forest on TF-IDF + Dense Metadata.
    2. Semantic-Tree: Random Forest on Embeddings + Dense Metadata.
    """

    def __init__(self):
        # Initialize Lexical-Tree Branch
        self.lexical_tree = RandomForestClassifier(**Config.RF_PARAMS)

        # Initialize Semantic-Tree Branch
        self.semantic_tree = RandomForestClassifier(**Config.RF_PARAMS)

    def _prepare_features(self, X_dict):
        """
        Prepares the specific feature sets for each branch from the input dictionary.

        Args:
            X_dict (dict): Dictionary containing 'tfidf', 'embedding', and 'dense'.

        Returns:
            tuple: (X_lex_tree, X_sem_tree)
        """
        # Feature set for Lexical-Tree (TF-IDF + Dense)
        # We use sparse hstack because TF-IDF is sparse
        X_lex_tree = sp.hstack([X_dict["tfidf"], sp.csr_matrix(X_dict["dense"])])

        # Feature set for Semantic-Tree (Embedding + Dense)
        # Both are dense arrays, use numpy hstack
        X_sem_tree = np.hstack([X_dict["embedding"], X_dict["dense"]])

        return X_lex_tree, X_sem_tree

    def fit(self, X_train_dict, y_train, X_val_dict=None, y_val=None):
        """
        Trains the internal estimators.

        Args:
            X_train_dict (dict): Training features.
            y_train (array-like): Training labels.
            X_val_dict (dict, optional): Validation features for monitoring.
            y_val (array-like, optional): Validation labels for monitoring.
        """
        logger.info("Preparing feature sets for training...")
        X_lex_train, X_sem_train = self._prepare_features(X_train_dict)

        # Train Lexical-Tree Branch
        logger.info("Training Lexical-Tree Branch (Random Forest)...")
        self.lexical_tree.fit(X_lex_train, y_train)

        # Train Semantic-Tree Branch
        logger.info("Training Semantic-Tree Branch (Random Forest)...")
        self.semantic_tree.fit(X_sem_train, y_train)

        logger.info("Training complete.")

        # Evaluation if validation set is provided
        if X_val_dict is not None and y_val is not None:
            self._evaluate(X_train_dict, y_train, X_val_dict, y_val)

    def _evaluate(self, X_train_dict, y_train, X_val_dict, y_val):
        """
        Internal method to evaluate and print metrics.
        """
        logger.info("Evaluating model performance...")

        # Get probabilities
        train_probs = self.predict_proba(X_train_dict)
        val_probs = self.predict_proba(X_val_dict)

        # Calculate AUC for the ensemble
        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        logger.info(f"Ensemble Training AUC: {train_auc}")
        logger.info(f"Ensemble Validation AUC: {val_auc}")

        # Individual Branch Performance on Validation
        X_lex_val, X_sem_val = self._prepare_features(X_val_dict)

        p_lex = self.lexical_tree.predict_proba(X_lex_val)[:, 1]
        p_sem = self.semantic_tree.predict_proba(X_sem_val)[:, 1]

        logger.info(f"Branch 1 (Lexical-Tree) Val AUC:   {roc_auc_score(y_val, p_lex)}")
        logger.info(f"Branch 2 (Semantic-Tree) Val AUC:  {roc_auc_score(y_val, p_sem)}")

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities for the input samples.
        Averages the probabilities from all branches (Soft Voting).

        Args:
            X_dict (dict): Input features.

        Returns:
            np.ndarray: Probability of the positive class (1).
        """
        X_lex, X_sem = self._prepare_features(X_dict)

        # Get probabilities for the positive class (index 1)
        prob_lexical = self.lexical_tree.predict_proba(X_lex)[:, 1]
        prob_semantic = self.semantic_tree.predict_proba(X_sem)[:, 1]

        # Average probabilities
        avg_prob = (prob_lexical + prob_semantic) / 2.0
        return avg_prob

    def predict(self, X_dict, threshold=0.5):
        """
        Predicts class labels based on a threshold.

        Args:
            X_dict (dict): Input features.
            threshold (float): Threshold for classification.

        Returns:
            np.ndarray: Binary class labels.
        """
        probs = self.predict_proba(X_dict)
        return (probs >= threshold).astype(int)
