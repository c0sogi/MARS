import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from library.config import (
    WEIGHT_LINEAR,
    WEIGHT_GENERATIVE,
    SUBMISSION_PATH,
    ID_COL,
)
from library.data_loader import load_and_process_data
from library.model_definitions import (
    build_linear_species_model,
    build_generative_species_model,
)


class HybridEnsemble:
    """
    Hybrid Generative-Discriminative Ensemble.
    Combines Regularized Logistic Regression and Shrinkage LDA.
    Cite solution_lesson_node_00006, solution_lesson_node_00030
    """

    def __init__(self):
        # Initialize Models
        self.linear_model = build_linear_species_model()
        self.generative_model = build_generative_species_model()

        # Ensemble Weights
        self.weights = {
            "linear": WEIGHT_LINEAR,
            "generative": WEIGHT_GENERATIVE,
        }

    def fit(self, X, y_species):
        """
        Trains component models.
        """
        print("Training HybridEnsemble components...")

        # 1. Train Discriminative Linear Branch
        print("  - Fitting Linear Species Model (LogisticRegressionCV)...")
        self.linear_model.fit(X, y_species)

        # 2. Train Generative Linear Branch
        print("  - Fitting Generative Species Model (LDA)...")
        self.generative_model.fit(X, y_species)

        print("Training complete.")

    def predict_proba(self, X):
        """
        Generates probabilities using weighted soft voting.
        """
        # 1. Get Probabilities from each branch
        p_linear = self.linear_model.predict_proba(X)
        p_generative = self.generative_model.predict_proba(X)

        # 2. Soft Voting
        total_weight = sum(self.weights.values())
        p_final = (
            self.weights["linear"] * p_linear
            + self.weights["generative"] * p_generative
        ) / total_weight

        return p_final

    def score(self, X, y):
        """
        Calculates Log Loss on the given data.
        """
        probs = self.predict_proba(X)
        return log_loss(y, probs)


def run_hierarchical_strategy():
    """
    Main execution function for the Hybrid Ensemble.
    Loads data, trains the ensemble, generates predictions, and saves the submission.
    """
    print("Initializing Hybrid Engine...")

    # 1. Load Data
    # We rely on the caching mechanism in data_loader
    X_train, X_test, y_species, y_genus, test_ids, scaler, species_le, genus_le = (
        load_and_process_data(load_cached_data=True)
    )

    # 2. Initialize and Train Ensemble
    ensemble = HybridEnsemble()
    ensemble.fit(X_train, y_species)

    # 3. Evaluate on Training Data (Sanity Check)
    # Since we merged Train+Val, this is effectively training error, but useful for convergence checks
    train_loss = ensemble.score(X_train, y_species)
    print(f"Training Log Loss: {train_loss}")

    # 4. Generate Test Predictions
    print("Generating predictions for test set...")
    test_probs = ensemble.predict_proba(X_test)

    # 5. Format and Save Submission
    # Clip probabilities to avoid log extremes as per metric definition
    # max(min(p, 1-10^-15), 10^-15)
    test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

    # Create DataFrame
    # Columns must be the species names in alphabetical order (handled by LabelEncoder)
    submission_df = pd.DataFrame(test_probs, columns=species_le.classes_)

    # Insert ID column at the beginning
    submission_df.insert(0, ID_COL, test_ids)

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
