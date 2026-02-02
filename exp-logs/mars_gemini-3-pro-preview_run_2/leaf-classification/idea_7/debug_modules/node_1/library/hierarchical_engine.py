import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from library.config import (
    WEIGHT_LINEAR,
    WEIGHT_GENERATIVE,
    WEIGHT_QUADRATIC,
    SUBMISSION_PATH,
    ID_COL,
)
from library.utils import get_species_to_genus_mapping
from library.data_loader import load_and_process_data
from library.model_definitions import (
    build_linear_species_model,
    build_generative_species_model,
    build_quadratic_species_model,
    build_genus_supervisor_model,
)


class TaxonomyEnsemble:
    """
    Hierarchical Taxonomy-Regularized Hybrid Ensemble.

    Combines a soft-voting ensemble of species-level models with a genus-level
    supervisor model to enforce biological constraints.
    """

    def __init__(self):
        # Initialize Species-Level Models
        self.linear_model = build_linear_species_model()
        self.generative_model = build_generative_species_model()
        self.quadratic_model = build_quadratic_species_model()

        # Initialize Genus-Level Supervisor
        self.genus_model = build_genus_supervisor_model()

        # Placeholders for metadata
        self.species_to_genus_indices = None
        self.species_le = None
        self.genus_le = None

        # Ensemble Weights
        self.weights = {
            "linear": WEIGHT_LINEAR,
            "generative": WEIGHT_GENERATIVE,
            "quadratic": WEIGHT_QUADRATIC,
        }

    def fit(self, X, y_species, y_genus, species_le, genus_le):
        """
        Trains all component models and builds the taxonomic mapping.
        """
        print("Training TaxonomyEnsemble components...")

        # Store encoders and build mapping
        self.species_le = species_le
        self.genus_le = genus_le
        self.species_to_genus_indices = get_species_to_genus_mapping(
            species_le, genus_le
        )

        # 1. Train Discriminative Linear Branch
        print("  - Fitting Linear Species Model (LogisticRegressionCV)...")
        self.linear_model.fit(X, y_species)

        # 2. Train Generative Linear Branch
        print("  - Fitting Generative Species Model (LDA)...")
        self.generative_model.fit(X, y_species)

        # 3. Train Discriminative Quadratic Branch
        print("  - Fitting Quadratic Species Model (PCA -> Poly -> LR)...")
        self.quadratic_model.fit(X, y_species)

        # 4. Train Genus Supervisor
        print("  - Fitting Genus Supervisor Model...")
        self.genus_model.fit(X, y_genus)

        print("Training complete.")

    def predict_proba(self, X):
        """
        Generates probabilities using the hierarchical Bayesian update strategy.

        P_final(Species) ~ P_ensemble(Species) * P_supervisor(Genus(Species))
        """
        # 1. Get Species Probabilities from each branch
        p_linear = self.linear_model.predict_proba(X)
        p_generative = self.generative_model.predict_proba(X)
        p_quadratic = self.quadratic_model.predict_proba(X)

        # 2. Soft Voting for Base Species Probability
        total_weight = sum(self.weights.values())
        p_species_base = (
            self.weights["linear"] * p_linear
            + self.weights["generative"] * p_generative
            + self.weights["quadratic"] * p_quadratic
        ) / total_weight

        # 3. Get Genus Probabilities from Supervisor
        p_genus = self.genus_model.predict_proba(X)

        # 4. Apply Taxonomic Constraint (Bayesian Update)
        # We map the genus probabilities to the species dimension
        # p_genus shape: (n_samples, n_genera)
        # self.species_to_genus_indices shape: (n_species,)
        # mapped_genus_probs shape: (n_samples, n_species)
        mapped_genus_probs = p_genus[:, self.species_to_genus_indices]

        # Element-wise multiplication: P(S) * P(G|S) where G is deterministic given S
        # Effectively we are doing P(S_updated) = P(S_base) * P(G_predicted)
        p_final_unnormalized = p_species_base * mapped_genus_probs

        # 5. Renormalize to ensure sum to 1 per sample
        row_sums = p_final_unnormalized.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        p_final = p_final_unnormalized / row_sums

        return p_final

    def score(self, X, y):
        """
        Calculates Log Loss on the given data.
        """
        probs = self.predict_proba(X)
        return log_loss(y, probs)


def run_hierarchical_strategy():
    """
    Main execution function for the Hierarchical Taxonomy-Regularized Hybrid Ensemble.
    Loads data, trains the ensemble, generates predictions, and saves the submission.
    """
    print("Initializing Hierarchical Engine...")

    # 1. Load Data
    # We rely on the caching mechanism in data_loader
    X_train, X_test, y_species, y_genus, test_ids, scaler, species_le, genus_le = (
        load_and_process_data(load_cached_data=True)
    )

    # 2. Initialize and Train Ensemble
    ensemble = TaxonomyEnsemble()
    ensemble.fit(X_train, y_species, y_genus, species_le, genus_le)

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
