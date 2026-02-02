import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class GlobalLDA(BaseEstimator, ClassifierMixin):
    """
    Baseline Linear Discriminant Analysis model trained on all classes globally.
    Uses automatic shrinkage (Ledoit-Wolf) to handle high-dimensional feature spaces.
    """

    def __init__(
        self, solver=Config.GLOBAL_LDA_SOLVER, shrinkage=Config.GLOBAL_LDA_SHRINKAGE
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = None

    def fit(self, X, y):
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("GlobalLDA model not fitted")
        return self.model.predict_proba(X)


class HierarchicalLDA(BaseEstimator, ClassifierMixin):
    """
    Taxonomy-aware classifier that decomposes the problem into Genus and Species prediction.

    Structure:
    1. Genus Expert: Estimates P(Genus | X)
    2. Species Experts: Estimate P(Species | Genus, X) using local LDA models per genus.

    Fallback:
    If a genus has insufficient samples, it defaults to a global model's conditional probabilities.
    """

    def __init__(
        self,
        min_samples=Config.MIN_SAMPLES_FOR_GENUS_MODEL,
        solver=Config.GLOBAL_LDA_SOLVER,
        shrinkage=Config.GLOBAL_LDA_SHRINKAGE,
    ):
        self.min_samples = min_samples
        self.solver = solver
        self.shrinkage = shrinkage

        self.genus_model = None
        self.species_models = {}
        self.fallback_model = None

        self.genus_to_species_map = {}
        self.n_classes_ = 0

    def fit(self, X, y, genus_labels):
        # Determine total number of classes (assuming 0..N-1 encoding)
        self.n_classes_ = len(np.unique(y))

        # 1. Train Level 1: Genus Expert (P(Genus | X))
        self.genus_model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.genus_model.fit(X, genus_labels)

        # 2. Train Fallback Model (Global P(Species | X) for fallback calculations)
        self.fallback_model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.fallback_model.fit(X, y)

        # 3. Build Hierarchy and Train Level 2 Models (P(Species | Genus, X))
        unique_genera = np.unique(genus_labels)

        for g_idx in unique_genera:
            # Identify samples for this genus
            mask = genus_labels == g_idx
            X_g = X[mask]
            y_g = y[mask]

            # Identify species belonging to this genus
            species_in_genus = np.unique(y_g)
            self.genus_to_species_map[g_idx] = species_in_genus

            # Condition A: Check for sufficient samples to estimate covariance
            if len(X_g) < self.min_samples:
                continue

            # Condition B: Check if discrimination is needed (more than 1 species)
            if len(species_in_genus) < 2:
                continue

            # Train local expert model
            local_model = LinearDiscriminantAnalysis(
                solver=self.solver, shrinkage=self.shrinkage
            )
            local_model.fit(X_g, y_g)
            self.species_models[g_idx] = local_model

        return self

    def predict_proba(self, X):
        if self.genus_model is None:
            raise RuntimeError("HierarchicalLDA model not fitted")

        n_samples = X.shape[0]
        # Initialize output probabilities matrix
        final_probs = np.zeros((n_samples, self.n_classes_))

        # Get Genus Probabilities: P(G | x)
        # Shape: (n_samples, n_genera)
        genus_probs_raw = self.genus_model.predict_proba(X)

        # Get Global Fallback Probabilities: P_global(S | x)
        global_probs = self.fallback_model.predict_proba(X)

        # Iterate over each genus to calculate P(S | x) = P(S | G, x) * P(G | x)
        # We iterate by index of the genus model's classes
        for i, g_label in enumerate(self.genus_model.classes_):
            # Probability of this genus for all samples
            p_genus = genus_probs_raw[:, i]  # Shape (N,)

            if g_label not in self.genus_to_species_map:
                continue

            species_indices = self.genus_to_species_map[g_label]

            # Determine P(S | G, x)
            if g_label in self.species_models:
                # --- CASE 1: Use Local Expert ---
                local_model = self.species_models[g_label]
                local_probs = local_model.predict_proba(X)

                # Map local columns to global species indices
                # local_model.classes_ contains the actual species IDs
                for j, s_label in enumerate(local_model.classes_):
                    final_probs[:, s_label] += local_probs[:, j] * p_genus

            else:
                # --- CASE 2: Fallback Strategy ---
                if len(species_indices) == 1:
                    # Trivial case: Only one species in this genus. P(S|G) = 1.0
                    s_label = species_indices[0]
                    final_probs[:, s_label] += 1.0 * p_genus
                else:
                    # Sparse data case: Use global model's conditional probabilities
                    # We extract the columns for the species in this genus from the global model

                    # Find column indices in the global model corresponding to these species
                    # fallback_model.classes_ is sorted, so we can use searchsorted
                    col_indices = np.searchsorted(
                        self.fallback_model.classes_, species_indices
                    )

                    # Extract subset of probabilities
                    subset_probs = global_probs[:, col_indices]

                    # Normalize to sum to 1 per row to get P(S | G, x)
                    # Add epsilon to avoid division by zero
                    row_sums = subset_probs.sum(axis=1, keepdims=True) + 1e-15
                    cond_probs = subset_probs / row_sums

                    # Distribute back to final matrix
                    for j, s_idx in enumerate(col_indices):
                        s_label = self.fallback_model.classes_[s_idx]
                        final_probs[:, s_label] += cond_probs[:, j] * p_genus

        return final_probs


class TaxonomyEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble model that averages predictions from the GlobalLDA and HierarchicalLDA.
    Acts as a regularizer: Global model stabilizes predictions where local data is sparse,
    while Hierarchical model refines predictions where distinct sub-population structures exist.
    """

    def __init__(self):
        self.global_model = GlobalLDA()
        self.hierarchical_model = HierarchicalLDA()

    def fit(self, X, y, genus_labels):
        self.global_model.fit(X, y)
        self.hierarchical_model.fit(X, y, genus_labels)
        return self

    def predict_proba(self, X):
        # Get predictions from both branches
        p_global = self.global_model.predict_proba(X)
        p_hierarchical = self.hierarchical_model.predict_proba(X)

        # Average the probabilities
        avg_probs = (p_global + p_hierarchical) / 2.0

        # Apply clipping to avoid log(0) in metric calculation
        # Range: [1e-15, 1 - 1e-15]
        clip_val = Config.PROB_CLIP
        avg_probs = np.clip(avg_probs, clip_val, 1.0 - clip_val)

        return avg_probs
