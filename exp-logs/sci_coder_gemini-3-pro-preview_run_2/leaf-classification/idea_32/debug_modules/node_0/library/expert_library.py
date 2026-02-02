import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from library.utils import set_seed


class BaseExpert(BaseEstimator, ClassifierMixin):
    """
    Abstract base class for all experts in the Hierarchical Ensemble.
    Enforces the interface for fitting and probability prediction.
    """

    def fit(self, X, y):
        """
        Fit the expert to the training data.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.
        Returns:
            np.ndarray: Shape (n_samples, n_classes), dtype=float64
        """
        raise NotImplementedError


class SklearnExpert(BaseExpert):
    """
    Wrapper for standard Scikit-Learn estimators (e.g., LDA, QDA).
    Ensures float64 precision for predictions.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        # Predict and cast to float64
        return self.estimator.predict_proba(X).astype(np.float64)


class BaggedExpert(BaseExpert):
    """
    Wrapper for BaggingClassifier to implement the Generative Bagging strategy.
    Designed to reduce variance of covariance estimators (e.g., LDA).
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        # Initialize the BaggingClassifier
        # Using 'estimator' parameter compatible with sklearn >= 1.2
        self.model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float64)


class TaxonomicExpert(BaseExpert):
    """
    Hierarchical expert that predicts Genus probabilities and distributes them
    uniformly among constituent Species.

    Acts as an additive taxonomic constraint/prior.
    """

    def __init__(self, estimator, species_classes, genus_classes):
        """
        Args:
            estimator: An initialized sklearn estimator (e.g., LDA) to predict Genus.
            species_classes (array-like): List/Array of all species names (strings).
            genus_classes (array-like): List/Array of all genus names (strings).
        """
        self.estimator = estimator
        self.species_classes = np.array(species_classes)
        self.genus_classes = np.array(genus_classes)
        self.genus_to_species_map = self._build_mapping()

    def _build_mapping(self):
        """
        Creates a mapping from Genus Index -> List of Species Indices.
        Assumes species names are formatted as 'Genus_Species'.
        """
        mapping = {}

        # Create a lookup for genus indices
        genus_to_idx = {name: idx for idx, name in enumerate(self.genus_classes)}

        for s_idx, s_name in enumerate(self.species_classes):
            # Extract genus from species name (e.g., 'Acer_Capillipes' -> 'Acer')
            g_name = s_name.split("_")[0]

            if g_name in genus_to_idx:
                g_idx = genus_to_idx[g_name]
                if g_idx not in mapping:
                    mapping[g_idx] = []
                mapping[g_idx].append(s_idx)
            else:
                # Fallback: If genus not found (should not happen in valid data)
                pass

        return mapping

    def fit(self, X, y):
        """
        Fit the underlying estimator.
        IMPORTANT: 'y' must be the GENUS labels (encoded integers), not species labels.
        """
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predicts Genus probabilities and maps them to Species space.
        """
        # 1. Get Genus Probabilities: (N, n_genera)
        genus_probs = self.estimator.predict_proba(X).astype(np.float64)

        n_samples = X.shape[0]
        n_species = len(self.species_classes)

        # 2. Initialize Species Probabilities: (N, n_species)
        species_probs = np.zeros((n_samples, n_species), dtype=np.float64)

        # 3. Distribute probability mass
        for g_idx, s_indices in self.genus_to_species_map.items():
            if not s_indices:
                continue

            # Get probability for this genus across all samples
            # Shape: (N,)
            p_genus = genus_probs[:, g_idx]

            # Uniform distribution among children species
            n_children = len(s_indices)
            p_species = p_genus / n_children

            # Add to the species columns
            for s_idx in s_indices:
                species_probs[:, s_idx] += p_species

        return species_probs
