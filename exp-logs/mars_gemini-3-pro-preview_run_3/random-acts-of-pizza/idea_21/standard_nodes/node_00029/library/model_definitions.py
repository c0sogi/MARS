import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library import config, utils


class ModelFactory:
    """
    Factory class to create model instances and prepare specific feature sets
    for the Hex-View Stacking Ensemble.
    """

    # Model Names
    LEXICAL_BAGGER = "lexical_bagger"
    COMMUNITY_BAGGER = "community_bagger"
    SEMANTIC_BOOSTER = "semantic_booster"
    SEMANTIC_BAGGER = "semantic_bagger"
    PERSONA_BOOSTER = "persona_booster"
    METADATA_ANCHOR = "metadata_anchor"

    @staticmethod
    def get_level1_models():
        """
        Instantiates and returns the dictionary of Level 1 base learners.

        Returns:
            dict: {model_name: model_instance}
        """
        models = {}

        # 1. Sparse Lexical Branch (Text Modality)
        # Random Forest on TF-IDF + Metadata
        models[ModelFactory.LEXICAL_BAGGER] = RandomForestClassifier(**config.RF_PARAMS)

        # 2. Sparse Behavioral Branch (History Modality)
        # Random Forest on History TF-IDF + Metadata
        models[ModelFactory.COMMUNITY_BAGGER] = RandomForestClassifier(
            **config.RF_PARAMS
        )

        # 3. Dense Semantic Text Branch (Text Modality)
        # XGBoost on Text Embeddings + Metadata
        models[ModelFactory.SEMANTIC_BOOSTER] = XGBClassifier(**config.XGB_PARAMS)
        # Random Forest on Text Embeddings + Metadata
        models[ModelFactory.SEMANTIC_BAGGER] = RandomForestClassifier(
            **config.RF_PARAMS
        )

        # 4. Dense Semantic History Branch (History Modality)
        # XGBoost on History Embeddings + Metadata (Isolated View)
        models[ModelFactory.PERSONA_BOOSTER] = XGBClassifier(**config.XGB_PARAMS)

        # 5. Contextual Branch (Metadata Modality)
        # Logistic Regression on Metadata only
        models[ModelFactory.METADATA_ANCHOR] = LogisticRegression(**config.LR_PARAMS)

        return models

    @staticmethod
    def get_meta_learner():
        """
        Instantiates and returns the Level 2 Meta-Learner.

        Returns:
            LogisticRegression: The meta-learner instance.
        """
        return LogisticRegression(**config.LR_PARAMS)

    @staticmethod
    def prepare_features(model_name, feature_dict, split):
        """
        Assembles the specific feature set required for a given model from the
        feature dictionary provided by FeatureExtractor.

        Args:
            model_name (str): The key identifying the model (e.g., 'lexical_bagger').
            feature_dict (dict): Dictionary containing 'train', 'val', 'test' sub-dicts
                                 with keys 'lexical', 'behavioral', 'semantic_text',
                                 'semantic_history', 'metadata'.
            split (str): The data split to retrieve ('train', 'val', or 'test').

        Returns:
            array-like: The concatenated feature matrix (sparse or dense) for the model.
        """
        # Retrieve the feature dictionary for the specific split
        feats = feature_dict[split]

        # Extract specific components
        # Sparse matrices
        lexical = feats.get("lexical")
        behavioral = feats.get("behavioral")

        # Dense arrays
        sem_text = feats.get("semantic_text")
        sem_hist = feats.get("semantic_history")
        metadata = feats.get("metadata")

        # Logic for Feature Combination based on Model Topology

        if model_name == ModelFactory.LEXICAL_BAGGER:
            # Sparse TF-IDF (Text) + Dense Metadata
            # Use scipy.sparse.hstack
            return sp.hstack([lexical, metadata], format="csr")

        elif model_name == ModelFactory.COMMUNITY_BAGGER:
            # Sparse TF-IDF (History) + Dense Metadata
            # Use scipy.sparse.hstack
            return sp.hstack([behavioral, metadata], format="csr")

        elif model_name in [
            ModelFactory.SEMANTIC_BOOSTER,
            ModelFactory.SEMANTIC_BAGGER,
        ]:
            # Dense Embeddings (Text) + Dense Metadata
            # Use numpy.hstack
            return np.hstack([sem_text, metadata])

        elif model_name == ModelFactory.PERSONA_BOOSTER:
            # Dense Embeddings (History) + Dense Metadata
            # Use numpy.hstack
            return np.hstack([sem_hist, metadata])

        elif model_name == ModelFactory.METADATA_ANCHOR:
            # Metadata only
            return metadata

        else:
            raise ValueError(f"Unknown model name: {model_name}")
