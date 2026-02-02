from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from library.config import Config


def get_base_models():
    """
    Instantiates and returns the dictionary of Level-1 base learners
    for the Pent-View Stacking architecture.

    Returns:
        dict: A dictionary where keys are model names and values are model instances.
              Keys: 'LexicalBagger', 'CommunityBagger', 'SemanticBooster',
                    'SemanticBagger', 'MetadataAnchor'.
    """
    models = {
        # Sparse Lexical Branch (Text Modality)
        # Random Forest on TF-IDF + Metadata
        "LexicalBagger": RandomForestClassifier(**Config.LEXICAL_RF_PARAMS),
        # Sparse Behavioral Branch (History Modality)
        # Random Forest on Subreddit History TF-IDF + Metadata
        "CommunityBagger": RandomForestClassifier(**Config.BEHAVIORAL_RF_PARAMS),
        # Dense Semantic Branch (Text Modality - Gradient Boosting)
        # XGBoost on Embeddings + Metadata
        "SemanticBooster": XGBClassifier(**Config.SEMANTIC_XGB_PARAMS),
        # Dense Semantic Branch (Text Modality - Bagging)
        # Random Forest on Embeddings + Metadata with strict depth constraints
        "SemanticBagger": RandomForestClassifier(**Config.SEMANTIC_RF_PARAMS),
        # Contextual Branch (Metadata Modality)
        # Logistic Regression on Metadata
        "MetadataAnchor": LogisticRegression(**Config.METADATA_LR_PARAMS),
    }
    return models


def get_meta_model():
    """
    Instantiates and returns the Level-2 Meta-Learner.

    Returns:
        sklearn.linear_model.LogisticRegression: The meta-learner instance.
    """
    return LogisticRegression(**Config.META_LEARNER_PARAMS)
