import os


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_32"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Model Identifiers
    # ==========================================
    # View 1: Semantic Anchor (High-Res)
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # View 2: Semantic Auxiliary (Deep Semantics)
    SEMANTIC_AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # View 3: Affective Auxiliary (Orthogonal Signal)
    AFFECTIVE_AUX_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Dimensionality reduction for Semantic Aux view
    PCA_COMPONENTS = 50

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5

    # Bagging Ensemble Settings
    N_BAGGING_ESTIMATORS = 20

    # Grid Search Space for Logistic Regression Base Estimator
    # Note: scikit-learn >= 1.2 uses 'estimator' instead of 'base_estimator' in BaggingClassifier
    LR_PARAM_GRID = {
        "bagging__estimator__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "bagging__estimator__class_weight": ["balanced", None],
    }

    # ==========================================
    # Runtime / Debugging
    # ==========================================
    # Set to a small integer (e.g., 100) to run on a subset for debugging.
    # Set to None for the full run.
    DEBUG_SAMPLE_SIZE = None
