import numpy as np
from library.config import Config
from library.utils import set_seed, load_model, save_submission
from library.data_loader import load_dataset
from library.features import get_features


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained Topology-Aware Stacking Ensemble.

    Steps:
    1. Loads datasets and retrieves processed features (Lexical, Behavioral, Semantic).
    2. Loads the trained Level 1 models (Lexical RF, Behavioral RF, Semantic XGB).
    3. Generates probability predictions from Level 1 models.
    4. Stacks Level 1 predictions to create meta-features.
    5. Loads the trained Level 2 Meta-Learner (Logistic Regression).
    6. Generates final probabilities and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached features. Defaults to True.
    """
    set_seed()

    print("Initializing prediction pipeline...")

    # ---------------------------------------------------------
    # 1. Data Loading and Feature Retrieval
    # ---------------------------------------------------------
    # We need to load the datasets to ensure the feature pipeline can be consistent
    # get_features handles caching, so if train.py has run, this will be fast.
    print("Retrieving test features...")
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    # This returns a dictionary with all feature views
    data = get_features(train_df, val_df, test_df, load_cached_data=load_cached_data)

    # Extract Test Data
    X_test_lexical = data["X_test_lexical"]
    X_test_behavioral = data["X_test_behavioral"]
    X_test_semantic = data["X_test_semantic"]
    test_ids = data["test_ids"]

    # ---------------------------------------------------------
    # 2. Load Trained Models
    # ---------------------------------------------------------
    print("Loading trained models...")
    try:
        # Note: train.py saves the underlying sklearn/xgb objects, not the wrapper classes
        lexical_rf = load_model("lexical_rf.joblib")
        behavioral_rf = load_model("behavioral_rf.joblib")
        semantic_xgb = load_model("semantic_xgb.joblib")
        meta_learner = load_model("meta_learner.joblib")
    except FileNotFoundError as e:
        print(
            f"Error: Could not load trained models. Ensure train.py has been executed successfully. Details: {e}"
        )
        return

    # ---------------------------------------------------------
    # 3. Level 1 Inference
    # ---------------------------------------------------------
    print("Generating Level 1 predictions...")

    # Lexical View (Sparse RF)
    # predict_proba returns [n_samples, n_classes], we want probability of class 1
    test_pred_lex = lexical_rf.predict_proba(X_test_lexical)[:, 1]

    # Behavioral View (Sparse RF)
    test_pred_beh = behavioral_rf.predict_proba(X_test_behavioral)[:, 1]

    # Semantic View (Dense XGB)
    test_pred_sem = semantic_xgb.predict_proba(X_test_semantic)[:, 1]

    # ---------------------------------------------------------
    # 4. Level 2 Inference (Stacking)
    # ---------------------------------------------------------
    print("Generating Level 2 Meta-Learner predictions...")

    # Stack predictions to match the shape expected by the meta-learner
    # Order must match training: [Lexical, Behavioral, Semantic]
    X_meta_test = np.column_stack([test_pred_lex, test_pred_beh, test_pred_sem])

    # Final prediction
    final_test_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

    # ---------------------------------------------------------
    # 5. Save Submission
    # ---------------------------------------------------------
    save_submission(test_ids, final_test_probs)
    print("Prediction pipeline completed.")
