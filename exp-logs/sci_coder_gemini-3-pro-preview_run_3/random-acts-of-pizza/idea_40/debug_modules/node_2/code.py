import os
import sys
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, suppress_warnings
from library.data_processing import process_data
from library.feature_extraction import FeatureFactory
from library.model_definitions import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    InteractionBooster,
    MetadataAnchor,
    MetaLearner,
)
from library.training_workflow import CrossValidator, FinalRetrainer


def demo_pipeline():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Patching
    # -------------------------------------------------------------------------
    print("\n[Demo] Setting up environment and patching Config for speed...")

    # Set a temporary cache directory for the demo to avoid conflicts
    DEMO_CACHE_DIR = "./working/demo_run_cache/"
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Patch Config to use the demo cache and reduce compute load
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    Config.N_FOLDS = 2  # Reduce folds for speed

    # Reduce Estimators for Speed
    Config.RF_LEXICAL_PARAMS["n_estimators"] = 5
    Config.RF_COMMUNITY_PARAMS["n_estimators"] = 5
    Config.XGB_SEMANTIC_PARAMS["n_estimators"] = 5
    Config.RF_SEMANTIC_PARAMS["n_estimators"] = 5
    Config.LGBM_INTERACTION_PARAMS["n_estimators"] = 5
    Config.LR_ANCHOR_PARAMS["max_iter"] = 20

    # Suppress verbose output
    suppress_warnings()
    set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Data Processing
    # -------------------------------------------------------------------------
    print("\n[Demo] Processing Data...")
    # Force processing from scratch to demonstrate logic
    train_df, val_df, test_df = process_data(load_cached_data=False)

    # Verification
    assert "text_combined" in train_df.columns, "Text cleaning failed."
    assert train_df.shape[0] > 0, "Training data is empty."

    # SUBSET DATA FOR SPEED
    # We take a tiny slice to make feature generation and training nearly instantaneous
    print(
        "[Demo] Subsetting data for rapid demonstration (Train=50, Val=20, Test=20)..."
    )
    train_df = train_df.head(50).reset_index(drop=True)
    val_df = val_df.head(20).reset_index(drop=True)
    test_df = test_df.head(20).reset_index(drop=True)

    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Demo] Extracting Features using FeatureFactory...")
    ff = FeatureFactory()

    # 3.1 Lexical Features
    print("  -> Lexical (TF-IDF)")
    X_train_lex, X_val_lex, X_test_lex = ff.get_lexical_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert sp.issparse(X_train_lex), "Lexical features should be sparse."
    assert X_train_lex.shape[0] == 50, "Lexical train shape mismatch."

    # 3.2 Behavioral Features
    print("  -> Behavioral (Community History)")
    X_train_beh, X_val_beh, X_test_beh = ff.get_behavioral_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert sp.issparse(X_train_beh), "Behavioral features should be sparse."

    # 3.3 Semantic Features
    print("  -> Semantic (Embeddings)")
    X_train_sem, X_val_sem, X_test_sem = ff.get_semantic_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert isinstance(
        X_train_sem, np.ndarray
    ), "Semantic features should be dense numpy array."
    assert (
        X_train_sem.shape[1] == 384
    ), "Unexpected embedding dimension (expected 384 for MiniLM)."

    # 3.4 Metadata Features
    print("  -> Metadata (Dense)")
    X_train_meta, X_val_meta, X_test_meta = ff.get_metadata_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert X_train_meta.shape[1] == len(
        Config.METADATA_DENSE_COLS
    ), "Metadata column count mismatch."

    # 3.5 Interaction Features
    print("  -> Latent Interaction (SVD + Meta)")
    X_train_int, X_val_int, X_test_int = ff.get_latent_interaction_features(
        X_train_lex,
        X_train_beh,
        X_train_meta,
        X_val_lex,
        X_val_beh,
        X_val_meta,
        X_test_lex,
        X_test_beh,
        X_test_meta,
        load_cached_data=False,
    )
    assert X_train_int.shape[0] == 50

    # Pack into dictionaries for workflow
    X_train_dict = {
        "lexical": X_train_lex,
        "behavioral": X_train_beh,
        "semantic": X_train_sem,
        "metadata": X_train_meta,
        "interaction": X_train_int,
    }
    X_val_dict = {
        "lexical": X_val_lex,
        "behavioral": X_val_beh,
        "semantic": X_val_sem,
        "metadata": X_val_meta,
        "interaction": X_val_int,
    }
    X_test_dict = {
        "lexical": X_test_lex,
        "behavioral": X_test_beh,
        "semantic": X_test_sem,
        "metadata": X_test_meta,
        "interaction": X_test_int,
    }

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Validation
    # -------------------------------------------------------------------------
    print("\n[Demo] Validating Level 1 Models...")

    # Test LexicalBagger
    print("  -> Testing LexicalBagger...")
    model_lex = LexicalBagger()
    model_lex.fit(X_train_lex, X_train_meta, y_train)
    probs = model_lex.predict_proba(X_val_lex, X_val_meta)
    assert probs.shape == (20, 2), "LexicalBagger output shape incorrect."

    # Test CommunityBagger
    print("  -> Testing CommunityBagger...")
    model_com = CommunityBagger()
    model_com.fit(X_train_beh, X_train_meta, y_train)
    probs = model_com.predict_proba(X_val_beh, X_val_meta)
    assert probs.shape == (20, 2)

    # Test SemanticBooster (XGB)
    print("  -> Testing SemanticBooster (XGBoost)...")
    model_sem_boost = SemanticBooster()
    model_sem_boost.fit(
        X_train_sem,
        X_train_meta,
        y_train,
        X_semantic_val=X_val_sem,
        X_metadata_val=X_val_meta,
        y_val=y_val,
    )
    probs = model_sem_boost.predict_proba(X_val_sem, X_val_meta)
    assert probs.shape == (20, 2)

    # Test InteractionBooster (LGBM)
    print("  -> Testing InteractionBooster (LightGBM)...")
    model_inter = InteractionBooster()
    model_inter.fit(X_train_int, y_train, X_interaction_val=X_val_int, y_val=y_val)
    probs = model_inter.predict_proba(X_val_int)
    assert probs.shape == (20, 2)

    # Test MetadataAnchor
    print("  -> Testing MetadataAnchor (Logistic Regression)...")
    model_meta = MetadataAnchor()
    model_meta.fit(X_train_meta, y_train)
    probs = model_meta.predict_proba(X_val_meta)
    assert probs.shape == (20, 2)

    # -------------------------------------------------------------------------
    # 5. Workflow Execution
    # -------------------------------------------------------------------------
    print("\n[Demo] Running Cross-Validation Workflow...")
    cv = CrossValidator(n_folds=Config.N_FOLDS, random_state=42)
    oof_preds = cv.run_cv(X_train_dict, y_train)

    assert oof_preds.shape == (
        50,
        6,
    ), f"OOF shape mismatch. Expected (50, 6), got {oof_preds.shape}"
    print(f"  -> OOF Predictions Generated. Shape: {oof_preds.shape}")

    print("\n[Demo] Running Final Retraining and Stacking...")
    retrainer = FinalRetrainer()
    final_probs = retrainer.run(
        X_train_dict, y_train, X_val_dict, y_val, X_test_dict, oof_preds
    )

    assert len(final_probs) == 20, "Final predictions length mismatch."
    assert np.all(
        (final_probs >= 0) & (final_probs <= 1)
    ), "Probabilities out of bounds."

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Demo] Generating Submission File...")
    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": final_probs}
    )

    submission_path = Config.SUBMISSION_PATH
    submission.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created."
    print(f"  -> Submission saved to {submission_path}")
    print("\n[Demo] Pipeline demonstration completed successfully.")


if __name__ == "__main__":
    demo_pipeline()
