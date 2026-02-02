import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_logger
from library.stacking_manager import StackingPipeline
from library.data_loader import load_and_preprocess_data
from library.feature_engineering import TextEmbedder

# =============================================================================
# 1. Configuration Overrides for Fast Baseline
# =============================================================================
# Reduce complexity to ensure execution within time limits (Fast Baseline)
Config.N_FOLDS = 3

# Reduce tree counts for base learners
Config.RF_LEXICAL_PARAMS["n_estimators"] = 50
Config.RF_COMMUNITY_PARAMS["n_estimators"] = 50
Config.RF_SEMANTIC_PARAMS["n_estimators"] = 50
Config.XGB_PARAMS["n_estimators"] = 100
Config.LGBM_PARAMS["n_estimators"] = 100

# Ensure GPU acceleration is enabled for dense learners
Config.XGB_PARAMS["tree_method"] = "hist"
Config.XGB_PARAMS["device"] = "cuda"
Config.LGBM_PARAMS["device"] = "gpu"


def main():
    logger = get_logger("RunFile")
    set_seed(Config.RANDOM_STATE)

    logger("Initializing Stacking Pipeline with fast baseline configuration...")
    pipeline = StackingPipeline()

    # Execute the full pipeline:
    # Data Load -> Feature Eng -> Level 1 CV -> Meta Training -> Final Retraining -> Submission Gen
    pipeline.run()

    # =========================================================================
    # 2. Validation Assessment & Failure Analysis
    # =========================================================================
    logger("Starting Post-Run Validation Assessment...")

    # Load processed data to ensure alignment with pipeline's internal state
    # We use load_cached_data=True to leverage the cache created during pipeline.run()
    _, val_df, _ = load_and_preprocess_data(load_cached_data=True)
    y_val = val_df[Config.TARGET_COL].values

    # Retrieve fitted feature engineering objects from the pipeline
    fe = pipeline.final_fe

    # Reconstruct features for the validation set
    # 1. Latent User Persona Features
    latent_val = fe["clusterer"].transform(val_df[Config.SUBREDDIT_COL])

    # 2. Sparse Features (Lexical + Behavioral)
    sparse_val = fe["featurizer"].transform(
        val_df["text_concat"], val_df[Config.SUBREDDIT_COL]
    )

    # 3. Metadata Features (Scaled)
    meta_val = fe["selector"].transform(val_df, latent_val)

    # 4. Semantic Embeddings
    # Reuse pre-computed embeddings if available to save time
    if pipeline.embeddings.get("val") is not None:
        semantic_val = pipeline.embeddings["val"]
    else:
        logger("Re-computing validation embeddings...")
        embedder = TextEmbedder()
        semantic_val = embedder.transform(val_df["text_concat"])

    # Aggregate feature dictionary
    val_feats = {
        "lexical": sparse_val["lexical"],
        "behavioral": sparse_val["behavioral"],
        "metadata": meta_val,
        "semantic": semantic_val,
    }

    # Generate Base Model Predictions on Validation Set
    base_preds_val = pd.DataFrame(index=val_df.index)

    for name, model in pipeline.final_base_models.items():
        # Extract specific feature view for this model
        X_val_model = pipeline._get_model_features(name, val_feats)
        # Predict
        base_preds_val[name] = model.predict_proba(X_val_model)[:, 1]

    # Generate Meta-Learner Predictions
    final_val_probs = pipeline.meta_learner.predict_proba(base_preds_val)[:, 1]

    # Calculate and Print Final Metric
    val_auc = roc_auc_score(y_val, final_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis: Correlation between Error Magnitude and Numerical Features
    logger("Performing Failure Analysis...")
    error_magnitude = np.abs(y_val - final_val_probs)

    # Use raw numerical columns for interpretability
    analysis_df = val_df[Config.NUMERICAL_COLS].copy()
    analysis_df["error_magnitude"] = error_magnitude

    # Compute correlation
    correlations = (
        analysis_df.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # =========================================================================
    # 3. Conditional Submission Logic
    # =========================================================================
    THRESHOLD = 0.7138293787137718
    submission_path = Config.SUBMISSION_PATH

    if val_auc > THRESHOLD:
        logger(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Submission saved to {submission_path}."
        )
    else:
        logger(
            f"Validation AUC ({val_auc}) does not exceed threshold ({THRESHOLD}). Discarding submission."
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)
            logger("Submission file deleted.")


if __name__ == "__main__":
    main()
