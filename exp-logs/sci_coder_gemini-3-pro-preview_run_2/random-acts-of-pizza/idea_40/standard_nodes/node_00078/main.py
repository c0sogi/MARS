import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.data_loader import DataLoader
from library.embedding_engine import EmbeddingEngine
from library.feature_engineer import CoherenceFeatureProcessor
from library.model_factory import ModelFactory
from library.execution_manager import ExecutionManager
from library.utils import set_seed, setup_logger


def main():
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Setup logger for the runfile (silencing non-critical logs if needed)
    # We rely on the library loggers for details, here we just print required outputs.

    # ==========================================
    # Phase 1: Strict Validation (Train on Train, Eval on Val)
    # ==========================================

    # 1. Load Data Splits
    loader = DataLoader()
    df_train = loader.load_split("train")
    df_val = loader.load_split("val")

    y_train = df_train["requester_received_pizza"].values
    y_val = df_val["requester_received_pizza"].values

    # 2. Generate Embeddings
    # We use the EmbeddingEngine. For Train, we use the standard caching mechanism.
    # For Val, we compute directly to avoid overwriting train caches or complex path management.
    embedder = EmbeddingEngine()

    # Train Embeddings (Cached)
    train_title, train_body, train_global = embedder.generate_train_embeddings(
        df_train, load_cached_data=True
    )

    # Val Embeddings (Computed on the fly)
    # Title
    val_title_texts = df_val[Config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
    val_title = embedder._compute_embeddings(val_title_texts, Config.MODEL_MINILM)

    # Body
    val_body_texts = df_val[Config.TEXT_COL_BODY].fillna("").astype(str).tolist()
    val_body = embedder._compute_embeddings(val_body_texts, Config.MODEL_MINILM)

    # Global (Concatenated)
    val_global_texts = (
        df_val[Config.TEXT_COL_TITLE].fillna("").astype(str)
        + " "
        + df_val[Config.TEXT_COL_BODY].fillna("").astype(str)
    ).tolist()
    val_global = embedder._compute_embeddings(val_global_texts, Config.MODEL_MPNET)

    # Metadata Features
    X_meta_train = df_train[Config.NUMERIC_COLS].fillna(0).values
    X_meta_val = df_val[Config.NUMERIC_COLS].fillna(0).values

    # 3. Feature Engineering
    # Fit processor ONLY on Training data to prevent leakage
    processor = CoherenceFeatureProcessor()
    processor.fit(train_title, train_body, train_global, X_meta_train)

    # Transform both sets
    X_train_fused = processor.transform(
        train_title, train_body, train_global, X_meta_train
    )
    X_val_fused = processor.transform(val_title, val_body, val_global, X_meta_val)

    # 4. Model Training
    # Optimize and train the Bagged Logistic Regression Ensemble
    factory = ModelFactory()
    model = factory.optimize_and_train(X_train_fused, y_train)

    # 5. Evaluation
    # Predict probabilities on Validation set
    y_pred_val = model.predict_proba(X_val_fused)[:, 1]
    val_score = roc_auc_score(y_val, y_pred_val)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate error magnitude
    errors = np.abs(y_val - y_pred_val)

    # Create a DataFrame to correlate errors with metadata features
    df_analysis = pd.DataFrame(X_meta_val, columns=Config.NUMERIC_COLS)
    df_analysis["error_magnitude"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )
    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # ==========================================
    # Phase 2: Submission (CV-Bagging Strategy)
    # ==========================================

    THRESHOLD = 0.7201989696216022

    if val_score > THRESHOLD:
        print(f"\nValidation metric ({val_score}) exceeds threshold ({THRESHOLD}).")
        print("Executing full CV-Bagging pipeline for submission generation...")

        # Instantiate ExecutionManager to run the robust pipeline (Idea 40)
        # This will:
        # 1. Combine Train + Val
        # 2. Run 5-Fold Stratified CV
        # 3. Train 5 Bagged Ensembles
        # 4. Average predictions on Test set
        # 5. Save submission.csv
        manager = ExecutionManager()
        manager.run_cv_and_inference(load_cached_data=True)

    else:
        print(
            f"\nValidation metric ({val_score}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
