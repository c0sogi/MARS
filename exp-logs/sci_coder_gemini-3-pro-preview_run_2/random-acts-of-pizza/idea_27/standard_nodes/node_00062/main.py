import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
from library.embedding_generator import EmbeddingService
import library.model_trainer as model_trainer


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    print("Initializing ADBEF Strategy Pipeline...")

    # 2. Data Loading
    print("Loading datasets...")
    # Load processed dataframes (Text concatenation + Metadata extraction happens here)
    df_train = data_loader.get_processed_data("train", load_cached_data=True)
    df_val = data_loader.get_processed_data("val", load_cached_data=True)
    df_test = data_loader.get_processed_data("test", load_cached_data=True)

    # 3. Feature Generation
    print("Generating/Loading Embeddings...")
    embedder = EmbeddingService()

    # Primary Backbone (MiniLM) - High Resolution
    print("Processing Primary Backbone (MiniLM)...")
    train_prim = embedder.get_embeddings(
        df_train["text_combined"],
        config.PRIMARY_MODEL_NAME,
        config.TRAIN_PRIMARY_EMBS_PATH,
    )
    val_prim = embedder.get_embeddings(
        df_val["text_combined"], config.PRIMARY_MODEL_NAME, config.VAL_PRIMARY_EMBS_PATH
    )
    test_prim = embedder.get_embeddings(
        df_test["text_combined"],
        config.PRIMARY_MODEL_NAME,
        config.TEST_PRIMARY_EMBS_PATH,
    )

    # Auxiliary Backbone (MPNet) - Deep Semantics (to be compressed)
    print("Processing Auxiliary Backbone (MPNet)...")
    train_aux = embedder.get_embeddings(
        df_train["text_combined"], config.AUX_MODEL_NAME, config.TRAIN_AUX_EMBS_PATH
    )
    val_aux = embedder.get_embeddings(
        df_val["text_combined"], config.AUX_MODEL_NAME, config.VAL_AUX_EMBS_PATH
    )
    test_aux = embedder.get_embeddings(
        df_test["text_combined"], config.AUX_MODEL_NAME, config.TEST_AUX_EMBS_PATH
    )

    # Numerical Metadata
    # Extract and fill NaNs to ensure robust pipeline execution
    print("Processing Metadata...")
    X_meta_train = df_train[config.NUMERICAL_COLS].fillna(0).values
    X_meta_val = df_val[config.NUMERICAL_COLS].fillna(0).values
    X_meta_test = df_test[config.NUMERICAL_COLS].fillna(0).values

    # Targets
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values

    # 4. Training (Cross-Validation on Train Split)
    print("Starting Model Training (5-Fold CV)...")
    # This trains the ensemble and returns the pipelines for each fold
    pipelines, _ = model_trainer.run_cross_validation(
        train_prim, train_aux, X_meta_train, y_train
    )

    # 5. Hold-out Validation
    print("Evaluating on Hold-out Validation Set...")
    # CV-Bagging Inference: Average predictions from all fold pipelines
    val_preds_accum = np.zeros(len(y_val))

    for pipe in pipelines:
        transformer = pipe["transformer"]
        model = pipe["model"]

        # Transform validation data using the specific fold's transformer statistics
        X_val_fused = transformer.transform(val_prim, val_aux, X_meta_val)

        # Predict
        val_preds_accum += model.predict_proba(X_val_fused)[:, 1]

    # Average predictions
    val_preds = val_preds_accum / len(pipelines)

    # Compute Metric
    final_metric = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds)

    # Create a DataFrame for correlation analysis
    df_analysis = pd.DataFrame(X_meta_val, columns=config.NUMERICAL_COLS)
    df_analysis["error_magnitude"] = errors

    # Compute correlation
    correlations = df_analysis.corrwith(df_analysis["error_magnitude"]).sort_values(
        ascending=False
    )

    print("Correlation between Numerical Features and Error Magnitude:")
    print(correlations)
    print("-" * 30)

    # 7. Submission
    THRESHOLD = 0.714740225132835

    if final_metric > THRESHOLD:
        print(
            f"Validation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        model_trainer.generate_submission(
            pipelines, test_prim, test_aux, X_meta_test, df_test["request_id"]
        )
    else:
        print(
            f"Validation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
