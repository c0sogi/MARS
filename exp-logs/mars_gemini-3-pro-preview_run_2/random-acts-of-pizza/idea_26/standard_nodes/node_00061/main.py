import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library import utils
from library import data_loader
from library import embedding_manager
from library import pipeline_factory
from library import trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization and Seeding
    utils.set_seed(Config.SEED)

    # 2. Data Loading
    # Load processed DataFrames for Train, Validation, and Test splits
    # This uses the metadata files to ensure correct splitting
    train_df, val_df, test_df = data_loader.load_dataset(load_from_cache=True)

    # Extract Targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values
    # Test set does not have targets

    # Extract Text Lists for Embedding Generation
    train_texts = train_df["text_combined"].tolist()
    val_texts = val_df["text_combined"].tolist()
    test_texts = test_df["text_combined"].tolist()

    # 3. Feature Engineering: Embeddings
    # Branch A: MiniLM (384 dim)
    print("\n[Feature Engineering] Generating Branch A (MiniLM) Embeddings...")
    emb_a_train = embedding_manager.get_embeddings(
        texts=train_texts,
        model_name=Config.MODEL_A_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_A_PREFIX}train.npy"
        ),
    )
    emb_a_val = embedding_manager.get_embeddings(
        texts=val_texts,
        model_name=Config.MODEL_A_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_A_PREFIX}val.npy"
        ),
    )
    emb_a_test = embedding_manager.get_embeddings(
        texts=test_texts,
        model_name=Config.MODEL_A_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_A_PREFIX}test.npy"
        ),
    )

    # Branch B: MPNet (768 dim)
    print("\n[Feature Engineering] Generating Branch B (MPNet) Embeddings...")
    emb_b_train = embedding_manager.get_embeddings(
        texts=train_texts,
        model_name=Config.MODEL_B_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_B_PREFIX}train.npy"
        ),
    )
    emb_b_val = embedding_manager.get_embeddings(
        texts=val_texts,
        model_name=Config.MODEL_B_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_B_PREFIX}val.npy"
        ),
    )
    emb_b_test = embedding_manager.get_embeddings(
        texts=test_texts,
        model_name=Config.MODEL_B_NAME,
        cache_path=os.path.join(
            Config.CACHE_DIR, f"{Config.CACHE_EMBEDDINGS_B_PREFIX}test.npy"
        ),
    )

    # 4. Feature Engineering: Metadata
    # Extract numerical metadata
    meta_cols = Config.NUMERIC_COLS
    meta_train = train_df[meta_cols].values
    meta_val = val_df[meta_cols].values
    meta_test = test_df[meta_cols].values

    meta_dim = len(meta_cols)

    # 5. Feature Concatenation (Early Fusion)
    # Combine Embeddings and Metadata for each branch

    # Branch A Data
    X_train_a = np.hstack([emb_a_train, meta_train])
    X_val_a = np.hstack([emb_a_val, meta_val])
    X_test_a = np.hstack([emb_a_test, meta_test])

    # Branch B Data
    X_train_b = np.hstack([emb_b_train, meta_train])
    X_val_b = np.hstack([emb_b_val, meta_val])
    X_test_b = np.hstack([emb_b_test, meta_test])

    # 6. Model Training: Branch A (MiniLM -> L2 -> RankGauss -> Bagged LR)
    print("\n[Training] Starting Branch A (MiniLM)...")
    models_a, oof_a, test_preds_a, scores_a = trainer.run_cross_validation(
        X=X_train_a,
        y=y_train,
        X_test=X_test_a,
        pipeline_creator=pipeline_factory.create_branch_a_pipeline,
        embedding_dim=Config.MODEL_A_DIM,
        meta_dim=meta_dim,
        model_name_prefix="branch_a",
    )

    # Inference on Validation Holdout for Branch A
    val_preds_a = np.zeros(len(X_val_a))
    for model in models_a:
        val_preds_a += model.predict_proba(X_val_a)[:, 1]
    val_preds_a /= len(models_a)

    # 7. Model Training: Branch B (MPNet -> PCA -> L2 -> RankGauss -> Bagged LR)
    print("\n[Training] Starting Branch B (MPNet)...")
    models_b, oof_b, test_preds_b, scores_b = trainer.run_cross_validation(
        X=X_train_b,
        y=y_train,
        X_test=X_test_b,
        pipeline_creator=pipeline_factory.create_branch_b_pipeline,
        embedding_dim=Config.MODEL_B_RAW_DIM,
        meta_dim=meta_dim,
        model_name_prefix="branch_b",
    )

    # Inference on Validation Holdout for Branch B
    val_preds_b = np.zeros(len(X_val_b))
    for model in models_b:
        val_preds_b += model.predict_proba(X_val_b)[:, 1]
    val_preds_b /= len(models_b)

    # 8. Consensus Ensemble (Soft Voting)
    # Average the probabilities from both branches
    final_val_preds = 0.5 * val_preds_a + 0.5 * val_preds_b
    final_test_preds = 0.5 * test_preds_a + 0.5 * test_preds_b

    # 9. Validation Evaluation
    final_val_auc = roc_auc_score(y_val, final_val_preds)
    print(f"\nFinal Validation Metric: {final_val_auc}")

    # 10. Failure Analysis
    print("\n[Analysis] Performing Failure Analysis on Validation Set...")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_preds)

    # Create analysis dataframe
    analysis_df = val_df[meta_cols].copy()
    analysis_df["error"] = errors

    # Compute correlation of features with error
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation of features with Prediction Error:")
    print(correlations)

    # 11. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 0.7141749705260098

    if final_val_auc > THRESHOLD:
        print(
            "\n[Submission] Validation metric exceeds threshold. Generating submission file..."
        )
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": final_test_preds,
            }
        )

        utils.ensure_directory(Config.SUBMISSION_PATH)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\n[Submission] Validation metric ({final_val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
