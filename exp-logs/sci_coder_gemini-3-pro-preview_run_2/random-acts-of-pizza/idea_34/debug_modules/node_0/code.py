import sys
import os
import numpy as np
import pandas as pd
import warnings
import shutil

# Add current directory to path
sys.path.append(".")

# Import library modules
from library.utils import set_seed
from library.data_loader import DataLoader
from library.embedding_manager import EmbeddingManager
from library.feature_engineering import JointPCATransformer, build_jbpce_pipeline
from library.trainer import Trainer

# Configuration
warnings.filterwarnings("ignore")
set_seed(42)


def demo_components_manual():
    """
    Demonstrates how to use individual components: DataLoader, EmbeddingManager,
    and Feature Engineering classes.
    """
    print("=" * 40)
    print("1. Component-Level Demonstration")
    print("=" * 40)

    # --- 1. Data Loading ---
    print("\n[DataLoader] Loading subset of data...")
    # Initialize DataLoader with a specific cache directory
    dl = DataLoader(cache_dir="./working/demo_components")

    # Load a small subset (100 samples) for quick verification
    train_json, test_json, train_meta, val_meta, test_meta = dl.load_data(
        debug_limit=100
    )

    print(f"  Train JSON entries: {len(train_json)}")
    print(f"  Test JSON entries: {len(test_json)}")

    # Validate loaded data
    assert len(train_json) == 100, "DataLoader failed to limit train data"
    assert len(test_json) == 100, "DataLoader failed to limit test data"
    assert not train_meta.empty, "Train metadata should not be empty"

    # Extract features
    print("[DataLoader] Extracting text and metadata...")
    train_texts = dl.extract_text_data(train_json)
    train_meta_feats = dl.extract_metadata(train_json)

    print(f"  Extracted Text Count: {len(train_texts)}")
    print(f"  Metadata Shape: {train_meta_feats.shape}")

    assert len(train_texts) == 100
    assert train_meta_feats.shape == (100, 10), "Metadata should have 10 features"

    # --- 2. Embedding Generation ---
    print("\n[EmbeddingManager] Computing embeddings...")
    em = EmbeddingManager(cache_dir="./working/demo_components")

    # Compute embeddings (using small batch size for demo)
    # Note: This will download models on first run
    emb_a, emb_b = em.get_dual_backbone_embeddings(
        train_texts, "demo_subset", batch_size=16
    )

    print(f"  Embedding A (MiniLM) Shape: {emb_a.shape}")
    print(f"  Embedding B (MPNet) Shape: {emb_b.shape}")

    # Verify shapes (MiniLM=384, MPNet=768)
    assert emb_a.shape == (100, 384)
    assert emb_b.shape == (100, 768)

    # --- 3. Feature Engineering ---
    print("\n[FeatureEngineering] Verifying JointPCATransformer...")
    # Concatenate embeddings as required by the transformer
    X_emb_combined = np.hstack([emb_a, emb_b])

    # Instantiate Transformer with small n_components compatible with N=100
    jpca = JointPCATransformer(n_components=10, split_index=384, random_state=42)

    # Fit and Transform
    X_pca = jpca.fit_transform(X_emb_combined)
    print(f"  Projected Shape: {X_pca.shape}")

    assert X_pca.shape == (100, 10), "PCA projection shape mismatch"

    # --- 4. Pipeline Construction ---
    print("\n[FeatureEngineering] Building and Fitting JBPCE Pipeline...")
    pipeline = build_jbpce_pipeline(
        emb_dim_a=384,
        emb_dim_b=768,
        pca_components=10,  # Small for demo
        n_estimators=2,  # Minimal ensemble
        max_samples=0.5,
    )

    # Prepare full input matrix [Embeddings | Metadata]
    X_full = np.hstack([X_emb_combined, train_meta_feats])
    # Generate dummy labels
    y_dummy = np.random.randint(0, 2, size=100)

    # Fit pipeline
    pipeline.fit(X_full, y_dummy)
    print("  Pipeline fitted successfully.")

    # Predict
    probs = pipeline.predict_proba(X_full)[:, 1]
    print(f"  Sample Predictions: {probs[:3]}")
    assert len(probs) == 100


def demo_trainer_end_to_end():
    """
    Demonstrates the Trainer class which orchestrates the entire workflow.
    """
    print("\n" + "=" * 40)
    print("2. End-to-End Trainer Demonstration")
    print("=" * 40)

    # We use debug_limit=200.
    # Reason: The Trainer's internal grid search explores PCA n_components=[100, 150, 200].
    # We need enough samples in the training folds of the CV to support at least 100 components.
    # With 5-fold CV, train size is 4/5 * N.
    # If N=200, train size=160. Internal 3-fold CV train size ~106. 106 > 100, so it's safe.
    debug_limit = 200

    print(f"[Trainer] Initializing with debug_limit={debug_limit}...")
    trainer = Trainer(cache_dir="./working/demo_trainer")

    # Run the full training and prediction loop
    # This handles: Loading -> Embeddings -> Stratified CV -> Grid Search -> Inference -> CSV Save
    trainer.train_and_predict(debug_limit=debug_limit)

    # Verify Submission
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        print(f"\n[Trainer] Submission saved to {submission_path}")
        df_sub = pd.read_csv(submission_path)
        print(f"  Submission Shape: {df_sub.shape}")
        print("  Head:")
        print(df_sub.head())

        # Verify that we have predictions for the test samples loaded
        # Since we limited input test.json to 200, we expect 200 predictions
        assert (
            len(df_sub) == debug_limit
        ), f"Expected {debug_limit} predictions, got {len(df_sub)}"

        # Verify probability range
        assert df_sub["requester_received_pizza"].min() >= 0
        assert df_sub["requester_received_pizza"].max() <= 1
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    try:
        # Run manual component check
        demo_components_manual()

        # Run full automated workflow
        demo_trainer_end_to_end()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e
