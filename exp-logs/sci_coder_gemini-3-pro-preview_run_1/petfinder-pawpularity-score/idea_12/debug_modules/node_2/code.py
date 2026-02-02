import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_handler import PetDataset, get_dataloader
from library.backbone_extractor import FeatureExtractor
from library.feature_processor import ExpertPreprocessor, create_interaction_matrix
from library.level0_experts import Level0Trainer
from library.level1_meta import MetaLearner


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n=== Step 1: Configuring for Fast Demonstration ===")

    # Modify Config for speed and debugging
    Config.DEBUG = True  # Limits dataset to 100 samples
    Config.N_FOLDS = 2  # Reduce CV folds
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce model complexity for speed
    Config.PCA_COMPONENTS = 10
    Config.RIDGE_ALPHAS = [1.0]
    Config.SVR_PARAMS = {
        "kernel": "rbf",
        "C": [1.0],
        "epsilon": [0.1],
        "gamma": "scale",
    }
    Config.ET_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["verbose"] = -1
    Config.META_MODEL_PARAMS["n_iter"] = 10

    # Clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    Config.setup()
    seed_everything(Config.SEED)

    print("Configuration updated. Debug mode enabled.")

    # ==========================================
    # 2. Data Handler Verification
    # ==========================================
    print("\n=== Step 2: Verifying Data Handler ===")

    # Test Dataset
    dataset = PetDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        model_name=Config.MODEL_CONVNEXT,
        augment=False,
    )

    # In DEBUG mode, dataset should be 100 samples
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 100, "Dataset length should be 100 in DEBUG mode"

    # Test Item
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    assert "id" in sample
    assert "pixel_values" in sample
    assert "meta_features" in sample
    assert "target" in sample

    # Check shapes (ConvNeXt uses 224x224)
    # pixel_values shape: (3, 224, 224)
    assert sample["pixel_values"].shape == (
        3,
        224,
        224,
    ), f"Unexpected image shape: {sample['pixel_values'].shape}"
    # meta_features shape: (12,)
    assert sample["meta_features"].shape == (
        12,
    ), f"Unexpected meta shape: {sample['meta_features'].shape}"

    # Test DataLoader
    loader = get_dataloader(
        metadata_path=Config.TRAIN_METADATA_PATH,
        model_name=Config.MODEL_CONVNEXT,
        batch_size=8,
        shuffle=False,
    )
    batch = next(iter(loader))
    print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
    assert batch["pixel_values"].shape == (8, 3, 224, 224)
    print("Data Handler verification passed.")

    # ==========================================
    # 3. Feature Extractor Verification
    # ==========================================
    print("\n=== Step 3: Verifying Feature Extractor ===")

    # Initialize Extractor (using ConvNeXt as it is generally smaller/faster than ViT-Large)
    extractor = FeatureExtractor(Config.MODEL_CONVNEXT)

    # Create a loader that matches the global configuration (specifically augmentation)
    # to ensure extracted features match the cached ones (which use Config.USE_FLIP_AUGMENTATION)
    loader_verif = get_dataloader(
        metadata_path=Config.TRAIN_METADATA_PATH,
        model_name=Config.MODEL_CONVNEXT,
        batch_size=8,
        shuffle=False,
        augment=Config.USE_FLIP_AUGMENTATION,
    )

    # Run extraction on the configured loader
    features, ids, meta, targets = extractor.extract(loader_verif)

    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted ids shape: {ids.shape}")

    # Assertions
    assert features.shape[0] == 100
    assert features.shape[1] > 0  # Embedding dim
    assert len(ids) == 100
    assert meta.shape == (100, 12)
    assert targets.shape == (100,)

    # Test caching mechanism
    print("Testing extract_and_cache...")
    f_cache, i_cache, m_cache, t_cache = extractor.extract_and_cache(
        split_name="train", metadata_path=Config.TRAIN_METADATA_PATH
    )

    # Verify files exist
    cache_path = Config.get_cache_path("convnext-large-224-22k-1k", "train")
    assert os.path.exists(cache_path), "Feature cache file not created"
    assert np.array_equal(
        features, f_cache
    ), "Cached features do not match extracted features"
    print("Feature Extractor verification passed.")

    # ==========================================
    # 4. Feature Processor Verification
    # ==========================================
    print("\n=== Step 4: Verifying Feature Processor ===")

    # Create dummy data
    n_samples = 50
    n_emb = 128
    n_meta = 12
    dummy_emb = np.random.rand(n_samples, n_emb)
    dummy_meta = np.random.randint(0, 2, (n_samples, n_meta))

    # Test Linear Preprocessor
    prep_linear = ExpertPreprocessor()
    prep_linear.fit_linear(dummy_emb, dummy_meta)
    out_linear = prep_linear.transform_linear(dummy_emb, dummy_meta)
    assert out_linear.shape == (n_samples, n_emb + n_meta)

    # Test Tree Preprocessor (PCA)
    prep_tree = ExpertPreprocessor(pca_components=10)
    prep_tree.fit_tree(dummy_emb)
    out_tree = prep_tree.transform_tree(dummy_emb, dummy_meta)
    assert out_tree.shape == (n_samples, 10 + n_meta)

    # Test Interaction Matrix
    # Assume 2 experts
    dummy_preds = np.random.rand(n_samples, 2)
    interaction_matrix = create_interaction_matrix(dummy_preds, dummy_meta)
    # Expected cols: Experts(2) + Meta(12) + Interactions(2*12) = 38
    print(f"Interaction Matrix shape: {interaction_matrix.shape}")
    assert interaction_matrix.shape == (n_samples, 2 + 12 + 24)
    print("Feature Processor verification passed.")

    # ==========================================
    # 5. Level-0 Expert Training Verification
    # ==========================================
    print("\n=== Step 5: Verifying Level-0 Trainer ===")

    trainer = Level0Trainer()

    # Use the extracted features from Step 3
    # We need to simulate train/test split.
    # In this demo, we'll just use the same data for "train" and "test" inputs to the function
    # to verify it runs.

    oof, test_preds = trainer.train_expert(
        backbone_name="convnext_demo",
        expert_name="ridge",
        train_embeddings=features,
        train_metadata=meta,
        train_targets=targets,
        test_embeddings=features,  # using train as test for shape check
        test_metadata=meta,
        load_cached_data=False,
    )

    print(f"OOF shape: {oof.shape}")
    print(f"Test Preds shape: {test_preds.shape}")

    assert oof.shape == (100,)
    assert test_preds.shape == (100,)

    # Verify cache files created
    oof_path = os.path.join(Config.WORKING_DIR, "convnext_demo_ridge_oof.npy")
    assert os.path.exists(oof_path)
    print("Level-0 Trainer verification passed.")

    # ==========================================
    # 6. Meta-Learner (Full Pipeline) Verification
    # ==========================================
    print("\n=== Step 6: Verifying Meta-Learner (Full Pipeline) ===")

    meta_learner = MetaLearner()

    # Monkey-patch _get_backbone_list to ONLY use ConvNeXt for speed
    # This prevents loading SigLIP and DINOv2 which would take too long
    def mocked_get_backbone_list(self):
        return [Config.MODEL_CONVNEXT]

    # Bind the mock method to the instance
    meta_learner._get_backbone_list = mocked_get_backbone_list.__get__(
        meta_learner, MetaLearner
    )

    # Monkey-patch _get_expert_list to reduce experts for speed (skip SVR as it can be slow)
    def mocked_get_expert_list(self):
        return ["ridge", "lgbm"]

    meta_learner._get_expert_list = mocked_get_expert_list.__get__(
        meta_learner, MetaLearner
    )

    print(
        "Starting Meta-Learner training (this involves feature extraction, L0 training, and L1 training)..."
    )
    meta_learner.train_predict(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())

    # Check submission length (should be 100 because test set is also limited by DEBUG=True)
    assert len(submission_df) == 100
    assert Config.ID_COL in submission_df.columns
    assert Config.TARGET_COL in submission_df.columns

    print("Meta-Learner verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
