import os
import sys
import numpy as np
import torch
import pandas as pd
import warnings
import shutil

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_rmse
from library.dataset import PawpularityDataset, get_transforms
from library.feature_extraction import extract_and_cache_features
from library.stacking_models import Level0Expert, Level1MetaLearner
from library.workflow import StackingManager


def demo_dataset():
    """
    Demonstrates the PawpularityDataset class and image transformations.
    """
    print("\n=== Demo: Dataset Loading and Processing ===")

    # Define a transform pipeline (resize to 224x224, normalize)
    transform = get_transforms("swin_large", split="valid")

    # Initialize dataset with training metadata
    dataset = PawpularityDataset(
        csv_path=Config.TRAIN_METADATA_PATH, transform=transform, return_target=True
    )

    print(f"Dataset initialized with {len(dataset)} samples.")

    # Fetch a single sample
    sample = dataset[0]

    # Verify keys
    expected_keys = {"image", "features", "id", "target"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in sample. Found: {sample.keys()}"

    # Verify Image Tensor
    # Shape should be (3, 224, 224) due to the transform
    assert sample["image"].shape == (
        3,
        224,
        224,
    ), f"Incorrect image shape: {sample['image'].shape}"
    assert sample["image"].dtype == torch.float32, "Image tensor should be float32"

    # Verify Metadata Features
    # Config.METADATA_COLS has 12 columns
    assert sample["features"].shape == (
        12,
    ), f"Incorrect features shape: {sample['features'].shape}"

    # Verify Target
    assert sample["target"].shape == (
        1,
    ), f"Incorrect target shape: {sample['target'].shape}"

    print("Dataset verification passed: Shapes and types are correct.")


def demo_feature_extraction():
    """
    Demonstrates feature extraction using a pre-trained backbone.
    Uses a small subset to ensure speed.
    """
    print("\n=== Demo: Feature Extraction ===")

    model_key = "swin_large"
    subset_size = 10

    # Run extraction
    # This will download the model (if not cached) and run inference on 10 images
    data = extract_and_cache_features(
        model_key=model_key,
        split="train",
        load_cached_data=False,  # Force run to demonstrate logic
        subset_size=subset_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Verify output structure
    assert "features" in data
    assert "meta" in data
    assert "targets" in data
    assert "ids" in data

    # Verify shapes
    # Swin Large output dim is typically 1536
    n_samples = len(data["features"])
    assert n_samples == subset_size, f"Expected {subset_size} samples, got {n_samples}"
    assert data["features"].ndim == 2, "Features should be 2D array"
    assert data["meta"].shape == (subset_size, 12), "Meta features shape mismatch"
    assert data["targets"].shape == (subset_size, 1), "Targets shape mismatch"

    print(f"Feature extraction successful. Feature shape: {data['features'].shape}")


def demo_models():
    """
    Demonstrates Level 0 Expert and Level 1 Meta-Learner using synthetic data.
    """
    print("\n=== Demo: Stacking Models ===")

    # --- Synthetic Data Generation ---
    n_samples = 50
    img_dim = 1536
    meta_dim = 12

    X_img = np.random.rand(n_samples, img_dim).astype(np.float32)
    X_meta = np.random.randint(0, 2, (n_samples, meta_dim)).astype(np.float32)
    y = np.random.uniform(10, 90, (n_samples,)).astype(np.float32)

    # --- Level 0 Expert Demo ---
    print("Testing Level 0 Expert...")
    expert = Level0Expert()

    # Fit
    expert.fit(X_img, X_meta, y)
    assert expert.is_fitted, "Expert should be fitted after calling fit()"

    # Predict
    preds = expert.predict(X_img, X_meta)
    assert preds.shape == (n_samples,), "Prediction shape mismatch"
    assert np.all((preds >= 0) & (preds <= 100)), "Predictions out of bounds [0, 100]"

    # Save/Load
    save_path = os.path.join(Config.WORKING_DIR, "demo_expert.joblib")
    expert.save(save_path)

    expert_loaded = Level0Expert()
    expert_loaded.load(save_path)
    assert expert_loaded.is_fitted, "Loaded expert should be fitted"

    # --- Level 1 Meta-Learner Demo ---
    print("Testing Level 1 Meta-Learner...")
    n_experts = 3
    # Synthetic expert predictions (N samples, M experts)
    expert_preds = np.random.uniform(20, 80, (n_samples, n_experts)).astype(np.float32)

    meta_learner = Level1MetaLearner(positive=True)
    meta_learner.fit(expert_preds, y)

    # Check positive constraint (coefficients should be >= 0)
    # Note: Scikit-learn's positive=True might result in some 0s or very small negatives due to float precision,
    # but generally enforces positivity.
    assert np.all(
        meta_learner.model.coef_ >= -1e-5
    ), "Coefficients should be non-negative"

    final_preds = meta_learner.predict(expert_preds)
    assert final_preds.shape == (n_samples,), "Final prediction shape mismatch"

    print("Model component verification passed.")


def demo_workflow():
    """
    Demonstrates the full StackingManager workflow.
    Uses a small subset and fewer folds to complete quickly.
    """
    print("\n=== Demo: Full Workflow (StackingManager) ===")

    # Use a small subset size to ensure this completes within strict time limits
    subset_size = 30
    n_folds = 2

    manager = StackingManager(
        subset_size=subset_size,
        load_cached_data=True,  # Use cache if available (from previous steps) or create new
        n_folds=n_folds,
        working_dir=Config.WORKING_DIR,
    )

    # 1. Train Level 0 Experts
    # This iterates through all models in Config.MODEL_CONFIGS
    manager.train_level_0()

    # Verify intermediate files exist
    for model_key in manager.model_keys:
        oof_path = os.path.join(Config.WORKING_DIR, f"{model_key}_oof.npy")
        assert os.path.exists(oof_path), f"OOF predictions for {model_key} not found"

    # 2. Train Level 1 Meta-Learner
    manager.train_level_1()
    assert manager.meta_learner is not None, "Meta learner not initialized"

    # 3. Generate Submission
    manager.predict()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(df_sub) == subset_size
    ), f"Submission should have {subset_size} rows (subset mode)"
    assert "Id" in df_sub.columns and "Pawpularity" in df_sub.columns

    print("Full workflow executed successfully.")


if __name__ == "__main__":
    # Setup
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    try:
        # Run all demonstrations
        demo_dataset()
        demo_feature_extraction()
        demo_models()
        demo_workflow()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print traceback for debugging if needed, but keeping output clean as requested
        import traceback

        traceback.print_exc()
        exit(1)
