import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_extractor
from library import model_factory
from library import optimization


def main():
    print("=== Starting Library Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("--- 1. Configuration Setup ---")
    # Redirect cache to a demo folder to avoid conflicts
    DEMO_CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_cache")
    config.CACHE_DIR = DEMO_CACHE_DIR
    config.BATCH_SIZE = 4  # Small batch size for demo

    # Ensure clean state
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    print(f"Cache directory set to: {config.CACHE_DIR}")
    print("Configuration updated successfully.\n")

    # ---------------------------------------------------------
    # 2. Utils Demonstration
    # ---------------------------------------------------------
    print("--- 2. Utils Demonstration ---")

    # Test Seeding
    utils.seed_everything(seed=123)
    print("Random seed set to 123.")

    # Test Submission Formatting
    dummy_ids = [101, 102, 103]
    dummy_classes = ["Species_A", "Species_B"]
    # Create probs that need clipping (0.0 and 1.0) and normalization check
    dummy_probs = np.array([[0.0, 1.0], [0.5, 0.5], [0.9, 0.1]])

    demo_sub_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    utils.format_submission(
        dummy_ids, dummy_classes, dummy_probs, output_path=demo_sub_path
    )

    # Verify file content
    assert os.path.exists(demo_sub_path), "Submission file was not created."
    df_demo = pd.read_csv(demo_sub_path)
    assert df_demo.shape == (3, 3), f"Expected shape (3, 3), got {df_demo.shape}"
    assert "id" in df_demo.columns, "Column 'id' missing."
    # Check clipping (should not be exactly 0 or 1)
    assert (df_demo.iloc[0, 1] > 0.0) and (
        df_demo.iloc[0, 2] < 1.0
    ), "Probabilities were not clipped."
    print("Submission formatting verified.\n")

    # ---------------------------------------------------------
    # 3. Data Loader Demonstration
    # ---------------------------------------------------------
    print("--- 3. Data Loader Demonstration ---")

    # Load metadata (force refresh to test loading logic)
    data_dict = data_loader.load_data(load_cached_data=False)

    # Verify keys
    required_keys = [
        "train_paths",
        "train_tabular",
        "train_labels",
        "train_ids",
        "classes",
    ]
    for k in required_keys:
        assert k in data_dict, f"Missing key {k} in loaded data."

    print(f"Loaded {len(data_dict['train_paths'])} training samples.")
    print(f"Number of classes: {len(data_dict['classes'])}")

    # Create a small subset for testing (10 samples)
    subset_size = 10
    subset_paths = data_dict["train_paths"][:subset_size]
    subset_tab = data_dict["train_tabular"][:subset_size]
    subset_labels = data_dict["train_labels"][:subset_size]
    subset_ids = data_dict["train_ids"][:subset_size]

    # Instantiate Dataset
    dataset = data_loader.LeafDataset(
        paths=subset_paths, tabular=subset_tab, labels=subset_labels, ids=subset_ids
    )

    # Verify __getitem__
    img, tab, label, img_id = dataset[0]

    # Check Image Shape: (4 views, 3 channels, 224, 224)
    assert img.shape == (
        4,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Incorrect image tensor shape: {img.shape}"

    # Check Tabular Shape: (192 features)
    assert tab.shape == (192,), f"Incorrect tabular feature shape: {tab.shape}"

    # Check ID match
    assert img_id == subset_ids[0], "ID mismatch in dataset retrieval."

    print("Dataset shapes verified: Images (4, 3, 224, 224), Tabular (192,).")

    # Create DataLoader
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    print("DataLoader created successfully.\n")

    # ---------------------------------------------------------
    # 4. Feature Extractor Demonstration
    # ---------------------------------------------------------
    print("--- 4. Feature Extractor Demonstration ---")

    # Initialize Extractor (uses GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extractor = feature_extractor.DualStreamExtractor(device=device)

    # Extract features for the subset
    # Note: This will download models if not present, but they are standard HF models.
    features_out = extractor.extract_features(
        loader, "demo_subset", load_cached_data=False
    )

    dino_feats = features_out["dino_features"]
    conv_feats = features_out["conv_features"]
    ids_out = features_out["ids"]
    labels_out = features_out["labels"]

    # Verify shapes
    # DINOv2-large usually 1024 dim, ConvNeXt-large usually 1536 dim
    assert dino_feats.shape[0] == subset_size, "DINO feature count mismatch."
    assert conv_feats.shape[0] == subset_size, "ConvNeXt feature count mismatch."
    assert dino_feats.ndim == 2, "DINO features should be 2D."
    assert conv_feats.ndim == 2, "ConvNeXt features should be 2D."
    assert np.array_equal(ids_out, subset_ids), "ID order mismatch after extraction."

    print(f"Extracted DINO shape: {dino_feats.shape}")
    print(f"Extracted ConvNeXt shape: {conv_feats.shape}")
    print("Feature extraction verified.\n")

    # ---------------------------------------------------------
    # 5. Model Factory Demonstration
    # ---------------------------------------------------------
    print("--- 5. Model Factory Demonstration ---")

    # Initialize FidelityBranch
    # Using a high variance threshold to ensure some reduction happens but not too much for small data
    branch = model_factory.FidelityBranch(pca_variance=0.99, quantile_dist="normal")

    # Fit model
    # Note: LDA requires num_samples > num_classes usually, or shrinkage.
    # Since we have 10 samples and ~99 classes, we need to handle this.
    # For the demo, we will remap labels to a smaller range [0, 1] just to make LDA fit without error
    # on this tiny subset.
    demo_labels = np.random.randint(0, 2, size=subset_size)

    print("Fitting FidelityBranch model...")
    branch.fit(dino_feats, conv_feats, subset_tab, demo_labels)
    assert branch.is_fitted, "Model should be marked as fitted."

    # Predict
    probs = branch.predict_proba(dino_feats, conv_feats, subset_tab)

    # Verify output
    # Should be (10, 2) because we used dummy labels with 2 classes
    assert probs.shape == (
        subset_size,
        2,
    ), f"Expected probs shape (10, 2), got {probs.shape}"

    # Save and Load
    model_path = os.path.join(config.WORKING_DIR, "demo_model.pkl")
    branch.save(model_path)
    assert os.path.exists(model_path), "Model file not saved."

    loaded_branch = model_factory.FidelityBranch.load(model_path)
    assert loaded_branch.is_fitted, "Loaded model lost fitted state."

    print("Model training, prediction, and persistence verified.\n")

    # ---------------------------------------------------------
    # 6. Optimization Demonstration
    # ---------------------------------------------------------
    print("--- 6. Ensemble Optimization Demonstration ---")

    optimizer = optimization.EnsembleOptimizer(step_size=0.1)

    # Create synthetic OOF predictions for 2 branches
    # 10 samples, 3 classes
    n_samples_opt = 10
    n_classes_opt = 3

    # Branch 1 is very confident on the correct class (index 0)
    preds_1 = np.zeros((n_samples_opt, n_classes_opt))
    preds_1[:, 0] = 0.9
    preds_1[:, 1] = 0.05
    preds_1[:, 2] = 0.05

    # Branch 2 is uniform (uncertain)
    preds_2 = np.full((n_samples_opt, n_classes_opt), 1.0 / n_classes_opt)

    # True labels are all class 0
    y_true_opt = np.zeros(n_samples_opt, dtype=int)

    # Optimize
    # We expect Branch 1 to get higher weight because it is more accurate
    weights = optimizer.optimize([preds_1, preds_2], y_true_opt, load_cached_data=False)

    # Verify weights
    assert len(weights) == 2, "Should return 2 weights."
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1."
    assert weights[0] > weights[1], "Optimizer failed to favor the better model."

    print(f"Optimal Weights: {weights}")

    # Verify caching
    cache_path = os.path.join(config.CACHE_DIR, "ensemble_weights.npy")
    assert os.path.exists(cache_path), "Weights were not cached."

    print("Ensemble optimization verified.\n")

    # ---------------------------------------------------------
    # 7. Cleanup
    # ---------------------------------------------------------
    print("--- 7. Cleanup ---")
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
        print("Demo cache cleaned up.")

    print("\n=== Demonstration Complete: All tests passed successfully. ===")


if __name__ == "__main__":
    main()
