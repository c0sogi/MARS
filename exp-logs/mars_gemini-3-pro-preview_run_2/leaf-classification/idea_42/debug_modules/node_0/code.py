import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided libraries
from library.config import Config
from library.morphology import process_single_image, get_morphometric_features
from library.pipelines import get_topology
from library.model_wrapper import get_lda_model
from library.data_loader import DataManager
from library.ensemble_selector import GreedySelector


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("[1/5] Setting up configuration...")
    np.random.seed(42)

    # Override Config for speed and demo purposes
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for fast execution
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR

    # Ensure clean slate for demo output
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    print(f"Working directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Test Morphology Module
    # ==========================================
    print("[2/5] Testing Morphology Module...")

    # Test single image processing
    # We read the metadata directly to find a valid image path
    train_meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_image_rel_path = train_meta_df.iloc[0]["image_path"]

    print(f"Processing single image: {sample_image_rel_path}")
    features_single = process_single_image(sample_image_rel_path)

    # Expecting 7 Hu moments + 4 Geometric properties = 11 features
    assert features_single.shape == (
        11,
    ), f"Morphology feature shape mismatch. Expected (11,), got {features_single.shape}"
    assert np.all(
        np.isfinite(features_single)
    ), "Morphology features contain non-finite values."

    # Test Batch Extraction with Caching
    print("Extracting batch features (Train)...")
    features_batch = get_morphometric_features(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=False
    )
    assert features_batch.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        11,
    ), f"Batch shape mismatch. Expected ({Config.DEBUG_SAMPLE_SIZE}, 11), got {features_batch.shape}"

    # Verify cache file creation
    cache_path = os.path.join(Config.WORKING_DIR, "morphometric_features_train.npy")
    assert os.path.exists(cache_path), "Morphology cache file was not created."

    # ==========================================
    # 3. Test Data Loader Module
    # ==========================================
    print("[3/5] Testing Data Loader Module...")

    # Load Train
    data_train = DataManager.load_split("train", load_cached_data=True)
    X_train_global = data_train["X_global"]
    X_train_combined = data_train["X_combined"]
    y_train = data_train["y"]
    ids_train = data_train["ids"]

    assert X_train_global.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        192,
    ), "X_global train shape incorrect."
    assert X_train_combined.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        203,
    ), "X_combined train shape incorrect (192+11)."
    assert len(y_train) == Config.DEBUG_SAMPLE_SIZE, "y_train length incorrect."

    # Load Val
    data_val = DataManager.load_split("val", load_cached_data=False)
    X_val_global = data_val["X_global"]
    y_val = data_val["y"]

    assert (
        len(y_val) == Config.DEBUG_SAMPLE_SIZE
    ), "y_val length incorrect (subsampled)."

    # ==========================================
    # 4. Test Pipelines and Model Wrapper
    # ==========================================
    print("[4/5] Testing Pipelines and Model Wrapper...")

    # Due to heavy subsampling (100 rows), the number of classes in train and val might differ.
    # For the purpose of this demo, we filter the validation set to only include classes
    # present in the training set to avoid dimension mismatches in LDA.

    train_classes = np.unique(y_train)
    val_mask = np.isin(y_val, train_classes)

    if np.sum(val_mask) < 2:
        print(
            "Warning: Too few overlapping classes between subsampled train/val. Creating synthetic val set for model test."
        )
        # Synthetic validation data drawn from train
        X_val_subset = X_train_global[:10].copy()
        y_val_subset = y_train[:10].copy()
    else:
        X_val_subset = X_val_global[val_mask]
        y_val_subset = y_val[val_mask]

    # Define a test configuration
    topology_name = "marginal"  # Simple Yeo-Johnson
    shrinkage = "auto"  # OAS

    print(f"Running Topology: {topology_name} with Shrinkage: {shrinkage}")

    # 1. Pipeline Transform
    pipeline = get_topology(topology_name)
    X_train_trans = pipeline.fit_transform(X_train_global)
    X_val_trans = pipeline.transform(X_val_subset)

    assert (
        X_train_trans.shape == X_train_global.shape
    ), "Pipeline transformed shape mismatch."

    # 2. Model Fit & Predict
    lda_model = get_lda_model(shrinkage)
    lda_model.fit(X_train_trans, y_train)

    preds_proba = lda_model.predict_proba(X_val_trans)

    # Check predictions shape: (n_val_samples, n_classes_in_train)
    assert preds_proba.shape == (
        len(y_val_subset),
        len(lda_model.classes_),
    ), f"Prediction shape mismatch. Got {preds_proba.shape}"

    print("Model training and prediction successful.")

    # ==========================================
    # 5. Test Ensemble Selector
    # ==========================================
    print("[5/5] Testing Ensemble Selector...")

    # We use a synthetic example to verify the Greedy Logic deterministically
    # independent of the model performance on the subsampled data.

    # Synthetic Ground Truth (3 classes, 10 samples)
    y_true_syn = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    classes_syn = np.array([0, 1, 2])
    n_samples_syn = len(y_true_syn)

    # Expert A: Perfect Predictions
    expert_a = np.zeros((n_samples_syn, 3))
    for i, c in enumerate(y_true_syn):
        expert_a[i, c] = 1.0

    # Expert B: Uniform Random (High Entropy)
    expert_b = np.ones((n_samples_syn, 3)) / 3.0

    # Expert C: Wrong Predictions (High Loss)
    expert_c = np.zeros((n_samples_syn, 3))
    for i, c in enumerate(y_true_syn):
        expert_c[i, (c + 1) % 3] = 1.0  # Shifted class

    predictions_dict = {
        "Expert_Perfect": expert_a,
        "Expert_Uniform": expert_b,
        "Expert_Wrong": expert_c,
    }

    selector = GreedySelector()

    # Fit Selector
    # We expect it to pick Expert_Perfect first and stop or keep it.
    selector.fit(predictions_dict, y_true_syn, max_iterations=3, verbose=False)

    print(f"Selected Experts: {selector.selected_experts}")
    print(f"Best Score: {selector.best_score}")

    # Assertions
    assert len(selector.selected_experts) > 0, "Selector failed to select any expert."
    assert (
        "Expert_Perfect" in selector.selected_experts
    ), "Selector failed to pick the perfect expert."
    assert (
        selector.best_score < 0.01
    ), f"Score should be near 0 for perfect expert, got {selector.best_score}"

    # Test Prediction Aggregation
    final_preds = selector.predict(predictions_dict)
    assert final_preds.shape == expert_a.shape, "Final prediction shape mismatch."

    # Check Row Sums (Normalization)
    row_sums = final_preds.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0
    ), "Final predictions are not normalized to sum to 1."

    print("Ensemble Selector logic verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
