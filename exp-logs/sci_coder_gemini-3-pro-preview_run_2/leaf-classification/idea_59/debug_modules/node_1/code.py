import os
import shutil
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library.utils import set_seed, ensure_float64, clipped_log_loss, save_submission
from library.image_processing import (
    process_dataset,
    get_polarity_corrected_image,
    extract_morphometrics,
)
from library.transformers import (
    MarginalBasis,
    RotationalBasis,
    RobustBasis,
    FactorizedInteractionProjector,
)
from library.library import (
    load_dataset,
    build_expert_library,
    train_experts,
    predict_experts,
)
from library.selection import GreedySelector, train_and_predict_library

# Constants
WORKING_DIR = "./working/demo_execution"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


def run_demo():
    print("Initializing Library Demo...")

    # Setup working directory
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # ==========================================================================
    # 1. UTILITIES DEMONSTRATION
    # ==========================================================================
    print("\n[1] Testing Utilities...")

    # Test ensure_float64
    data_int = np.array([1, 2, 3], dtype=int)
    data_float = ensure_float64(data_int)
    assert data_float.dtype == np.float64, "ensure_float64 failed to convert to float64"
    print("  - ensure_float64: Passed")

    # Test clipped_log_loss
    # Ground truth: Class 0, Class 1
    y_true_demo = np.array([0, 1])
    # Predictions: High confidence correct
    y_pred_demo = np.array([[0.99, 0.01], [0.05, 0.95]])
    loss = clipped_log_loss(y_true_demo, y_pred_demo)
    print(f"  - clipped_log_loss: {loss:.6f}")
    assert loss < 0.1, "Log loss should be low for accurate predictions"

    # Test save_submission
    dummy_ids = [99991, 99992]
    dummy_classes = ["SpeciesA", "SpeciesB"]
    dummy_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    sub_path = os.path.join(WORKING_DIR, "dummy_submission.csv")

    save_submission(dummy_ids, dummy_classes, dummy_probs, sub_path)
    assert os.path.exists(sub_path), "Submission file was not created"
    df_sub = pd.read_csv(sub_path)
    assert df_sub.shape == (2, 3), "Submission file has incorrect shape"
    print(f"  - save_submission: Passed (Saved to {sub_path})")

    # ==========================================================================
    # 2. IMAGE PROCESSING DEMONSTRATION
    # ==========================================================================
    print("\n[2] Testing Image Processing...")

    # Create a synthetic binary image (100x100)
    # Background: Black (0), Object: White (255)
    syn_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(syn_img, (30, 30), (70, 70), 255, -1)

    # Test Polarity Correction
    # Case A: Already correct (White on Black) -> Should remain same
    pol_img = get_polarity_corrected_image(syn_img)
    assert np.mean(pol_img) < 127, "Polarity correction inverted a correct image"

    # Case B: Inverted (Black on White) -> Should invert back
    inv_img = cv2.bitwise_not(syn_img)
    corr_img = get_polarity_corrected_image(inv_img)
    # The result should be mostly black (low mean)
    assert np.mean(corr_img) < 127, "Polarity correction failed to fix inverted image"
    print("  - get_polarity_corrected_image: Passed")

    # Test Morphometric Extraction
    feats = extract_morphometrics(pol_img)
    print(f"  - extract_morphometrics: Extracted {len(feats)} features")
    assert len(feats) == 11, "Morphometrics should return 11 features (7 Hu + 4 Geo)"
    assert np.all(np.isfinite(feats)), "Features contain non-finite values"

    # Test Batch Processing (process_dataset)
    # We use the validation set as it is smaller.
    # We force load_cached_data=False to verify processing logic.
    print("  - Running process_dataset on validation set (this may take a moment)...")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")

    # This function creates a cache file in ./working/idea_59/
    df_morph = process_dataset(val_meta_path, "val_demo", load_cached_data=False)

    assert "id" in df_morph.columns
    assert df_morph.shape[1] == 12, "DataFrame should have id + 11 features"
    assert len(df_morph) > 0, "No data processed"
    print(f"  - process_dataset: Processed {len(df_morph)} images successfully")

    # ==========================================================================
    # 3. TRANSFORMERS DEMONSTRATION
    # ==========================================================================
    print("\n[3] Testing Transformers...")

    # Generate synthetic data matching the dataset structure (192 columns)
    # 64 Margin + 64 Shape + 64 Texture
    n_samples = 200
    n_features = 192
    X_syn = np.random.rand(n_samples, n_features)
    y_syn = np.random.randint(0, 10, n_samples)

    # Test MarginalBasis
    mb = MarginalBasis()
    X_mb = mb.fit_transform(X_syn)
    assert X_mb.shape == X_syn.shape
    print("  - MarginalBasis: Passed")

    # Test RotationalBasis
    rb = RotationalBasis()
    X_rb = rb.fit_transform(X_syn)
    assert X_rb.shape == X_syn.shape
    print("  - RotationalBasis: Passed")

    # Test RobustBasis
    rob = RobustBasis(n_quantiles=10)
    X_rob = rob.fit_transform(X_syn)
    assert X_rob.shape == X_syn.shape
    print("  - RobustBasis: Passed")

    # Test FactorizedInteractionProjector
    # This transformer splits the 192 features into 3 groups, projects them, and computes interactions.
    fip = FactorizedInteractionProjector(n_components=5)
    fip.fit(X_syn, y_syn)
    X_fip = fip.transform(X_syn)
    print(f"  - FactorizedInteractionProjector: Output shape {X_fip.shape}")
    assert X_fip.shape[0] == n_samples
    # Expected features: Poly(degree=2) of (5+5+5)=15 inputs.
    # N_out = (15+2 choose 2) - 1 (bias) = 136 - 1 = 135
    assert X_fip.shape[1] == 135, f"Expected 135 features, got {X_fip.shape[1]}"

    # ==========================================================================
    # 4. LIBRARY & SELECTION DEMONSTRATION
    # ==========================================================================
    print("\n[4] Testing Expert Library & Selection...")

    # Load real data subset for realistic testing
    # We use 'Global' scope to get the 192 raw features quickly
    print("  - Loading subset of training data...")
    X_global, y_global, ids_global = load_dataset("train", "Global")

    # Subset to 50 samples for speed
    subset_size = 50
    X_sub = X_global[:subset_size]
    y_sub = y_global[:subset_size]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_sub)
    n_classes = len(le.classes_)

    # Build the library of experts
    library = build_expert_library()
    print(f"  - Built library with {len(library)} experts")

    # Prepare Feature Dictionary
    # The library expects 'Global', 'Physical', and 'Factorized' keys.
    # We reuse X_sub for Global/Factorized.
    # For Physical, we create random mock data to avoid re-running image processing on train set.
    X_phys_mock = np.random.rand(subset_size, 11)

    X_dict = {"Global": X_sub, "Factorized": X_sub, "Physical": X_phys_mock}

    # Train Experts
    print("  - Training experts on subset...")
    trained_library = train_experts(library, X_dict, y_enc)

    # Predict (using the same subset for demonstration)
    print("  - Generating predictions...")
    preds_dict = predict_experts(trained_library, X_dict)

    # Verify prediction shape
    first_key = list(preds_dict.keys())[0]
    assert preds_dict[first_key].shape == (subset_size, n_classes)

    # Greedy Forward Selection
    print("  - Running Greedy Selector...")
    selector = GreedySelector(max_iter=5)
    selector.fit(preds_dict, y_enc)

    selected_experts, weights = selector.get_selected_experts_with_weights()
    print(f"    Selected: {selected_experts}")
    print(f"    Weights: {weights}")

    # Ensemble Prediction
    final_probs = selector.predict(preds_dict)
    assert final_probs.shape == (subset_size, n_classes)
    print("  - Ensemble prediction generated successfully")

    # ==========================================================================
    # 5. WRAPPER FUNCTION DEMONSTRATION
    # ==========================================================================
    print("\n[5] Testing Wrapper (train_and_predict_library)...")

    # This function encapsulates the train -> predict flow
    # We simulate a validation set using the same dictionary
    t_lib, v_preds = train_and_predict_library(X_dict, y_enc, X_dict, subset_size=20)

    assert len(t_lib) == len(library)
    assert len(v_preds) == len(library)
    print("  - Wrapper function executed successfully")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
