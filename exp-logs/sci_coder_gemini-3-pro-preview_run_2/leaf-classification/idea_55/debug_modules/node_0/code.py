import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Import from the provided library
from library.config import RANDOM_SEED, METADATA_DIR, INPUT_DIR, FEATURE_SLICES
from library.utils import set_seed, clipped_log_loss
from library.image_processing import process_single_image
from library.data_factory import get_data_splits
from library.custom_transformers import StratifiedDiscriminantProjector
from library.model_factory import build_expert_library
from library.ensemble_optimizer import GreedySelector


def run_demo():
    # 1. Setup
    print(">>> 1. Initializing and Setting Seeds...")
    set_seed(RANDOM_SEED)

    # 2. Verify Utils (Metric)
    print("\n>>> 2. Verifying Utility Functions (Log Loss)...")
    # Case 1: Perfect prediction
    y_true_dummy = np.array([0, 1, 2])
    y_pred_dummy = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss_perfect = clipped_log_loss(y_true_dummy, y_pred_dummy)
    print(f"   Perfect Prediction Loss: {loss_perfect:.6f} (Expected ~0.0)")
    assert loss_perfect < 1e-6, "Log loss for perfect prediction should be near zero."

    # Case 2: Uniform prediction (High uncertainty)
    y_pred_uniform = np.array(
        [[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]
    )
    loss_uniform = clipped_log_loss(y_true_dummy, y_pred_uniform)
    expected_uniform = -np.log(0.33)
    print(
        f"   Uniform Prediction Loss: {loss_uniform:.6f} (Expected ~{expected_uniform:.6f})"
    )
    assert np.isclose(
        loss_uniform, expected_uniform, atol=0.01
    ), "Log loss calculation mismatch."

    # 3. Verify Image Processing
    print("\n>>> 3. Verifying Image Processing...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_train = pd.read_csv(train_meta_path)
        # Pick the first image
        first_img_rel_path = df_train.iloc[0]["image_path"]
        first_img_full_path = os.path.join(INPUT_DIR, first_img_rel_path)

        print(f"   Processing image: {first_img_full_path}")
        features = process_single_image(first_img_full_path)

        print(f"   Extracted Feature Vector Shape: {features.shape}")
        print(f"   First 5 features: {features[:5]}")

        assert features.shape == (
            12,
        ), "Image processing should return a 12-element vector."
        assert features.dtype == np.float64, "Features should be float64."
        assert not np.all(
            features == 0
        ), "Features should not be all zeros for a valid image."
    else:
        print("   Warning: Metadata file not found, skipping image processing check.")

    # 4. Verify Data Loading
    print("\n>>> 4. Verifying Data Factory...")
    # This will load from cache if available or compute from scratch
    # We use load_cached_data=True to be efficient, but logic handles scratch computation
    X_train, y_train, X_val, y_val, classes = get_data_splits(load_cached_data=True)

    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   y_val shape:   {y_val.shape}")
    print(f"   Classes:       {len(classes)} unique species")

    # Assertions
    assert (
        X_train.shape[1] == 204
    ), "Feature matrix should have 204 columns (192 Global + 12 Morph)."
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels."
    assert len(X_val) == len(y_val), "Mismatch in validation samples and labels."
    assert len(classes) == 99, "Expected 99 classes."

    # 5. Verify Custom Transformer (StratifiedDiscriminantProjector)
    print("\n>>> 5. Verifying StratifiedDiscriminantProjector...")
    # The projector works on the Global Features (first 192 columns)
    X_global_train = X_train[:, :192]

    sdp = StratifiedDiscriminantProjector(shrinkage=0.1, n_components=None)
    sdp.fit(X_global_train, y_train)

    X_projected = sdp.transform(X_global_train)
    print(f"   Input Shape: {X_global_train.shape}")
    print(f"   Projected Shape: {X_projected.shape}")

    # Expected output dimension:
    # 3 groups (margin, shape, texture).
    # For each group, LDA produces min(n_classes-1, n_features) components.
    # n_classes = 99.
    # margin (64 feats) -> min(98, 64) = 64
    # shape (64 feats) -> min(98, 64) = 64
    # texture (64 feats) -> min(98, 64) = 64
    # Total = 64 + 64 + 64 = 192.
    # Note: If n_components was set or features were fewer, this would differ.

    expected_dim = 0
    for group in ["margin", "shape", "texture"]:
        n_feats = FEATURE_SLICES[group].stop - FEATURE_SLICES[group].start
        expected_dim += min(len(classes) - 1, n_feats)

    print(f"   Expected Projected Dimension: {expected_dim}")
    assert (
        X_projected.shape[1] == expected_dim
    ), f"Projected dimension mismatch. Got {X_projected.shape[1]}, expected {expected_dim}."

    # 6. Verify Model Factory
    print("\n>>> 6. Verifying Model Factory...")
    experts = build_expert_library()
    print(f"   Built {len(experts)} experts in the library.")

    # List a few names
    expert_names = [name for name, _ in experts]
    print(f"   Sample Expert Names: {expert_names[:3]} ... {expert_names[-1]}")

    assert len(experts) > 0, "Expert library is empty."

    # 7. Verify Ensemble Optimizer (GreedySelector)
    print("\n>>> 7. Verifying Ensemble Optimizer (GreedySelector)...")

    # To save time, we select a small subset of experts for the demo
    # We pick one from each Topology type if available
    demo_experts = []

    # TopoA example
    topo_a = next((e for e in experts if "TopoA" in e[0]), None)
    if topo_a:
        demo_experts.append(topo_a)

    # TopoB example
    topo_b = next((e for e in experts if "TopoB" in e[0]), None)
    if topo_b:
        demo_experts.append(topo_b)

    # TopoC example
    topo_c = next((e for e in experts if "TopoC" in e[0]), None)
    if topo_c:
        demo_experts.append(topo_c)

    print(
        f"   Running GreedySelector with {len(demo_experts)} experts for demonstration..."
    )

    selector = GreedySelector(
        experts=demo_experts,
        max_iterations=3,  # Small number for speed
        tolerance=1e-6,
        verbose=True,
    )

    # Fit the selector
    selector.fit(X_train, y_train, X_val, y_val)

    selected = selector.get_selected_experts()
    best_loss = selector.get_best_loss()

    print("\n   Selection Results:")
    for name, count in selected:
        print(f"     - {name}: {count}")

    print(f"   Best Validation Log Loss: {best_loss:.6f}")

    assert len(selected) > 0, "No experts were selected."
    assert best_loss < 5.0, "Loss seems unusually high (random guess is ~4.6)."

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
