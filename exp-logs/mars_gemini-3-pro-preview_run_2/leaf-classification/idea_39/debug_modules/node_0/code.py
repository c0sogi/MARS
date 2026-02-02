import os
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import log_loss

# Import provided library modules
from library import config
from library import features
from library import data
from library import preprocessing
from library import library as model_lib
from library import ensemble


def run_demo():
    print("Starting DSPGL Library Demo...")

    # Ensure reproducibility
    np.random.seed(config.RANDOM_SEED)

    # =========================================================================
    # 1. Demonstrate Feature Extraction (features.py)
    # =========================================================================
    print("\n[1] Testing Feature Extraction...")

    # Load metadata to find a valid image path
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    df_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = df_train_meta.iloc[0]
    sample_img_path = os.path.join(config.INPUT_DIR, sample_row["image_path"])

    print(f"Extracting morphometrics from: {sample_img_path}")
    morph_feats = features.extract_morphometrics(sample_img_path)

    # Verification
    expected_keys = [f"hu_{i}" for i in range(1, 8)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]
    for k in expected_keys:
        if k not in morph_feats:
            raise AssertionError(f"Missing key {k} in extracted features.")

    print("Feature extraction successful. Sample features:")
    print(f"  Aspect Ratio: {morph_feats['aspect_ratio']:.4f}")
    print(f"  Solidity:     {morph_feats['solidity']:.4f}")

    # =========================================================================
    # 2. Demonstrate Data Loading (data.py)
    # =========================================================================
    print("\n[2] Testing Data Loading...")

    # We use load_cached_data=True to speed up if cache exists,
    # but the logic handles computation if missing.
    raw_data = data.get_data(load_cached_data=True)

    # Unpack
    X_train_global, X_train_macro, y_train = raw_data["train"]
    X_val_global, X_val_macro, y_val = raw_data["val"]
    X_test_global, X_test_macro, test_ids = raw_data["test"]
    le = raw_data["label_encoder"]

    print(
        f"Train shapes: Global {X_train_global.shape}, Macro {X_train_macro.shape}, y {y_train.shape}"
    )
    print(
        f"Val shapes:   Global {X_val_global.shape}, Macro {X_val_macro.shape}, y {y_val.shape}"
    )
    print(
        f"Test shapes:  Global {X_test_global.shape}, Macro {X_test_macro.shape}, IDs {test_ids.shape}"
    )

    # Verification
    assert X_train_global.shape[0] == len(y_train), "Train samples mismatch"
    assert X_val_global.shape[0] == len(y_val), "Val samples mismatch"
    assert X_test_global.shape[0] == len(test_ids), "Test samples mismatch"
    assert (
        X_train_global.shape[1] == 192
    ), f"Expected 192 global features, got {X_train_global.shape[1]}"
    # (64 margin + 64 shape + 64 texture = 192)

    # =========================================================================
    # 3. Demonstrate Preprocessing (preprocessing.py)
    # =========================================================================
    print("\n[3] Testing Dual-Stream Preprocessing...")

    transformed_data = preprocessing.get_transformed_data(
        raw_data, load_cached_data=True
    )

    train_views = transformed_data["train"]
    val_views = transformed_data["val"]
    test_views = transformed_data["test"]

    # Verify views exist
    required_views = ["global_a", "global_b", "macro_a", "macro_b"]
    for view in required_views:
        assert view in train_views, f"Missing view {view} in training data"
        assert not np.isnan(train_views[view]).any(), f"NaNs detected in {view}"

    print("Transformation successful. Views generated:")
    for view in required_views:
        print(f"  {view}: {train_views[view].shape}")

    # =========================================================================
    # 4. Demonstrate Expert Training (library.py)
    # =========================================================================
    print("\n[4] Testing LDA Expert Training...")

    expert_configs = model_lib.generate_expert_configs()
    print(f"Generated {len(expert_configs)} expert configurations.")

    # Dictionary to store validation predictions for the ensemble
    val_preds_dict = {}
    test_preds_dict = {}

    # Train all experts
    for i, conf in enumerate(expert_configs):
        name = conf["name"]
        view_name = conf["view"]
        shrinkage = conf["shrinkage"]

        # Instantiate
        expert = model_lib.LDAExpert(shrinkage=shrinkage, view_name=view_name)

        # Get data for specific view
        X_tr = train_views[view_name]
        y_tr = train_views["y"]
        X_v = val_views[view_name]
        X_te = test_views[view_name]

        # Fit
        expert.fit(X_tr, y_tr)

        # Predict
        p_val = expert.predict_proba(X_v)
        p_test = expert.predict_proba(X_te)

        val_preds_dict[name] = p_val
        test_preds_dict[name] = p_test

        # Basic check on first expert
        if i == 0:
            assert p_val.shape == (
                len(y_val),
                len(le.classes_),
            ), "Probability shape mismatch"
            # Check probability sum constraint
            row_sums = p_val.sum(axis=1)
            assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
            print(f"  Trained expert: {name}")

    print("All experts trained and evaluated.")

    # =========================================================================
    # 5. Demonstrate Ensemble Selection (ensemble.py)
    # =========================================================================
    print("\n[5] Testing Greedy Ensemble Selection...")

    selector = ensemble.GreedySelector(max_experts=20, tolerance=1e-5, verbose=True)

    # Fit selector on validation data
    selector.fit(val_preds_dict, y_val)

    selected = selector.selected_experts
    weights = selector.get_weights()

    print(f"Selected {len(selected)} experts.")
    print("Top 5 Weights:")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for name, w in sorted_weights[:5]:
        print(f"  {name}: {w:.4f}")

    assert len(selected) > 0, "No experts selected by GreedySelector"

    # =========================================================================
    # 6. Generate Submission
    # =========================================================================
    print("\n[6] Generating Submission...")

    # Predict on test set using the ensemble
    final_test_probs = selector.predict(test_preds_dict)

    # Verify output shape
    n_test_samples = len(test_ids)
    n_classes = len(le.classes_)
    assert final_test_probs.shape == (n_test_samples, n_classes)

    # Construct DataFrame
    submission_df = pd.DataFrame(final_test_probs, columns=le.classes_)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Final check of the file
    saved_df = pd.read_csv(submission_path)
    print(f"Loaded submission shape: {saved_df.shape}")
    assert saved_df.shape == (99, 100), f"Expected (99, 100), got {saved_df.shape}"
    assert "id" in saved_df.columns
    assert "Acer_Capillipes" in saved_df.columns

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
