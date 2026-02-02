import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from the provided library
from library.config import RANDOM_SEED, METADATA_DIR, FLOAT_PRECISION
from library.feature_engineering import process_single_image
from library.data_loader import get_combined_dataset
from library.preprocessing import StereoscopicPreprocessor
from library.model_factory import get_expert_library
from library.ensemble_selector import optimize_ensemble, calculate_log_loss


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Starting Library Usage Demo...")
    set_seed(RANDOM_SEED)

    # =========================================================================
    # 1. Demonstrate Feature Engineering
    # =========================================================================
    print("\n[1] Verifying Feature Engineering...")

    # Load train metadata to find a valid image path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df_train = pd.read_csv(train_meta_path)
        # Pick the first image
        sample_img_path = df_train.iloc[0]["image_path"]

        # Extract features
        macro_features = process_single_image(sample_img_path)

        # Validation
        print(f"   Processed image: {sample_img_path}")
        print(f"   Extracted feature shape: {macro_features.shape}")

        # Expecting 11 features (7 Hu Moments + 4 Geometric)
        assert macro_features.shape == (11,), "Macro features should have 11 elements."
        assert macro_features.dtype == FLOAT_PRECISION, "Feature dtype mismatch."
    else:
        print("   Skipping image check (metadata not found).")

    # =========================================================================
    # 2. Demonstrate Data Loading
    # =========================================================================
    print("\n[2] Loading Combined Datasets...")

    # Load Training Data
    # This combines Global features (192) and Macro features (11) -> Total 203
    X_train, y_train, ids_train = get_combined_dataset("train", load_cached_data=True)
    print(f"   Training Data: X={X_train.shape}, y={y_train.shape}")

    # Load Validation Data
    X_val, y_val, ids_val = get_combined_dataset("val", load_cached_data=True)
    print(f"   Validation Data: X={X_val.shape}, y={y_val.shape}")

    # Assertions
    assert X_train.shape[1] == 203, "Combined dataset must have 203 columns."
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels."
    assert not np.isnan(X_train).any(), "Training data contains NaNs."

    # =========================================================================
    # 3. Demonstrate Stereoscopic Preprocessing
    # =========================================================================
    print("\n[3] Running Stereoscopic Preprocessing...")

    # Instantiate the preprocessor
    preprocessor = StereoscopicPreprocessor()

    # Fit on Training Data ONLY
    preprocessor.fit(X_train)
    print("   Preprocessor fitted on training data.")

    # Transform Data for different views
    # We need to prepare data for the specific experts in the library

    data_views = {}
    views_to_generate = ["global_parametric", "global_rank", "macro"]

    for split_name, X_raw in [("train", X_train), ("val", X_val)]:
        data_views[split_name] = {}
        for view in views_to_generate:
            X_trans = preprocessor.transform(X_raw, view=view)
            data_views[split_name][view] = X_trans

            # Validation
            if view == "macro":
                assert (
                    X_trans.shape[1] == 11
                ), f"Macro view should have 11 features, got {X_trans.shape[1]}"
            else:
                assert (
                    X_trans.shape[1] == 192
                ), f"{view} should have 192 features, got {X_trans.shape[1]}"

            print(f"   Generated '{view}' view for {split_name}: {X_trans.shape}")

    # =========================================================================
    # 4. Demonstrate Model Training (Expert Library)
    # =========================================================================
    print("\n[4] Training Expert Models...")

    experts = get_expert_library()
    print(f"   Loaded {len(experts)} experts from the factory.")

    trained_models = {}
    val_predictions = {}

    for expert in experts:
        name = expert["name"]
        view = expert["view"]
        model = expert["model"]

        # Get appropriate training data for this expert's view
        X_train_view = data_views["train"][view]
        X_val_view = data_views["val"][view]

        # Train
        # Note: LDA is fast, so we can train on the full set
        model.fit(X_train_view, y_train)
        trained_models[name] = model

        # Predict on Validation
        preds = model.predict_proba(X_val_view)
        val_predictions[name] = preds

        # Quick check on score for this individual expert
        # We need to encode y_val to indices for the custom log loss function
        # Or use sklearn's log_loss for a quick check
        score = log_loss(y_val, preds)
        print(f"   Expert '{name}' trained. Val Log Loss: {score:.4f}")

    # =========================================================================
    # 5. Demonstrate Ensemble Selection
    # =========================================================================
    print("\n[5] Optimizing Ensemble Weights...")

    # The optimize_ensemble function performs Greedy Forward Selection
    # It requires predictions dictionary and true labels

    ensemble_weights = optimize_ensemble(
        val_predictions,
        y_val,
        max_iter=20,  # Limit iterations for demo speed
        verbose=True,
    )

    print("\n   Selected Ensemble Weights:")
    for expert, weight in ensemble_weights.items():
        print(f"     - {expert}: {weight}")

    # =========================================================================
    # 6. Final Validation Score Calculation
    # =========================================================================
    print("\n[6] Calculating Final Ensemble Score...")

    # Construct weighted average prediction
    final_preds = np.zeros_like(next(iter(val_predictions.values())))
    total_weight = sum(ensemble_weights.values())

    for expert, weight in ensemble_weights.items():
        final_preds += val_predictions[expert] * weight

    final_preds /= total_weight

    # Calculate custom metric
    # Map class names to indices
    classes = trained_models[list(trained_models.keys())[0]].classes_
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    final_loss = calculate_log_loss(y_val_indices, final_preds)

    print(f"   Final Ensemble Validation Log Loss: {final_loss:.6f}")

    # Assertion to ensure we achieved a reasonable score (sanity check)
    # Random guessing for 99 classes is ~ln(99) = 4.6
    assert (
        final_loss < 2.0
    ), f"Ensemble score {final_loss} is too high, something might be wrong."

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
