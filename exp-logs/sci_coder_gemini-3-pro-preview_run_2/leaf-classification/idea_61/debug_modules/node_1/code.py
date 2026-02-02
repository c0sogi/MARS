import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import importlib

# Import from the provided library files
import library.config
import library.custom_transformers
import library.expert_library

# Reload modules to ensure updates are applied (Cite debug_lesson_17)
importlib.reload(library.config)
importlib.reload(library.custom_transformers)
importlib.reload(library.expert_library)

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    WORKING_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
)
from library.utils import set_seed, clipped_log_loss, save_submission
from library.feature_extraction import load_image_features
from library.custom_transformers import FactorizedDiscriminantProjector
from library.expert_library import get_expert_library
from library.ensemble_selection import GreedySelector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Library Usage Demo")
    print("----------------------------------------------------------------")

    # 1. Setup and Reproducibility
    set_seed(42)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n[1] Loading Metadata...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        raise FileNotFoundError(
            "Metadata CSV files not found. Ensure ./metadata exists."
        )

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    print(f"Train Set: {df_train.shape}")
    print(f"Val Set:   {df_val.shape}")

    # Extract Global Features (192 columns) and Targets
    global_cols = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

    X_train_global = df_train[global_cols].values
    X_val_global = df_val[global_cols].values

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    print(f"Classes: {len(classes)}")

    # 3. Feature Extraction Demo (Morphometrics)
    print("\n[2] Demonstrating Image Feature Extraction...")

    # We use a small subset of the validation data to demonstrate the actual image processing
    # without consuming too much time.
    demo_subset_size = 20
    df_subset = df_val.head(demo_subset_size).copy()

    print(f"Processing {demo_subset_size} images from validation set...")
    # dataset_name ensures unique caching
    X_morph_subset = load_image_features(df_subset, dataset_name="demo_val_subset")

    # Verification
    assert X_morph_subset.shape == (
        demo_subset_size,
        11,
    ), f"Expected shape ({demo_subset_size}, 11), got {X_morph_subset.shape}"
    print("Morphometric features extracted successfully.")

    # For the pipeline training step (Group B), we need morphometrics for the full training set.
    # To strictly optimize for speed as requested, we will generate random mock morphometrics
    # for the training set. This allows us to test the pipeline code flow without waiting
    # for full dataset image processing.
    print("Generating mock morphometrics for training set (Speed Optimization)...")
    X_train_morph = np.random.rand(len(df_train), 11)

    # Mock full validation set morphometrics for consistency in the prediction loop
    X_val_morph = np.random.rand(len(df_val), 11)

    # 4. Custom Transformer Demo
    print("\n[3] Demonstrating FactorizedDiscriminantProjector...")

    # Initialize transformer
    # n_components=9, resulting in 3 interaction pairs
    fdp = FactorizedDiscriminantProjector(
        n_components=9, solver="lsqr", shrinkage="auto"
    )

    # Fit on global training data
    fdp.fit(X_train_global, y_train)

    # Transform validation data
    X_transformed = fdp.transform(X_val_global)

    # Calculation of expected features:
    # 3 pairs: (Margin-Texture), (Shape-Texture), (Margin-Shape)
    # Each pair has 9+9=18 inputs.
    # Poly(degree=2, interaction_only=True, include_bias=False) on 18 inputs:
    # Features = 18 (linear) + 18*17/2 (interactions) = 18 + 153 = 171
    # Total = 171 * 3 = 513
    print(f"FDP Transformed Shape: {X_transformed.shape}")
    assert X_transformed.shape == (
        len(df_val),
        513,
    ), f"Expected 513 features, got {X_transformed.shape[1]}"

    # 5. Expert Library & Training
    print("\n[4] Training Selected Experts...")
    library = get_expert_library()

    # Select one expert from each group
    selected_keys = [
        "GroupA_Marginal_LDA_auto",  # Group A
        "GroupB_Physical_Poly_LDA_Auto",  # Group B (uses morphometrics)
        "GroupC_Factorized_LDA_0.1",  # Group C
    ]

    expert_predictions = {}

    for key in selected_keys:
        if key not in library:
            continue

        expert = library[key]
        print(f"Training {key}...")

        # Select Data
        if expert["features"] == "global":
            X_t, X_v = X_train_global, X_val_global
        elif expert["features"] == "morphometrics":
            X_t, X_v = X_train_morph, X_val_morph

        # Fit Pipeline
        expert["pipeline"].fit(X_t, y_train)

        # Predict Probabilities
        preds = expert["pipeline"].predict_proba(X_v)
        expert_predictions[key] = preds

        # Evaluate
        loss = clipped_log_loss(y_val, preds)
        print(f"  -> Log Loss: {loss:.4f}")

    # 6. Ensemble Selection
    print("\n[5] Running Greedy Ensemble Selection...")

    selector = GreedySelector(n_iterations=5, tolerance=1e-5)
    selector.fit(expert_predictions, y_val)

    best_weights = selector.get_best_weights()
    print(f"Selected Weights: {best_weights}")

    # Predict with Ensemble
    ensemble_preds = selector.predict(expert_predictions)
    ensemble_loss = clipped_log_loss(y_val, ensemble_preds)
    print(f"Ensemble Log Loss: {ensemble_loss:.4f}")

    # Validation: Ensemble should be at least as good as the best single expert
    # (or very close, allowing for float precision nuances)
    best_single_loss = min(
        [clipped_log_loss(y_val, p) for p in expert_predictions.values()]
    )
    assert (
        ensemble_loss <= best_single_loss + 1e-9
    ), "Ensemble failed to match or outperform the best single expert."

    # 7. Submission Format
    print("\n[6] Generating Sample Submission...")

    # Create dummy test IDs
    test_ids = np.arange(1, 6)
    # Create dummy probabilities (5 samples, 99 classes)
    dummy_probs = np.random.dirichlet(np.ones(len(classes)), size=5)

    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    save_submission(test_ids, classes, dummy_probs, submission_path)

    assert os.path.exists(submission_path), "Submission file not created."

    # Verify file content
    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (
        5,
        100,
    ), f"Submission shape mismatch. Expected (5, 100), got {sub_df.shape}"
    print(f"Submission saved to {submission_path}")

    print("\n----------------------------------------------------------------")
    print("Demo Completed Successfully")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    run_demo()
