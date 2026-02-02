import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.utils import set_seed, save_submission, calculate_log_loss
from library.config import WORKING_DIR, RANDOM_SEED, FLOAT_PRECISION, TOPOLOGIES
from library.features import extract_morphometrics
from library.data_factory import DataFactory
from library.transformations import MarginalTopology, SpectralTopology, RankTopology
from library.model_library import LDAExpert
from library.ensemble_selection import ExpertLibrary, GreedySelector


def demo_feature_extraction():
    print("\n=== Demo 1: Feature Extraction ===")
    # Test on a single image if available
    # We use a path from the metadata if possible, or construct a likely one
    # metadata/train.csv is guaranteed to exist
    train_meta = pd.read_csv("./metadata/train.csv")
    sample_rel_path = train_meta.iloc[0]["image_path"]

    print(f"Extracting features for: {sample_rel_path}")
    features = extract_morphometrics(sample_rel_path)

    print(f"Extracted feature vector shape: {features.shape}")
    print(f"Feature vector sample: {features[:5]}")

    # Validation
    assert features.shape == (10,), "Morphometric feature vector must have 10 elements"
    assert features.dtype == FLOAT_PRECISION, "Feature vector must be float64"
    print("Feature extraction verification passed.")


def demo_data_factory():
    print("\n=== Demo 2: Data Factory & Caching ===")
    factory = DataFactory(load_cached_data=True)

    # 1. Get Classes
    classes = factory.get_classes()
    print(f"Number of classes: {len(classes)}")
    assert len(classes) == 99, "Expected 99 plant species"

    # 2. Get Global View (Pre-computed features only)
    print("Loading Train Data (Global View)...")
    X_global, y_train = factory.get_data("train", "global")
    print(f"X_global shape: {X_global.shape}")
    assert X_global.shape[1] == 192, "Global view should have 192 features"

    # 3. Get Combined View (Global + Morphometrics)
    # This triggers process_image_batch internally if not cached
    print("Loading Train Data (Combined View)...")
    X_combined, _ = factory.get_data("train", "combined")
    print(f"X_combined shape: {X_combined.shape}")
    assert (
        X_combined.shape[1] == 202
    ), "Combined view should have 192 + 10 = 202 features"

    return factory


def demo_transformations(X_sample, y_sample):
    print("\n=== Demo 3: Transformation Topologies ===")

    # Test Spectral Topology
    print("Testing SpectralTopology...")
    topo = SpectralTopology()

    # Fit and Transform
    topo.fit(X_sample, y_sample)
    X_trans = topo.transform(X_sample)

    print(f"Original shape: {X_sample.shape}")
    print(f"Transformed shape: {X_trans.shape}")

    # Validation
    assert X_trans.shape == X_sample.shape, "Topology should not change sample count"
    assert not np.isnan(X_trans).any(), "Transformation produced NaNs"
    print("Transformation verification passed.")

    return X_trans


def demo_model_training(X_train, y_train, X_val, y_val):
    print("\n=== Demo 4: LDA Expert Training ===")

    # Initialize LDA with OAS shrinkage
    lda = LDAExpert(shrinkage="oas")

    # Fit
    print("Fitting LDA (OAS)...")
    lda.fit(X_train, y_train)
    print(f"Estimated Shrinkage: {lda.shrinkage_val_}")

    # Predict
    probs = lda.predict_proba(X_val)
    print(f"Prediction shape: {probs.shape}")

    # Validation
    score = calculate_log_loss(y_val, probs)
    print(f"Validation Log Loss: {score:.4f}")
    assert probs.shape[1] == 99, "Probabilities must cover all 99 classes"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"
    print("Model training verification passed.")


def demo_ensemble_selection(factory):
    print("\n=== Demo 5: Ensemble Selection ===")

    # 1. Generate Library of Experts
    # This iterates through Topologies x Shrinkage x Views
    library = ExpertLibrary(factory)
    preds_dict, y_val = library.generate_val_predictions(load_cached_data=True)

    print(f"Generated predictions for {len(preds_dict)} experts.")

    # 2. Run Greedy Selection
    selector = GreedySelector(max_iterations=10, tolerance=1e-4)
    selector.fit(preds_dict, y_val)

    ensemble_config = selector.get_best_ensemble()
    print("\nSelected Ensemble Configuration:")
    for key, count in ensemble_config:
        print(f"  - Expert: {key}, Weight: {count}")

    return ensemble_config


def demo_final_submission(factory, ensemble_config):
    print("\n=== Demo 6: Final Inference & Submission ===")

    # We need to retrain the selected experts on the full training set (train + val)
    # and predict on the test set.

    # 1. Load Full Training Data and Test Data
    # We load both views to be ready
    X_full_global, y_full = factory.get_data("train_full", "global")
    X_full_combined, _ = factory.get_data("train_full", "combined")

    X_test_global, test_ids = factory.get_data("test", "global")
    X_test_combined, _ = factory.get_data("test", "combined")

    # Map for data access
    data_map = {
        "global": (X_full_global, X_test_global),
        "combined": (X_full_combined, X_test_combined),
    }

    # Map for topology classes
    topo_map = {
        "marginal": MarginalTopology,
        "spectral": SpectralTopology,
        "rank": RankTopology,
    }

    n_test_samples = len(test_ids)
    n_classes = len(factory.get_classes())

    # Accumulator for weighted probabilities
    final_probs_sum = np.zeros((n_test_samples, n_classes), dtype=FLOAT_PRECISION)
    total_weight = 0

    print("Retraining ensemble experts on full data...")

    for key, weight in ensemble_config:
        # Parse key: topology___shrinkage___view
        topo_name, shrinkage_str, view_name = key.split("___")

        # Handle shrinkage type conversion
        try:
            shrinkage = float(shrinkage_str)
        except ValueError:
            shrinkage = shrinkage_str  # 'oas' or 'ledoit_wolf'

        print(
            f"  Processing: {topo_name} | {shrinkage} | {view_name} (Weight: {weight})"
        )

        # Get data
        X_train_curr, X_test_curr = data_map[view_name]

        # 1. Transform
        transformer = topo_map[topo_name]()
        transformer.fit(X_train_curr, y_full)
        X_train_trans = transformer.transform(X_train_curr)
        X_test_trans = transformer.transform(X_test_curr)

        # 2. Train Model
        model = LDAExpert(shrinkage=shrinkage)
        model.fit(X_train_trans, y_full)

        # 3. Predict
        probs = model.predict_proba(X_test_trans)

        # 4. Accumulate
        final_probs_sum += probs * weight
        total_weight += weight

    # Average
    final_probs = final_probs_sum / total_weight

    # Save Submission
    output_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    save_submission(test_ids, factory.get_classes(), final_probs, output_path)
    print(f"Submission saved to {output_path}")

    # Verify file exists
    assert os.path.exists(output_path)

    # Quick check on the file content
    df_sub = pd.read_csv(output_path)
    print(f"Submission shape: {df_sub.shape}")
    assert df_sub.shape == (99, 100)  # 99 test samples, 1 id col + 99 class cols
    print("Submission verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(RANDOM_SEED)

    try:
        # 1. Feature Extraction
        demo_feature_extraction()

        # 2. Data Loading
        factory = demo_data_factory()

        # 3. Transformations
        # Use a small subset for demonstration
        X_train_global, y_train = factory.get_data("train", "global")
        X_trans = demo_transformations(X_train_global, y_train)

        # 4. Model Training
        # Split the training data further just for this isolated demo function
        # (In real flow, we use train/val splits provided by factory)
        X_val_global, y_val = factory.get_data("val", "global")

        # We need to transform validation data using the same transformer for a valid test
        # but for the demo_model_training function we just need compatible shapes.
        # Let's do it properly:
        topo = SpectralTopology()
        topo.fit(X_train_global, y_train)
        X_train_t = topo.transform(X_train_global)
        X_val_t = topo.transform(X_val_global)

        demo_model_training(X_train_t, y_train, X_val_t, y_val)

        # 5. Ensemble Selection
        ensemble_config = demo_ensemble_selection(factory)

        # 6. Final Submission
        demo_final_submission(factory, ensemble_config)

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
