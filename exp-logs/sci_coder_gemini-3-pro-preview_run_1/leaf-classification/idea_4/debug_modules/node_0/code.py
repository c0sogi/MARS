import os
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.data_loader import load_data
from library.models import GlobalLDA, HierarchicalLDA, TaxonomyEnsemble
from library.evaluation import calculate_log_loss, create_submission


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Initialize Configuration
    print("\n[1] Initializing Configuration...")
    config = Config(debug=True)

    # Verify directory creation
    assert os.path.exists(config.WORKING_DIR), "Working directory was not created."
    assert os.path.exists(
        config.SUBMISSION_DIR
    ), "Submission directory was not created."

    # Verify utility functions
    feature_cols = config.get_feature_columns()
    assert (
        len(feature_cols) == 192
    ), f"Expected 192 feature columns, got {len(feature_cols)}"

    genus = config.get_genus_from_species("Acer_Rubrum")
    assert genus == "Acer", f"Genus extraction failed. Expected 'Acer', got '{genus}'"
    print("Configuration and utilities verified.")

    # 2. Data Loading & Preprocessing
    print("\n[2] Loading and Preprocessing Data (Debug Mode)...")
    # We force load_cached_data=False to demonstrate the processing pipeline
    (
        X_train,
        y_train,
        genus_train,
        X_val,
        y_val,
        genus_val,
        X_test,
        test_ids,
        species_encoder,
        genus_encoder,
    ) = load_data(debug=True, load_cached_data=False)

    # Validate shapes
    # Debug mode loads 50 rows for train, val, and test
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")

    assert X_train.shape == (
        50,
        192,
    ), f"Expected (50, 192) for X_train, got {X_train.shape}"
    assert len(y_train) == 50, "y_train length mismatch"
    assert len(genus_train) == 50, "genus_train length mismatch"

    # Validate Preprocessing (StandardScaler should result in mean ~0, std ~1)
    # We check the first feature as a proxy
    mean_val = np.mean(X_train[:, 0])
    std_val = np.std(X_train[:, 0])
    print(f"Feature 0 Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    # Note: With N=50, stats might fluctuate, but shouldn't be raw values (raw are ~0.01)
    assert abs(mean_val) < 1.0, "Feature scaling mean seems off (expected near 0)"

    print("Data loading and preprocessing verified.")

    # 3. Model Demonstration
    print("\n[3] Training and Evaluating Models...")

    # A. Global LDA
    print("Training GlobalLDA...")
    global_model = GlobalLDA()
    global_model.fit(X_train, y_train)

    # Predict on Validation
    global_probs = global_model.predict_proba(X_val)
    assert global_probs.shape == (
        50,
        len(species_encoder.classes_),
    ), f"GlobalLDA output shape mismatch. Got {global_probs.shape}"

    # B. Hierarchical LDA
    print("Training HierarchicalLDA...")
    # Note: With debug data, this will mostly trigger the fallback logic due to low sample counts
    hier_model = HierarchicalLDA()
    hier_model.fit(X_train, y_train, genus_train)

    hier_probs = hier_model.predict_proba(X_val)
    assert hier_probs.shape == (
        50,
        len(species_encoder.classes_),
    ), f"HierarchicalLDA output shape mismatch. Got {hier_probs.shape}"

    # C. Taxonomy Ensemble
    print("Training TaxonomyEnsemble...")
    ensemble = TaxonomyEnsemble()
    ensemble.fit(X_train, y_train, genus_train)

    ensemble_probs = ensemble.predict_proba(X_val)
    assert ensemble_probs.shape == (
        50,
        len(species_encoder.classes_),
    ), f"Ensemble output shape mismatch. Got {ensemble_probs.shape}"

    # Check probability properties
    row_sums = np.sum(ensemble_probs, axis=1)
    # Allow small float error
    assert np.allclose(row_sums, 1.0), "Ensemble probabilities do not sum to 1."

    print("Models trained and inference verified.")

    # 4. Evaluation Metric
    print("\n[4] Calculating Metrics...")
    loss = calculate_log_loss(y_val, ensemble_probs)
    print(f"Validation Log Loss: {loss:.4f}")
    assert loss >= 0, "Log loss cannot be negative."

    # 5. Submission Generation
    print("\n[5] Generating Submission...")
    # Predict on Test set
    test_probs = ensemble.predict_proba(X_test)

    # Create submission file
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    create_submission(
        test_ids, test_probs, species_encoder.classes_, output_path=submission_path
    )

    # Verify file content
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created at {submission_path}")
    print(f"Submission shape: {df_sub.shape}")

    # Check columns: id + 99 species
    expected_cols = 1 + len(species_encoder.classes_)
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Submission column count mismatch. Expected {expected_cols}, got {df_sub.shape[1]}"

    # Check ID column
    assert (
        config.ID_COL in df_sub.columns
    ), f"'{config.ID_COL}' column missing from submission."

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
