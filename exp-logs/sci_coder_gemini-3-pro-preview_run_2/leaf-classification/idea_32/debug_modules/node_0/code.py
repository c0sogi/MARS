import os
import numpy as np
import pandas as pd
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Import provided library modules
from library.utils import set_seed, clipped_log_loss
from library.feature_engineering import load_and_process_data
from library.expert_library import SklearnExpert, BaggedExpert, TaxonomicExpert
from library.ensemble_selection import GreedySelector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting demonstration of library components...")

    # 1. Setup
    set_seed(42)
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    # 2. Data Loading and Feature Engineering
    print("\n[Step 1] Loading and Processing Data...")
    # We force load_cached_data=False to demonstrate the feature extraction logic
    # In a real run with limited time, one might set this to True if cache exists.
    data = load_and_process_data(load_cached_data=False)

    # Unpack data
    X_train = data["X_train_global"]
    y_train = data["y_train"]
    y_train_genus = data["y_train_genus"]

    X_val = data["X_val_global"]
    y_val = data["y_val"]

    X_test = data["X_test_global"]
    test_ids = data["test_ids"]

    species_classes = data["classes"]
    genus_classes = data["genus_classes"]

    # Validation assertions
    assert X_train.shape[0] == y_train.shape[0], "Train features and labels mismatch"
    assert X_train.dtype == np.float64, "Features should be float64"
    print(
        f"Data loaded successfully. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
    )
    print(
        f"Number of Species: {len(species_classes)}, Number of Genera: {len(genus_classes)}"
    )

    # 3. Expert Models
    print("\n[Step 2] Instantiating and Training Experts...")

    # A. SklearnExpert
    # Using LDA as it is fast and generally effective for this type of data
    print("  -> Training SklearnExpert (LDA)...")
    base_lda = LinearDiscriminantAnalysis()
    expert_sklearn = SklearnExpert(base_lda)
    expert_sklearn.fit(X_train, y_train)

    preds_sklearn_val = expert_sklearn.predict_proba(X_val)
    preds_sklearn_test = expert_sklearn.predict_proba(X_test)

    assert preds_sklearn_val.shape == (len(y_val), len(species_classes))
    assert preds_sklearn_val.dtype == np.float64

    # B. BaggedExpert
    # Bagging LDA to improve stability
    print("  -> Training BaggedExpert (Bagged LDA)...")
    expert_bagged = BaggedExpert(
        base_estimator=LinearDiscriminantAnalysis(),
        n_estimators=5,  # Small number for speed in demo
        random_state=42,
    )
    expert_bagged.fit(X_train, y_train)
    preds_bagged_val = expert_bagged.predict_proba(X_val)
    preds_bagged_test = expert_bagged.predict_proba(X_test)

    # C. TaxonomicExpert
    # Hierarchical model: Predict Genus -> Distribute to Species
    print("  -> Training TaxonomicExpert (Genus Level)...")
    # Note: We must pass y_train_genus here, not y_train (species)
    expert_taxo = TaxonomicExpert(
        estimator=LogisticRegression(
            C=1.0, solver="liblinear", multi_class="auto", max_iter=100
        ),
        species_classes=species_classes,
        genus_classes=genus_classes,
    )
    expert_taxo.fit(X_train, y_train_genus)
    preds_taxo_val = expert_taxo.predict_proba(X_val)
    preds_taxo_test = expert_taxo.predict_proba(X_test)

    # Validate Taxonomic Logic:
    # Sum of species probs for a genus should equal the genus prob.
    # We check one sample and one genus mapping.
    sample_idx = 0
    # Find a genus with multiple species
    for g_idx, s_indices in expert_taxo.genus_to_species_map.items():
        if len(s_indices) > 1:
            # Get internal genus prob (we can't access it directly easily without modifying class,
            # but we know the logic: p_species = p_genus / n_children)
            # So sum(p_species) should equal p_genus.
            # Let's verify that p_species are uniform within the genus for a single sample
            probs_in_genus = preds_taxo_val[sample_idx, s_indices]
            if not np.allclose(probs_in_genus, probs_in_genus[0]):
                raise AssertionError(
                    "TaxonomicExpert did not distribute probabilities uniformly among children."
                )
            break
    print("  -> Experts trained and validated.")

    # 4. Ensemble Selection
    print("\n[Step 3] Running Ensemble Selection...")

    predictions_dict = {
        "LDA_Standard": preds_sklearn_val,
        "LDA_Bagged": preds_bagged_val,
        "Hierarchical_LogReg": preds_taxo_val,
    }

    # Instantiate Selector
    selector = GreedySelector(n_iterations=10, tolerance=1e-5, verbose=True)

    # Fit Selector on Validation Data
    selector.fit(predictions_dict, y_val)

    best_weights = selector.get_best_weights()
    print(f"  -> Selected Ensemble Weights: {best_weights}")

    # 5. Metric Verification
    print("\n[Step 4] Verifying Metric (Clipped Log Loss)...")
    # Create a dummy case
    y_true_dummy = np.array([0, 1])
    # Preds: perfectly wrong (should be clipped) and perfectly right
    y_pred_dummy = np.array([[0.0, 1.0], [0.0, 1.0]])  # Wrong  # Right

    # Calculate loss
    loss = clipped_log_loss(y_true_dummy, y_pred_dummy)

    # Expected:
    # Row 0: clipped to [1e-15, 1-1e-15]. True class 0 has prob 1e-15. Log loss term ~ -log(1e-15) ≈ 34.5
    # Row 1: True class 1 has prob 1-1e-15. Log loss term ~ -log(1) = 0
    # Mean ≈ 17.26
    print(f"  -> Calculated Dummy Loss: {loss:.4f}")
    assert loss > 0, "Loss should be positive"
    assert loss < 40, "Loss should be around 17.2 for this dummy case"

    # 6. Generate Submission
    print("\n[Step 5] Generating Submission for Test Set...")

    # Combine test predictions based on weights
    final_test_preds = np.zeros_like(preds_sklearn_test)
    total_weight = sum(best_weights.values())

    test_candidates = {
        "LDA_Standard": preds_sklearn_test,
        "LDA_Bagged": preds_bagged_test,
        "Hierarchical_LogReg": preds_taxo_test,
    }

    for name, weight in best_weights.items():
        final_test_preds += weight * test_candidates[name]

    final_test_preds /= total_weight

    # Normalize rows (just to be safe, though weighted avg of normalized rows is normalized)
    row_sums = final_test_preds.sum(axis=1)
    final_test_preds = final_test_preds / row_sums[:, np.newaxis]

    # Create DataFrame
    submission_df = pd.DataFrame(final_test_preds, columns=species_classes)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_path = os.path.join(working_dir, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"  -> Submission saved to {submission_path}")
    print(f"  -> Submission shape: {submission_df.shape}")

    # Final check on submission format
    assert submission_df.shape[0] == 99, "Submission should have 99 rows"
    assert (
        submission_df.shape[1] == 100
    ), "Submission should have 100 columns (id + 99 species)"
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"

    print("\nDemonstration complete successfully.")


if __name__ == "__main__":
    main()
