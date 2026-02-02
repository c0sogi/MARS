import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Import Library Modules
import library.config as config
from library.data_factory import load_dataset
from library.model_lib import generate_expert_library
from library.ensemble_selector import GreedySelector

# =============================================================================
# OPTIMIZATION & CONFIGURATION
# =============================================================================
# Modify configuration for a fast demonstration run
print("[Demo] Configuring runtime parameters...")

# Limit LDA shrinkage candidates to a single value to reduce the number of experts
config.LDA_SHRINKAGE_CANDIDATES = [0.5]

# Reduce interaction components for Topology D to speed up dimensionality reduction
config.INTERACTION_N_COMPONENTS = 5

# Ensure reproducibility
np.random.seed(config.RANDOM_SEED)

# =============================================================================
# DATA LOADING & VERIFICATION
# =============================================================================
print("[Demo] Loading dataset and extracting features...")

# Load dataset (load_cached_data=False ensures we run the extraction pipeline)
dataset = load_dataset(load_cached_data=False)

# Verify Dataset Structure
print("[Demo] Verifying dataset structure...")
assert "train" in dataset, "Dataset missing 'train' key"
assert "val" in dataset, "Dataset missing 'val' key"
assert "test" in dataset, "Dataset missing 'test' key"
assert "classes" in dataset, "Dataset missing 'classes' key"

# Verify Shapes (Based on metadata counts: Train ~712, Val ~179, Test ~99)
# Global view has 192 features
n_train = dataset["train"]["global"].shape[0]
n_val = dataset["val"]["global"].shape[0]
n_test = dataset["test"]["global"].shape[0]
n_features = dataset["train"]["global"].shape[1]

print(f"  - Train samples: {n_train}")
print(f"  - Val samples:   {n_val}")
print(f"  - Test samples:  {n_test}")
print(f"  - Global features: {n_features}")

assert n_train > 0, "Training set is empty"
assert n_val > 0, "Validation set is empty"
assert n_features == 192, f"Expected 192 global features, got {n_features}"

# Verify Morphometric Features (Topology E input)
# These are extracted from images, so checking them verifies the image pipeline
morph_dim = dataset["train"]["morph"].shape[1]
print(f"  - Morphometric features: {morph_dim}")
assert morph_dim > 0, "Morphometric features were not extracted."

# =============================================================================
# EXPERT TRAINING & PREDICTION
# =============================================================================
print("[Demo] Generating and training experts...")

experts = generate_expert_library()
print(f"  - Generated {len(experts)} experts based on configuration.")

# Dictionaries to store predictions
val_preds = {}
test_preds = {}

# Training Loop
for i, expert in enumerate(experts):
    print(
        f"  - Training Expert {i+1}/{len(experts)}: {expert.name} (View: {expert.view_name})"
    )

    # Select the appropriate view for the expert
    X_train = dataset["train"][expert.view_name]
    y_train = dataset["train"]["y"]
    X_val = dataset["val"][expert.view_name]
    X_test = dataset["test"][expert.view_name]

    # Fit
    expert.fit(X_train, y_train)

    # Predict
    val_p = expert.predict_proba(X_val)
    test_p = expert.predict_proba(X_test)

    # Store
    val_preds[expert.name] = val_p
    test_preds[expert.name] = test_p

    # Basic assertion on probability shape
    assert val_p.shape == (
        n_val,
        len(dataset["classes"]),
    ), "Validation prediction shape mismatch"

# =============================================================================
# ENSEMBLE SELECTION
# =============================================================================
print("[Demo] Running Greedy Forward Selection...")

y_val_true = dataset["val"]["y"]

# Initialize Selector
selector = GreedySelector(n_iterations=10, tolerance=1e-5)

# Fit Selector
selector.fit(val_preds, y_val_true)

# Verify Selection
selected = selector.get_selected_experts()
weights = selector.get_weights()

print(f"  - Selected Experts: {selected}")
print(f"  - Weights: {weights}")

assert len(selected) > 0, "No experts were selected by the greedy selector."
assert np.isclose(sum(weights.values()), 1.0), "Ensemble weights do not sum to 1.0"

# =============================================================================
# SUBMISSION GENERATION
# =============================================================================
print("[Demo] Generating submission...")

# Predict on Test Set using the optimized ensemble
final_test_probs = selector.predict(test_preds)

# Verify output shape and constraints
assert final_test_probs.shape == (
    n_test,
    len(dataset["classes"]),
), "Test prediction shape mismatch"
assert np.all(final_test_probs >= 0) and np.all(
    final_test_probs <= 1
), "Probabilities out of bounds"

# Construct Submission DataFrame
submission_df = pd.DataFrame(final_test_probs, columns=dataset["classes"])
submission_df.insert(0, "id", dataset["test"]["ids"])

# Save
submission_path = config.SUBMISSION_PATH
submission_df.to_csv(submission_path, index=False)

print(f"  - Submission saved to: {submission_path}")
print(f"  - Submission shape: {submission_df.shape}")

# Final check of the file
saved_df = pd.read_csv(submission_path)
assert saved_df.shape == (
    99,
    100,
), f"Expected (99, 100) submission shape, got {saved_df.shape}"

print("[Demo] Completed successfully.")
