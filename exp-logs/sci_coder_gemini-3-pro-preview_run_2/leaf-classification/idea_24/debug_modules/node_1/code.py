import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.data_handler import DataHandler
from library.model_factory import get_expert_library, create_expert_pipeline
from library.ensemble_strategy import GreedyForwardSelector

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting demonstration script...")
    set_seed(Config.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 1. Data Handling
    # -------------------------------------------------------------------------
    print("\n[1] Initializing DataHandler...")
    # We use debug=False because the dataset is small (Total < 1000 images),
    # so full processing is fast and ensures class consistency across splits.
    data_handler = DataHandler(debug=False)

    # Load data splits (forcing re-computation to demonstrate the logic)
    print("Loading and processing data splits...")
    train_data, val_data, test_data = data_handler.get_data_splits(
        load_cached_data=False
    )

    # Verification of Data Shapes
    n_train = len(train_data["ids"])
    n_val = len(val_data["ids"])
    n_test = len(test_data["ids"])

    # Check if classes were correctly extracted
    if "classes" not in train_data:
        raise ValueError("Class labels not found in training data.")
    n_classes = len(train_data["classes"])

    print(
        f"Data loaded: Train={n_train}, Val={n_val}, Test={n_test}, Classes={n_classes}"
    )

    # Assertions to verify data integrity
    assert "X_global" in train_data
    assert "X_morphological" in train_data
    assert (
        train_data["X_global"].shape[1] == 192
    ), "Global features should have 192 columns"
    assert train_data["X_morphological"].shape[1] > 0, "Morphological features missing"
    assert len(train_data["y"]) == n_train, "Label count mismatch"

    # -------------------------------------------------------------------------
    # 2. Model Factory & Training
    # -------------------------------------------------------------------------
    print("\n[2] Configuring Expert Library...")
    all_experts = get_expert_library(debug=False)

    # For demonstration speed, we select one expert of each distinct type/view combination
    # instead of running the full hyperparameter grid.
    # We expect to pick:
    # 1. LDA on Global features
    # 2. LDA_OAS on Global features
    # 3. QDA on Global features
    # 4. LDA on Morphological features

    selected_experts = []
    types_seen = set()

    for exp in all_experts:
        # Create a unique key based on model type and feature view
        key = (exp["type"], exp["view"])
        if key not in types_seen:
            selected_experts.append(exp)
            types_seen.add(key)

    print(f"Selected {len(selected_experts)} representative experts for demonstration:")
    for exp in selected_experts:
        print(f" - {exp['name']} ({exp['type']} on {exp['view']})")

    val_preds_dict = {}
    test_preds_dict = {}

    print("Training experts...")
    for exp_config in selected_experts:
        name = exp_config["name"]
        view = exp_config["view"]

        # Select appropriate feature view based on expert config
        if view == "global":
            X_train = train_data["X_global"]
            X_val = val_data["X_global"]
            X_test = test_data["X_global"]
        elif view == "morphological":
            X_train = train_data["X_morphological"]
            X_val = val_data["X_morphological"]
            X_test = test_data["X_morphological"]
        else:
            print(f"Warning: Unknown view '{view}' for expert {name}. Skipping.")
            continue

        y_train = train_data["y"]

        # Create Pipeline from factory
        pipeline = create_expert_pipeline(exp_config)

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predict probabilities
        val_p = pipeline.predict_proba(X_val)
        test_p = pipeline.predict_proba(X_test)

        # Verify prediction shape
        assert val_p.shape == (n_val, n_classes), f"Val shape mismatch for {name}"
        assert test_p.shape == (n_test, n_classes), f"Test shape mismatch for {name}"

        val_preds_dict[name] = val_p
        test_preds_dict[name] = test_p

        # Quick check on log loss for this expert
        score = log_loss(val_data["y"], val_p)
        print(f"   {name} Val Log Loss: {score:.4f}")

    # -------------------------------------------------------------------------
    # 3. Ensemble Strategy
    # -------------------------------------------------------------------------
    print("\n[3] Running Greedy Forward Selection...")
    # Initialize selector with limited iterations for demo speed
    selector = GreedyForwardSelector(max_iter=10, verbose=True)

    # Fit selector on validation predictions and true labels
    selector.fit(val_preds_dict, val_data["y"])

    # Verify selector state
    assert len(selector.weights) > 0, "Selector failed to select any models"

    # Generate final aggregated predictions for the test set
    final_test_preds = selector.predict(test_preds_dict)

    # Verify final predictions integrity
    assert final_test_preds.shape == (n_test, n_classes)
    # Check if probabilities sum to approximately 1
    assert np.allclose(
        final_test_preds.sum(axis=1), 1.0
    ), "Final probabilities do not sum to 1"

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[4] Generating Submission File...")

    # Get class names from DataHandler to ensure correct column order
    class_names = train_data["classes"]

    # Create DataFrame
    submission_df = pd.DataFrame(final_test_preds, columns=class_names)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", test_data["ids"])

    # Ensure output directory exists (defined in Config)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Validate file creation
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Submission file was not created at {save_path}")

    # Validate file format by reading it back
    check_df = pd.read_csv(save_path)
    assert check_df.shape == (n_test, n_classes + 1), "Submission shape mismatch"
    assert "id" in check_df.columns, "ID column missing in submission"
    # Check if columns match expected classes (excluding 'id')
    assert list(check_df.columns[1:]) == list(class_names), "Column names mismatch"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
