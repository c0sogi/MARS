import os
import sys
import numpy as np
import pandas as pd
from library import config, utils, data_loader, pipelines, models, ensemble


def run_pipeline_demo():
    # 1. Setup and Configuration
    print("Initializing Leaf Classification Demo...")
    utils.set_seed(config.RANDOM_STATE)

    # 2. Load Data
    # This handles loading CSVs and extracting morphometric features from images
    print("Loading data...")
    data = data_loader.load_data(load_cached_data=True)

    # Validation of loaded data
    assert "train" in data and "val" in data and "test" in data
    assert len(data["class_names"]) == 99

    # Extract targets
    y_train = data["train"]["y"]
    y_val = data["val"]["y"]

    # Containers for expert predictions
    # Keys: Expert Name, Values: Probability Matrix (N_samples, N_classes)
    val_predictions_dict = {}
    test_predictions_dict = {}

    # 3. Process Views and Train Experts
    # We iterate through the 4 views defined in the CIPGE strategy
    view_codes = ["A", "B", "C", "D"]

    print(f"Training experts for views: {view_codes}")

    for view_code in view_codes:
        # A. Determine Input Feature Type
        ftype = pipelines.ViewFactory.get_feature_type(view_code)

        if ftype == "global":
            X_train = data["train"]["X_global"]
            X_val = data["val"]["X_global"]
            X_test = data["test"]["X_global"]
        elif ftype == "morphometric":
            X_train = data["train"]["X_morph"]
            X_val = data["val"]["X_morph"]
            X_test = data["test"]["X_morph"]
        else:
            raise ValueError(f"Unknown feature type for view {view_code}")

        # B. Construct and Fit Pipeline
        # Pipelines handle scaling, power transforms, PCA, and polynomial expansion
        pipeline = pipelines.ViewFactory.get_pipeline(view_code)

        # Fit pipeline on training data
        X_train_trans = pipeline.fit_transform(X_train, y_train)

        # Transform validation and test data
        X_val_trans = pipeline.transform(X_val)
        X_test_trans = pipeline.transform(X_test)

        # C. Instantiate and Train Experts
        # Get list of LDA models with different shrinkage parameters
        experts = models.get_view_experts(view_code)

        for expert in experts:
            model = expert["model"]
            name = expert["name"]

            # Train the LDA model
            model.fit(X_train_trans, y_train)

            # Generate probabilities
            # Note: LDA predict_proba returns columns sorted by class name,
            # which matches data['class_names']
            val_probs = model.predict_proba(X_val_trans)
            test_probs = model.predict_proba(X_test_trans)

            # Store predictions
            val_predictions_dict[name] = val_probs
            test_predictions_dict[name] = test_probs

    print(f"Total experts trained: {len(val_predictions_dict)}")

    # 4. Ensemble Selection
    # Use Greedy Forward Selection to optimize ensemble weights on validation set
    print("Running Greedy Forward Selection...")

    # We use 20 iterations for the demo to ensure speed,
    # though the library default is 50.
    selector = ensemble.GreedySelector(iterations=20, random_state=config.RANDOM_STATE)

    # Fit selector
    # Note: y_val contains string labels. The metric function handles this
    # provided all classes are present (stratified split ensures this).
    selector.fit(val_predictions_dict, y_val)

    # Validate selection
    if not selector.selected_experts:
        raise RuntimeError("Ensemble selection failed to select any experts.")

    # 5. Generate Final Submission
    print("Generating final predictions...")

    # Aggregate test predictions using learned weights
    final_test_probs = selector.predict(test_predictions_dict)

    # Save to CSV
    output_filename = "submission.csv"
    utils.save_submission(
        ids=data["test"]["ids"],
        predictions=final_test_probs,
        class_names=data["class_names"],
        filename=output_filename,
    )

    # Verify output exists
    output_path = os.path.join(config.SUBMISSION_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"Success! Submission generated at: {output_path}")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_pipeline_demo()
