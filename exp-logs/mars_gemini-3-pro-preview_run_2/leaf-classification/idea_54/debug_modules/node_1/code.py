import os
import numpy as np
import pandas as pd
import importlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Reload libraries to ensure config changes are picked up in persistent environments
import library.config

importlib.reload(library.config)
import library.data_manager

importlib.reload(library.data_manager)

# Import provided library components
from library.config import TRAIN_CSV, VAL_CSV, SUBMISSION_DIR, FLOAT_PRECISION
from library.utils import set_seed, clipped_log_loss, save_submission
from library.image_features import process_single_image
from library.data_manager import DataManager
from library.pipeline_factory import (
    get_global_pipeline,
    get_physical_pipeline,
    get_interaction_pipeline,
)
from library.ensemble_selector import HillClimbingOptimizer


def run_demonstration():
    # 1. Setup and Reproducibility
    print("1. Setting up environment...")
    set_seed(42)

    # 2. Data Management
    print("\n2. Demonstrating DataManager...")
    # Initialize DataManager (it will use caching mechanism in ./working/idea_54)
    dm = DataManager(load_cached_data=True)

    # Load data
    data = dm.load_data()

    # Verify data structure
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    print(f"   Training samples: {len(y_train)}")
    print(f"   Validation samples: {len(y_val)}")
    print(f"   Test samples: {len(test_ids)}")
    print(f"   Number of classes: {len(classes)}")

    # Assertions to ensure data integrity
    assert isinstance(X_train, dict), "X_train should be a dictionary of views"
    assert "global" in X_train, "Global view missing in X_train"
    assert X_train["global"].shape[0] == len(y_train), "Mismatch in X_train samples"

    # 3. Image Feature Extraction (Low-level verification)
    print("\n3. Demonstrating Image Feature Extraction...")
    # We'll manually process one image to verify the logic in library.image_features
    # Load metadata to get a valid path
    df_train = pd.read_csv(TRAIN_CSV)
    sample_image_path = df_train.iloc[0]["image_path"]

    print(f"   Processing single image: {sample_image_path}")
    features = process_single_image(sample_image_path)

    print(f"   Extracted feature vector shape: {features.shape}")
    # Expecting 7 Hu moments + 4 Geometric features = 11
    assert features.shape == (11,), f"Expected 11 features, got {features.shape[0]}"
    assert features.dtype == FLOAT_PRECISION, "Feature precision mismatch"

    # 4. Pipeline Factory & Preprocessing
    print("\n4. Demonstrating Pipeline Factory...")

    # A. Global Pipeline (Marginal Topology)
    print("   A. Fitting Global Pipeline (Marginal)...")
    global_pipe = get_global_pipeline(topology="marginal")
    X_train_global_trans = global_pipe.fit_transform(X_train["global"])
    X_val_global_trans = global_pipe.transform(X_val["global"])

    assert (
        X_train_global_trans.shape == X_train["global"].shape
    ), "Pipeline changed shape unexpectedly"

    # B. Interaction Pipeline (Discriminative Bottleneck)
    # This pipeline uses LDA internally, so it requires labels (y)
    print("   B. Fitting Interaction Pipeline (Supervised)...")
    interaction_pipe = get_interaction_pipeline()

    # Use 'margin_texture' view if available (defined in config INTERACTION_PAIRS)
    if "margin_texture" in X_train:
        X_train_int = X_train["margin_texture"]
        # Fit with y_train because of LDA step
        X_train_int_trans = interaction_pipe.fit_transform(X_train_int, y_train)
        print(f"      Input shape: {X_train_int.shape}")
        print(f"      Transformed shape: {X_train_int_trans.shape}")
    else:
        print(
            "      'margin_texture' view not found, skipping specific interaction check."
        )

    # 5. Model Training (Simulation)
    print("\n5. Training Base Models (LDA)...")

    # Model 1: LDA on Global Features
    lda_global = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_global.fit(X_train_global_trans, y_train)

    probs_global = lda_global.predict_proba(X_val_global_trans)
    acc_global = accuracy_score(y_val, lda_global.predict(X_val_global_trans))
    loss_global = clipped_log_loss(y_val, probs_global)
    print(
        f"   Model 1 (Global) - Val Acc: {acc_global:.4f}, Log Loss: {loss_global:.4f}"
    )

    # Model 2: LDA on Morphometric Features (Physical Pipeline)
    # First apply physical pipeline
    phys_pipe = get_physical_pipeline()
    X_train_morph = phys_pipe.fit_transform(X_train["morphometric"])
    X_val_morph = phys_pipe.transform(X_val["morphometric"])

    lda_morph = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_morph.fit(X_train_morph, y_train)

    probs_morph = lda_morph.predict_proba(X_val_morph)
    acc_morph = accuracy_score(y_val, lda_morph.predict(X_val_morph))
    loss_morph = clipped_log_loss(y_val, probs_morph)
    print(f"   Model 2 (Morph)  - Val Acc: {acc_morph:.4f}, Log Loss: {loss_morph:.4f}")

    # 6. Ensemble Selection
    print("\n6. Demonstrating Ensemble Selection (Hill Climbing)...")

    predictions_dict = {"global_expert": probs_global, "morph_expert": probs_morph}

    optimizer = HillClimbingOptimizer(n_iterations=10, verbose=True)
    selected_experts = optimizer.fit(predictions_dict, y_val)

    print(f"   Selected Experts: {selected_experts}")
    weights = optimizer.get_weights()
    print(f"   Ensemble Weights: {weights}")

    # Generate ensemble predictions on validation set
    ensemble_probs_val = optimizer.predict(predictions_dict)
    ensemble_loss = clipped_log_loss(y_val, ensemble_probs_val)
    print(f"   Ensemble Val Log Loss: {ensemble_loss:.4f}")

    # Verify improvement (or equality if one model is much better)
    assert (
        ensemble_loss <= min(loss_global, loss_morph) + 1e-9
    ), "Ensemble should be at least as good as the single best model on validation set"

    # 7. Generating Submission
    print("\n7. Generating Submission...")

    # Preprocess Test Data
    X_test_global_trans = global_pipe.transform(X_test["global"])
    X_test_morph_trans = phys_pipe.transform(X_test["morphometric"])

    # Predict
    test_probs_global = lda_global.predict_proba(X_test_global_trans)
    test_probs_morph = lda_morph.predict_proba(X_test_morph_trans)

    test_pred_dict = {
        "global_expert": test_probs_global,
        "morph_expert": test_probs_morph,
    }

    # Ensemble Predict
    final_test_probs = optimizer.predict(test_pred_dict)

    # Save
    output_path = os.path.join(SUBMISSION_DIR, "demonstration_submission.csv")
    save_submission(test_ids, final_test_probs, classes, output_path=output_path)

    # Verify file creation
    assert os.path.exists(output_path), "Submission file was not created"

    # Verify submission content format
    df_sub = pd.read_csv(output_path)
    assert df_sub.shape == (
        len(test_ids),
        len(classes) + 1,
    ), "Submission shape mismatch"
    assert "id" in df_sub.columns, "id column missing"
    assert list(df_sub.columns[1:]) == list(classes), "Class columns mismatch"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
