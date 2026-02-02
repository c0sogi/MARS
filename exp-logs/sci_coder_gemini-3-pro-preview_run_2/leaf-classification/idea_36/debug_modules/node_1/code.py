import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    ESTIMATOR_CONFIGS,
    BASIS_CONFIGS,
    VIEW_CONFIGS,
    RANDOM_SEED,
)
from library.utils import set_seed, clipped_log_loss, save_submission
from library.features import DataLoader
from library.preprocessing import GaussianBasisFactory
from library.models import get_expert
from library.ensemble import GreedyEnsembleSelector


def run_demo():
    print("Initializing Demo...")
    set_seed(RANDOM_SEED)

    # =========================================================================
    # 1. DATA LOADING & FEATURE EXTRACTION
    # =========================================================================
    print("\n[1] Loading Data & Extracting Features...")
    loader = DataLoader()

    # Load Train and Validation splits
    # This automatically extracts 'macro' features from images if not cached
    ids_train, y_train_raw, views_train = loader.load_split(
        "train", TRAIN_METADATA_PATH, load_cached_data=False
    )
    ids_val, y_val_raw, views_val = loader.load_split(
        "val", VAL_METADATA_PATH, load_cached_data=False
    )

    # Verify Data Integrity
    print(f"    Train Samples: {len(ids_train)}")
    print(f"    Val Samples:   {len(ids_val)}")

    assert len(ids_train) == len(y_train_raw)
    assert "global" in views_train and "macro" in views_train
    assert views_train["global"].shape[0] == len(ids_train)

    # Encode Targets (Strings -> Integers) for internal processing
    # Note: sklearn models usually handle strings, but having consistent encoding is safer
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_val_raw]))
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    print(f"    Classes: {len(classes)}")

    # =========================================================================
    # 2. PREPROCESSING (GAUSSIAN BASIS TRANSFORMATION)
    # =========================================================================
    print("\n[2] Preprocessing (Gaussian Basis Transformation)...")
    factory = GaussianBasisFactory()

    # For demonstration speed, we only use a subset of views/bases if needed,
    # but the dataset is small enough to run full config.
    # We pass views_val as 'test' argument just to satisfy the signature for this demo.
    processed_train, processed_val, _ = factory.process(
        views_train, views_val, views_val, load_cached_data=False
    )

    # Verify Transformation Structure
    # Expect: processed_train[basis_name][view_name] -> np.array
    basis_names = list(BASIS_CONFIGS.keys())
    view_names = list(VIEW_CONFIGS.keys())

    print(f"    Bases: {basis_names}")
    print(f"    Views: {view_names}")

    assert basis_names[0] in processed_train
    assert view_names[0] in processed_train[basis_names[0]]

    # Check shape of a transformed array
    sample_X = processed_train["parametric"]["global"]
    assert sample_X.shape[0] == len(ids_train)
    assert (
        sample_X.shape[1] == views_train["global"].shape[1]
    )  # PowerTransformer keeps dims

    # =========================================================================
    # 3. MODEL TRAINING (EXPERTS)
    # =========================================================================
    print("\n[3] Training Experts...")

    # We will generate predictions for the validation set from multiple experts.
    # To keep the demo fast, we pick specific combinations of (Model, Basis, View).

    expert_predictions = {}

    # Define a few specific scenarios to demonstrate diversity
    scenarios = [
        # Scenario A: LDA on Global Features (Parametric Basis)
        {"model_idx": 0, "basis": "parametric", "view": "global"},  # lda_shrink_0.001
        # Scenario B: LDA on Macro Features (Quantile Coarse)
        {"model_idx": 2, "basis": "quantile_coarse", "view": "macro"},  # lda_oas
        # Scenario C: LogReg on Global Features (Quantile Fine)
        {"model_idx": 3, "basis": "quantile_fine", "view": "global"},  # logreg_cv
    ]

    for i, scen in enumerate(scenarios):
        # Get Config
        cfg = ESTIMATOR_CONFIGS[scen["model_idx"]].copy()

        # Optimize LogReg for speed in demo
        if cfg["type"] == "logreg_cv":
            cfg["params"]["cv"] = 2
            cfg["params"]["max_iter"] = 100
            cfg["params"]["Cs"] = 5

        expert_name = f"{cfg['name']}_{scen['basis']}_{scen['view']}"
        print(f"    Training Expert {i+1}/{len(scenarios)}: {expert_name}")

        # Get Data
        X_tr = processed_train[scen["basis"]][scen["view"]]
        X_val = processed_val[scen["basis"]][scen["view"]]

        # Instantiate & Fit
        expert = get_expert(cfg)
        expert.fit(
            X_tr, y_train_raw
        )  # Passing raw string labels to let sklearn handle classes

        # Predict
        probs = expert.predict_proba(X_val)
        expert_predictions[expert_name] = probs

        # Verify Probabilities
        assert probs.shape == (len(ids_val), len(classes))
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Score individual expert
        score = clipped_log_loss(y_val_raw, probs)
        print(f"      -> Log Loss: {score:.5f}")

    # =========================================================================
    # 4. ENSEMBLE SELECTION
    # =========================================================================
    print("\n[4] Ensemble Selection (Greedy Forward)...")

    selector = GreedyEnsembleSelector()

    # Fit selector on validation predictions
    # Note: y_val_raw (strings) works because clipped_log_loss handles it via sklearn
    selector.fit(expert_predictions, y_val_raw)

    # Get Ensemble Weights
    weights = selector.get_weights()
    print("    Selected Weights:")
    for name, w in weights.items():
        print(f"      {name}: {w:.4f}")

    # Generate Ensemble Predictions
    ensemble_preds = selector.predict(expert_predictions)
    final_score = clipped_log_loss(y_val_raw, ensemble_preds)
    print(f"    Final Ensemble Log Loss: {final_score:.5f}")

    # Verify improvement (or at least non-degradation vs best single)
    best_single_score = min(
        [clipped_log_loss(y_val_raw, p) for p in expert_predictions.values()]
    )
    assert (
        final_score <= best_single_score + 1e-9
    ), "Ensemble should not be worse than best single expert"

    # =========================================================================
    # 5. SUBMISSION GENERATION
    # =========================================================================
    print("\n[5] Generating Submission...")

    # For demo purposes, we use the validation IDs and predictions as if they were test data
    # In a real run, we would load test data, process it, predict, and ensemble.

    output_path = "./working/demo_submission.csv"
    save_submission(ids_val, classes, ensemble_preds, output_path)

    print(f"    Submission saved to {output_path}")

    # Verify File
    df_sub = pd.read_csv(output_path)
    assert df_sub.shape == (len(ids_val), len(classes) + 1)  # +1 for id column
    assert "id" in df_sub.columns
    assert list(df_sub.columns[1:]) == list(classes)

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
