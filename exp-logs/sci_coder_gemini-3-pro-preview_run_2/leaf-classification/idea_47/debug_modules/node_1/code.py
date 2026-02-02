import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, save_submission, log_loss_score
from library.data_loader import get_class_names, load_dataset
from library.expert_library import get_expert_pool
from library.ensemble import GreedySelector


def main():
    # 1. Initialization and Setup
    print("Step 1: Initializing Environment...")
    Config.initialize()
    set_seed(Config.RANDOM_SEED)

    # 2. Load Class Names
    print("\nStep 2: Loading Class Names...")
    class_names = get_class_names()
    print(f"Number of classes: {len(class_names)}")

    # Verify class names
    assert len(class_names) == 99, f"Expected 99 classes, got {len(class_names)}"

    # 3. Load Datasets
    # load_dataset handles caching and feature extraction (both tabular and image morphometrics)
    print("\nStep 3: Loading Datasets (Train, Val, Test)...")
    train_data = load_dataset("train")
    val_data = load_dataset("val")
    test_data = load_dataset("test")

    # Verify Data Integrity
    print("Verifying loaded data shapes...")
    # Check Global Features (192 dimensions)
    assert train_data["X_global"].shape[1] == 192
    assert val_data["X_global"].shape[1] == 192
    assert test_data["X_global"].shape[1] == 192

    # Check Morphometric Features (11 dimensions)
    assert train_data["X_morph"].shape[1] == 11
    assert val_data["X_morph"].shape[1] == 11
    assert test_data["X_morph"].shape[1] == 11

    # Check Labels
    assert len(train_data["y"]) == train_data["X_global"].shape[0]
    assert len(val_data["y"]) == val_data["X_global"].shape[0]

    print("Data verification passed.")

    # 4. Define Expert Pool
    print("\nStep 4: Generating Expert Pool...")
    experts = get_expert_pool()
    print(f"Generated {len(experts)} candidate experts.")

    # Verify experts exist
    assert len(experts) > 0, "Expert pool is empty."

    # 5. Ensemble Selection (Training Phase)
    print("\nStep 5: Running Greedy Forward Selection...")
    selector = GreedySelector(experts)

    # Fit selector on Train/Val split
    # This trains all candidates on Train, evaluates on Val, and selects the best combination
    selector.fit(
        data_train=train_data,
        y_train=train_data["y"],
        data_val=val_data,
        y_val=val_data["y"],
    )

    # Verify selection
    n_selected = len(selector.selected_experts)
    print(f"Selected {n_selected} experts for the final ensemble.")
    assert n_selected > 0, "No experts were selected."
    assert selector.best_score < float(
        "inf"
    ), "Selection failed to produce a valid score."

    # 6. Refitting on Full Data
    print("\nStep 6: Refitting Ensemble on Full Dataset (Train + Val)...")

    # Combine Train and Val data
    X_full_global = np.vstack([train_data["X_global"], val_data["X_global"]])
    X_full_morph = np.vstack([train_data["X_morph"], val_data["X_morph"]])
    y_full = np.concatenate([train_data["y"], val_data["y"]])

    data_full = {"X_global": X_full_global, "X_morph": X_full_morph}

    # Refit selected experts
    selector.refit(data_full, y_full)

    # Verify models are fitted
    assert len(selector.fitted_models) == n_selected, "Mismatch in fitted models count."

    # 7. Prediction on Test Set
    print("\nStep 7: Generating Predictions for Test Set...")
    test_preds = selector.predict(test_data)

    # Verify Predictions
    assert test_preds.shape == (
        test_data["X_global"].shape[0],
        99,
    ), f"Prediction shape mismatch. Expected ({test_data['X_global'].shape[0]}, 99), got {test_preds.shape}"

    # Check probability constraints
    assert np.all(test_preds >= 0) and np.all(
        test_preds <= 1.0 + 1e-9
    ), "Probabilities out of range [0, 1]."

    # 8. Save Submission
    print("\nStep 8: Saving Submission...")
    save_submission(
        ids=test_data["ids"],
        class_names=class_names,
        probs=test_preds,
        filename=Config.SUBMISSION_PATH,
    )

    # Verify file creation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")

    # Optional: Validate logic by checking score on validation set using the refitted model
    # (Note: This is biased because we refitted on val, but ensures the predict function works)
    print("\nValidating predict function on validation set...")
    val_preds_check = selector.predict(val_data)

    # We need to encode y_val to integers for log_loss_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(y_full)  # Fit on full to ensure all classes cover
    y_val_enc = le.transform(val_data["y"])

    score = log_loss_score(y_val_enc, val_preds_check)
    print(f"Sanity Check - Log Loss on Val Set (after refit): {score:.4f}")
    assert score < 5.0, "Sanity check failed: Log loss is unusually high."

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
