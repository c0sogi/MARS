import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from PIL import Image

# Import from provided library
from library import config
from library import utils
from library import feature_extractor
from library import classifier


def main():
    # 1. Setup
    utils.seed_everything()

    # 2. Feature Extraction
    # This will load from cache if available, or compute on GPU if not.
    # Returns a dict: model_name -> {'train': (emb, lbl, ids), 'val': ..., 'test': ...}
    features_data = {}
    for model_name in config.MODELS.keys():
        print(f"\n--- Processing Features for {model_name} ---")
        features_data[model_name] = feature_extractor.run_feature_extraction(
            model_name, load_cached_data=True
        )

    # 3. Training & Inference (Ensemble)
    # We will collect predictions from all models to perform weighted ensembling
    model_predictions = {}
    val_labels = None
    test_ids = None

    print("\n--- Training Classifiers and Ensembling ---")

    for model_name, datasets in features_data.items():
        # Unpack data
        train_emb, train_lbl, _ = datasets["train"]
        val_emb, val_lbl, _ = datasets["val"]
        test_emb, _, current_test_ids = datasets["test"]

        # Keep track of labels and IDs (should be consistent across models)
        if val_labels is None:
            val_labels = val_lbl
        if test_ids is None:
            test_ids = current_test_ids

        # Load or Train Classifier
        clf = classifier.load_model(model_name)

        if clf is None:
            # Train new model
            clf, score, best_c = classifier.train_single_model(
                model_name, train_emb, train_lbl, val_emb, val_lbl
            )
            classifier.save_model(clf, model_name, score, best_c)

        # Predict
        print(f"Generating predictions for {model_name}...")
        val_probs = clf.predict_proba(val_emb)
        test_probs = clf.predict_proba(test_emb)

        model_predictions[model_name] = {"val": val_probs, "test": test_probs}

    # Weighted Ensembling (Cite solution_lesson_node_00012)
    # If we have exactly 2 models, we optimize the weight w: w*A + (1-w)*B
    model_names = list(model_predictions.keys())

    if len(model_names) == 2:
        m1, m2 = model_names[0], model_names[1]
        print(f"Optimizing ensemble weights for {m1} and {m2}...")

        p1_val = model_predictions[m1]["val"]
        p2_val = model_predictions[m2]["val"]

        def objective(w):
            # Constrain w to [0, 1]
            w = np.clip(w, 0, 1)
            p_ensemble = w * p1_val + (1 - w) * p2_val
            return log_loss(val_labels, p_ensemble)

        res = minimize_scalar(objective, bounds=(0, 1), method="bounded")
        best_w = res.x
        print(f"Optimal weight for {m1}: {best_w:.4f}")
        print(f"Optimal weight for {m2}: {1 - best_w:.4f}")
        print(f"Ensemble Validation Loss: {res.fun:.5f}")

        # Apply to test
        val_probs_ensemble = best_w * p1_val + (1 - best_w) * p2_val
        test_probs_ensemble = (
            best_w * model_predictions[m1]["test"]
            + (1 - best_w) * model_predictions[m2]["test"]
        )

    elif len(model_names) == 1:
        print("Single model detected. Skipping ensemble optimization.")
        m1 = model_names[0]
        val_probs_ensemble = model_predictions[m1]["val"]
        test_probs_ensemble = model_predictions[m1]["test"]

    else:
        # Fallback to simple averaging for >2 models (or implement more complex optimization if needed)
        print(f"More than 2 models ({len(model_names)}). Using simple averaging.")
        val_probs_ensemble = sum(m["val"] for m in model_predictions.values()) / len(
            model_names
        )
        test_probs_ensemble = sum(m["test"] for m in model_predictions.values()) / len(
            model_names
        )

    # 4. Evaluation
    # Calculate Log Loss
    final_metric = log_loss(val_labels, val_probs_ensemble)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude: -log(p_true)
    # Get probability assigned to the true class
    true_class_probs = val_probs_ensemble[np.arange(len(val_labels)), val_labels]
    # Clip to avoid log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    error_magnitudes = -np.log(true_class_probs)

    # Load validation metadata to get image paths
    val_df = pd.read_csv(config.VAL_CSV)

    # Ensure alignment (DogDataset does not shuffle val, so indices match)
    if len(val_df) != len(error_magnitudes):
        print(
            "Warning: Validation dataframe length mismatch. Skipping detailed image analysis."
        )
    else:
        widths = []
        heights = []
        ratios = []

        # Read image dimensions
        # This is fast enough for ~1800 images
        for rel_path in val_df["file_path"]:
            full_path = os.path.join(config.INPUT_DIR, rel_path)
            try:
                with Image.open(full_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    ratios.append(w / h)
            except Exception:
                widths.append(0)
                heights.append(0)
                ratios.append(0)

        widths = np.array(widths)
        heights = np.array(heights)
        ratios = np.array(ratios)

        # Calculate correlations
        corr_w, _ = pearsonr(error_magnitudes, widths)
        corr_h, _ = pearsonr(error_magnitudes, heights)
        corr_r, _ = pearsonr(error_magnitudes, ratios)

        print(f"Correlation (Error vs Width): {corr_w:.4f}")
        print(f"Correlation (Error vs Height): {corr_h:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_r:.4f}")

    # 5. Submission
    threshold = 0.12293165333323357
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        class_names = classifier.get_class_names()
        classifier.generate_submission(
            test_ids, test_probs_ensemble, class_names, config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
