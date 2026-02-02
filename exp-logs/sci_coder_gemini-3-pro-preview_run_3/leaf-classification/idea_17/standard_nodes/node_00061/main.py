import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_log_loss, clip_probabilities
from library.data_processing import LeafDataProcessor
from library.modeling import StackedEnsemble


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Create the specific submission directory required by the prompt
    os.makedirs("./submission", exist_ok=True)

    print("Initializing Data Processor...")
    # load_raw_cache=True attempts to load the extracted features (DINO/ConvNeXt) from disk
    # to save time if they have already been computed.
    processor = LeafDataProcessor(load_raw_cache=True)
    classes = processor.get_all_classes()

    # Containers for global evaluation (Validation)
    val_preds_list = []
    val_targets_list = []
    val_features_list = []

    # Container for test predictions (summed for averaging)
    test_preds_sum = None
    test_ids = None

    # 2. Cross-Validation Loop
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold in range(Config.NUM_FOLDS):
        print(f"\n--- Fold {fold} ---")

        # Load fold data
        # This handles hyper-densification (9 centroids for train) and PCA/Quantile transforms
        data = processor.get_fold_data(fold, load_cache=True)

        # Initialize and Train Model
        # The StackedEnsemble handles the Inner CV for Meta-Learner training
        model = StackedEnsemble()
        model.fit(
            data["X_dino_train"],
            data["X_conv_train"],
            data["X_tab_train"],
            data["y_train"],
            data["ids_train"],
        )

        # Save Model
        model_path = os.path.join(Config.MODELS_DIR, f"model_fold_{fold}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Inference on Validation Set
        val_probs = model.predict_proba(
            data["X_dino_val"], data["X_conv_val"], data["X_tab_val"]
        )

        # Store results for global metric calculation
        val_preds_list.append(val_probs)
        val_targets_list.append(data["y_val"])
        val_features_list.append(data["X_tab_val"])

        # Inference on Test Set
        test_probs = model.predict_proba(
            data["X_dino_test"], data["X_conv_test"], data["X_tab_test"]
        )

        # Accumulate Test Predictions
        if test_preds_sum is None:
            test_preds_sum = test_probs
            test_ids = data["ids_test"]
        else:
            test_preds_sum += test_probs

    # 3. Global Evaluation
    print("\n--- Global Evaluation ---")
    val_preds_concat = np.concatenate(val_preds_list, axis=0)
    val_targets_concat = np.concatenate(val_targets_list, axis=0)
    val_features_concat = np.concatenate(val_features_list, axis=0)

    # Calculate Log Loss
    # We pass the class indices [0..98] as labels to ensure correct mapping
    metric = calculate_log_loss(
        val_targets_concat, val_preds_concat, labels=np.arange(len(classes))
    )
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(val_preds_concat, epsilon, 1.0 - epsilon)
    # Normalize rows (standard procedure before scoring)
    preds_norm = preds_clipped / preds_clipped.sum(axis=1, keepdims=True)

    # Extract probability assigned to the true class
    n_samples = len(val_targets_concat)
    true_class_probs = preds_norm[np.arange(n_samples), val_targets_concat]
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation with features
    correlations = []

    # Features are: margin(64) + shape(64) + texture(64) = 192 total
    for i in range(val_features_concat.shape[1]):
        feat_values = val_features_concat[:, i]
        # Handle constant features to avoid NaN correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_losses, feat_values)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    feature_types = ["margin"] * 64 + ["shape"] * 64 + ["texture"] * 64
    feature_indices = list(range(1, 65)) * 3

    for idx, corr in correlations[:5]:
        feat_name = f"{feature_types[idx]}_{feature_indices[idx]}"
        print(f"  {feat_name}: {corr:.4f}")

    # 5. Submission Generation
    print("\n--- Generating Submission ---")

    # Average predictions across folds
    avg_test_preds = test_preds_sum / Config.NUM_FOLDS

    # Row-wise normalization
    row_sums = avg_test_preds.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    avg_test_preds = avg_test_preds / row_sums[:, np.newaxis]

    # Create DataFrame
    submission_df = pd.DataFrame(avg_test_preds, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save to prompt-specified location
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Also save to working directory for reference
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)


if __name__ == "__main__":
    main()
