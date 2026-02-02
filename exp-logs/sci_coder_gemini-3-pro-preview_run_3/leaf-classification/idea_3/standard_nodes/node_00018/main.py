import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.feature_extraction import FeatureExtractor
from library.preprocessing import FusionPipeline
from library.model import EnsembleClassifier


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.setup()
    print("Starting execution...")

    # 2. Feature Extraction
    # Loads data, extracts features using Dual-Backbones (CNN+ViT) with 4-view averaging,
    # and retrieves tabular features. Uses caching to minimize runtime.
    extractor = FeatureExtractor()
    feature_data = extractor.run(load_cached_data=True)

    train_data = feature_data["train"]
    val_data = feature_data["val"]
    test_data = feature_data["test"]

    # 3. Preprocessing and Feature Fusion
    # Fits StandardScaler (tabular) and PCA (images) on training data, then transforms all sets.
    pipeline = FusionPipeline()
    pipeline.fit(train_data, load_cached_data=True)

    X_train = pipeline.transform(train_data)
    y_train = train_data["tgt"]

    X_val = pipeline.transform(val_data)
    y_val = val_data["tgt"]

    X_test = pipeline.transform(test_data)
    test_ids = test_data["ids"]

    print(
        f"Fused Feature Dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # 4. Model Training
    # Trains Logistic Regression and LDA on the fused training features.
    classifier = EnsembleClassifier()
    classifier.fit(X_train, y_train)

    # 5. Validation and Optimization
    # Optimizes the ensemble weight (alpha * LDA + (1-alpha) * LR) on validation data.
    best_weight = classifier.optimize_ensemble_weight(X_val, y_val)

    # Generate validation probabilities with the optimal weight
    val_probs = classifier.predict(X_val, weight=best_weight)

    # Calculate Final Validation Metric (Log Loss)
    # Note: calculate_log_loss handles clipping and row-normalization internally
    final_val_metric = calculate_log_loss(
        y_val, val_probs, labels=classifier.lr.classes_
    )
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude (log loss) per sample
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    # Get probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val.astype(int)]
    sample_losses = -np.log(true_class_probs)

    # Correlate error magnitude with original tabular features
    val_tabular = val_data["tab"]
    num_features = val_tabular.shape[1]

    # Reconstruct feature names based on dataset description
    feature_names = []
    for i in range(1, 65):
        feature_names.append(f"margin_{i}")
    for i in range(1, 65):
        feature_names.append(f"shape_{i}")
    for i in range(1, 65):
        feature_names.append(f"texture_{i}")

    correlations = []
    for i in range(num_features):
        feat_vals = val_tabular[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_vals) > 0:
            corr = np.corrcoef(feat_vals, sample_losses)[0, 1]
            correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    # Strict threshold check as per requirements
    threshold = 1.2267462032028139e-12

    if final_val_metric < threshold:
        print(
            f"\nValidation metric ({final_val_metric}) is lower than threshold ({threshold}). Generating submission..."
        )

        # Generate predictions for test set
        test_probs = classifier.predict(X_test, weight=best_weight)

        # Load class names
        classes = np.load(
            Config.get_cache_path(Config.CACHE_CLASSES), allow_pickle=True
        )

        # Load sample submission to ensure correct column order
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        sample_df = pd.read_csv(sample_sub_path)
        sample_cols = [col for col in sample_df.columns if col != "id"]

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids})

        # Assign probabilities to class columns
        # Ensure we map the classifier's class order to the dataframe columns
        for i, class_name in enumerate(classes):
            submission[class_name] = test_probs[:, i]

        # Ensure column order matches sample submission
        final_cols = ["id"] + sample_cols

        # Fill missing columns with 0 if any (though classes should match)
        for col in sample_cols:
            if col not in submission.columns:
                submission[col] = 0.0

        submission = submission[final_cols]

        # Save submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_val_metric}) is NOT lower than threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
