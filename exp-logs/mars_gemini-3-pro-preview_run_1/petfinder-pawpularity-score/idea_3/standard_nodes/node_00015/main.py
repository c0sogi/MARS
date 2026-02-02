import os
import numpy as np
import pandas as pd
import warnings

# Import functionality from the provided library files
from library.utils import set_seed, SUBMISSION_PATH
from library.feature_extractor import DualBackboneExtractor
from library.model import RidgeRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(42)

    # 2. Feature Extraction
    # Initialize the extractor which manages Swin + ConvNeXt backbones
    extractor = DualBackboneExtractor()

    # Retrieve features for all splits.
    # load_cached_data=True ensures we use disk cache if available to save time.
    train_feats, train_meta, train_targets = extractor.get_features(
        "train", load_cached_data=True
    )
    val_feats, val_meta, val_targets = extractor.get_features(
        "val", load_cached_data=True
    )
    test_feats, test_meta, test_ids = extractor.get_features(
        "test", load_cached_data=True
    )

    # 3. Feature Fusion
    # Combine the deep learning image embeddings with the explicit metadata features
    X_train = np.hstack([train_feats, train_meta])
    X_val = np.hstack([val_feats, val_meta])
    X_test = np.hstack([test_feats, test_meta])

    # 4. Model Training
    # The RidgeRegressor pipeline includes StandardScaler and RidgeCV
    model = RidgeRegressor()
    model.fit(X_train, train_targets)

    # 5. Validation
    val_preds = model.predict(X_val)
    val_rmse = model.get_rmse(val_targets, val_preds)

    # Print the required metric
    print(f"Final Validation Metric: {val_rmse}")

    # 6. Failure Analysis
    # Identify systematic errors by correlating absolute error with metadata features
    abs_errors = np.abs(val_targets - val_preds)

    # Define metadata column names (order matches library.data.PetDataset)
    meta_cols = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # Create analysis DataFrame
    df_analysis = pd.DataFrame(val_meta, columns=meta_cols)
    df_analysis["Error"] = abs_errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["Error"].drop("Error").sort_values(ascending=False)
    )

    print("\nFailure Analysis - Correlation with Absolute Error:")
    print(correlations)

    # 7. Submission Generation
    # Only generate submission if the model meets the performance requirement
    threshold = 17.735125135690733

    if val_rmse < threshold:
        test_preds = model.predict(X_test)

        submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_rmse} did not beat threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
