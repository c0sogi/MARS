import os
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.preprocessing as preprocessing
import library.model as model_lib


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Configuration Setup
    # Override working directory to keep demo artifacts separate
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    config.WORKING_DIR = demo_working_dir
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Reduce optimization iterations for the demo to ensure speed
    config.CONFIG["optimization"]["maxiter"] = 10

    print(f"Working directory set to: {config.WORKING_DIR}")

    # 2. Data Loading
    print("\n[Step 1] Loading Dataset...")
    # We use debug_size=100 to load a small subset for rapid verification
    X_train, y_train, X_val, y_val, X_test, ids_test = data_loader.load_dataset(
        load_cached_data=False, debug_size=100
    )

    # Assertions to verify data loading
    assert len(X_train) == 100, "X_train debug size mismatch"
    assert len(y_train) == 100, "y_train debug size mismatch"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"
    assert X_train.dtypes.iloc[0] == np.float64, "Features should be float64"
    print("Data loaded successfully. Shapes verified.")

    # 3. Preprocessing
    print("\n[Step 2] Preprocessing Features...")
    # Demonstrate the HighPrecisionTransformer directly
    transformer = preprocessing.HighPrecisionTransformer()

    # Fit on train, transform train
    X_train_trans = transformer.fit_transform(X_train)

    # Transform validation and test
    X_val_trans = transformer.transform(X_val)
    X_test_trans = transformer.transform(X_test)

    # Verify transformation properties
    assert isinstance(X_train_trans, np.ndarray), "Output should be numpy array"
    assert X_train_trans.dtype == np.float64, "Transformed data must remain float64"

    # Check standardization (Mean ~ 0, Std ~ 1) on training set
    # Note: With Yeo-Johnson + StandardScaling, this should hold generally
    means = np.mean(X_train_trans, axis=0)
    stds = np.std(X_train_trans, axis=0)
    assert np.all(np.abs(means) < 1e-6), "Transformed features not centered"
    assert np.all(np.abs(stds - 1.0) < 1e-6), "Transformed features not scaled"
    print("Preprocessing complete. Statistical properties verified.")

    # Demonstrate the caching wrapper function
    print("Testing preprocessing cache mechanism...")
    # This should save files to the demo working directory
    _ = preprocessing.get_transformed_data(
        X_train, X_val, X_test, load_cached_data=False
    )
    assert os.path.exists(
        os.path.join(demo_working_dir, "X_train_transformed.npy")
    ), "Cache file not created"
    print("Cache mechanism verified.")

    # 4. Model Training (Hybrid Generative-Discriminative)
    print("\n[Step 3] Training Calibrated OAS Discriminant Model...")
    clf = model_lib.CalibratedOASDiscriminant()

    # Fit the model
    # This triggers the Generative Phase (OAS-LDA) and Discriminative Phase (Calibration)
    clf.fit(X_train_trans, y_train)

    # Verify Model Attributes
    assert clf.classes_ is not None, "Classes not learned"
    assert clf.W_ is not None, "Projection matrix W not learned"
    assert clf.b_init_ is not None, "Initial bias not computed"
    assert clf.s_ is not None, "Scale parameter s not optimized"
    assert clf.b_ is not None, "Bias vector b not optimized"

    print(f"Model trained. Learned Scale (s): {clf.s_:.4f}")
    print(f"Number of classes: {len(clf.classes_)}")

    # 5. Prediction and Validation
    print("\n[Step 4] Generating Predictions...")

    # Predict on Validation Set
    val_probs = clf.predict_proba(X_val_trans)

    # Verify Probability Properties
    # 1. Shape
    assert val_probs.shape == (
        len(X_val),
        len(clf.classes_),
    ), "Probability shape mismatch"
    # 2. Sum to 1
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    # 3. Clipping (Values should be within [epsilon, 1-epsilon])
    epsilon = config.CLIP_EPSILON
    assert np.all(val_probs >= epsilon), "Probabilities violate lower bound clipping"
    assert np.all(
        val_probs <= 1.0 - epsilon
    ), "Probabilities violate upper bound clipping"

    # Calculate Log Loss
    # We need to encode y_val to match the model's classes
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(clf.classes_)
    # Handle potentially unseen labels in validation (though unlikely with stratified split)
    # For this demo, we assume y_val labels exist in training classes

    # Filter y_val to only include classes seen in training (for the debug subset scenario)
    valid_mask = np.isin(y_val, clf.classes_)
    if np.sum(valid_mask) < len(y_val):
        print(
            f"Note: {len(y_val) - np.sum(valid_mask)} validation samples skipped due to unseen classes in debug subset."
        )

    if np.sum(valid_mask) > 0:
        y_val_filtered = y_val[valid_mask]
        val_probs_filtered = val_probs[valid_mask]

        loss = log_loss(y_val_filtered, val_probs_filtered, labels=clf.classes_)
        print(f"Validation Log Loss (Debug Subset): {loss:.4f}")
    else:
        print("Skipping log loss calculation: No overlapping classes in debug subset.")

    # 6. Submission Generation
    print("\n[Step 5] Creating Submission File...")
    test_probs = clf.predict_proba(X_test_trans)

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=clf.classes_)
    submission_df.insert(0, "id", ids_test)

    # Save
    submission_path = os.path.join(demo_working_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created"
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    # Ensure reproducible execution
    np.random.seed(config.SEED)
    run_demo()
