import numpy as np
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as feature_engineering
import library.model as model_lib


def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Builds and trains the Logistic Regression model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).

    Returns:
        model: The trained LogisticRegression model.
    """
    # Instantiate the model using the library factory
    model = model_lib.build_model()

    # Train the model (and evaluate if validation data is provided)
    # The library function handles the fitting and internal metric printing
    model, metrics = model_lib.train_model(model, X_train, y_train, X_val, y_val)

    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model on the validation set and prints the Log Loss.

    Args:
        model: The trained model.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        float: The calculated log loss.
    """
    # Predict probabilities
    y_pred = model.predict_proba(X_val)

    # Calculate Log Loss using the competition-specific metric utility
    # We assume y_val matches the format expected by the model (strings)
    loss = utils.calculate_log_loss(y_val, y_pred)

    # Print full precision metric
    print(f"Validation Log Loss: {loss}")

    return loss


def run_training(debug=False, load_cached_data=True):
    """
    Executes the full training pipeline:
    1. Load and preprocess data.
    2. Extract features.
    3. Train model.
    4. Evaluate model.
    5. Generate submission.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        model: The trained model.
    """
    # Ensure reproducibility
    utils.set_seed(config.RANDOM_STATE)

    # Determine dataset size for debugging
    nrows = config.DEBUG_SAMPLE_SIZE if debug else None

    # 1. Load Data
    train_df, val_df, test_df = data_loader.load_and_preprocess_data(
        load_cached_data=load_cached_data, nrows=nrows
    )

    # 2. Extract Features
    # Returns sparse matrices for X and integer-encoded arrays for y
    X_train, y_train, X_val, y_val, X_test = feature_engineering.extract_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Convert integer labels back to strings for compatibility.
    # library.model and library.utils default to using config.CLASSES (strings).
    # Training on strings ensures model.classes_ matches config.CLASSES,
    # making predict_proba columns align correctly with submission headers.
    classes = np.array(config.CLASSES)
    y_train_str = classes[y_train]
    y_val_str = classes[y_val]

    # 3. Train Model
    # We pass validation data here so the library function can also monitor performance
    model = train_model(X_train, y_train_str, X_val, y_val_str)

    # 4. Evaluate Model
    # Explicit evaluation call as requested
    evaluate_model(model, X_val, y_val_str)

    # 5. Generate Submission
    print("Generating predictions for test set...")
    y_test_pred = model.predict_proba(X_test)

    # Retrieve test IDs
    test_ids = test_df["id"].values

    # Save submission
    utils.save_submission(test_ids, y_test_pred)

    return model
