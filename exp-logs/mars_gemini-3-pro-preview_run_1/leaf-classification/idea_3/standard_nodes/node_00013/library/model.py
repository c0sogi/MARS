from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config, utils, preprocessing


def get_model():
    """
    Returns the configured Linear Discriminant Analysis model.
    Bagging is removed as it degrades performance for stable classifiers (Cite solution_lesson_node_00011).
    """
    return LinearDiscriminantAnalysis(
        solver=config.LDA_SOLVER, shrinkage=config.LDA_SHRINKAGE
    )


def train_model(load_cached_data=True):
    """
    Loads preprocessed data, trains the LDA model,
    and evaluates it on the validation set.
    """
    # 1. Load Preprocessed Data
    # The preprocessing pipeline (PowerTransformer + StandardScaler) is handled in library/preprocessing.py
    print("Loading preprocessed data for training...")
    X_train, y_train, X_val, y_val, _, _, _ = preprocessing.get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing LDA model...")
    model = get_model()

    # 3. Fit Model
    print("Fitting model...")
    model.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating on validation set...")
    y_pred_val = model.predict_proba(X_val)

    # Calculate Log Loss
    loss = utils.calculate_log_loss(y_val, y_pred_val)
    print(f"Validation Log Loss: {loss}")

    return model


def predict_and_submit(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.
    """
    # 1. Load Test Data
    print("Loading preprocessed test data...")
    _, _, _, _, X_test, test_ids, classes = preprocessing.get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    # 2. Generate Predictions
    print("Predicting probabilities for test set...")
    y_pred_test = model.predict_proba(X_test)

    # 3. Save Submission
    utils.save_submission(
        test_ids, y_pred_test, classes, output_path=config.SUBMISSION_CSV
    )
