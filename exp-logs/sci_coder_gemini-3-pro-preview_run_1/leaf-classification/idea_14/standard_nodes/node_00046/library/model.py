from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import SUBMISSION_PATH
from library.utils import compute_log_loss, save_submission
from library.preprocessing import get_preprocessed_data


class LeafLDA(LinearDiscriminantAnalysis):
    """
    Wrapper around sklearn's LinearDiscriminantAnalysis with optimal settings.
    Cite {solution_lesson_node_00038}: Prefer 'eigen' solver for precision.
    """

    def __init__(self):
        super().__init__(solver="eigen", shrinkage="auto")


def train_and_predict(load_cached_data=True):
    """
    Orchestrates the training process:
    1. Loads preprocessed (Gaussianized) data.
    2. Trains the OASDiscriminant model.
    3. Evaluates on the validation set.
    4. Generates predictions for the test set.
    5. Saves the submission file.
    """
    print("Loading data...")
    # Load data using the preprocessing pipeline (Iterative Gaussianization)
    (train_data, val_data, test_data) = get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # Initialize Model
    print("Initializing LeafLDA model...")
    model = LeafLDA()

    # Fit Model
    print("Fitting model on training set...")
    model.fit(X_train, y_train)

    # Validate
    print("Evaluating on validation set...")
    y_val_pred = model.predict_proba(X_val)
    val_loss = compute_log_loss(y_val, y_val_pred, model.classes_)

    print("-" * 30)
    print(f"Validation Multi-class Log Loss: {val_loss}")
    print("-" * 30)

    # Predict on Test
    print("Generating predictions for test set...")
    y_test_pred = model.predict_proba(X_test)

    # Save Submission
    save_submission(ids_test, y_test_pred, model.classes_, filename=SUBMISSION_PATH)
    print("Process completed successfully.")
