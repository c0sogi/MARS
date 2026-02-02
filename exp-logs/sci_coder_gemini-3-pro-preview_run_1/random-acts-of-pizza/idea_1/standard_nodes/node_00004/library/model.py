import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from library.config import XGB_PARAMS, RANDOM_STATE
from library.preprocessor import create_preprocessor
from library.utils import print_metrics


class PizzaXGBoost:
    """
    A wrapper class for the XGBoost model pipeline to predict pizza request success.
    """

    def __init__(self, n_estimators=None, max_depth=None, random_state=RANDOM_STATE):
        """
        Initialize the PizzaXGBoost model.

        Args:
            n_estimators (int, optional): Number of trees. Overrides config if provided.
            max_depth (int, optional): Maximum depth of the tree. Overrides config if provided.
            random_state (int): Seed for reproducibility.
        """
        # Copy default params from config
        self.params = XGB_PARAMS.copy()

        # Override with provided arguments if they are not None
        if n_estimators is not None:
            self.params["n_estimators"] = n_estimators
        if max_depth is not None:
            self.params["max_depth"] = max_depth

        self.params["random_state"] = random_state

        # Create the preprocessor
        self.preprocessor = create_preprocessor()

        # Initialize the Classifier
        self.classifier = XGBClassifier(**self.params)

        # Construct the Pipeline
        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", self.classifier)]
        )

    def train(self, train_df, target_col):
        """
        Trains the XGBoost pipeline.

        Args:
            train_df (pd.DataFrame): Training data containing features and target.
            target_col (str): Name of the target column.
        """
        print(f"Training XGBoost with {self.params['n_estimators']} estimators...")

        # Separate features and target
        X_train = train_df.drop(columns=[target_col], errors="ignore")
        y_train = train_df[target_col]

        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def predict_proba(self, df):
        """
        Generates probability predictions for the positive class.

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            np.ndarray: Array of probabilities for the positive class (True).
        """
        # The pipeline expects a DataFrame with the same columns as training X
        # We don't need to manually drop target here if it's not present,
        # but ColumnTransformer is robust to extra columns if they aren't specified in transformers.
        # However, to be safe and consistent with sklearn API:

        # Predict probabilities
        # classes_ usually maps to [False, True] or [0, 1] sorted.
        # We return the probability of the second class (index 1), which corresponds to True/1.
        probs = self.pipeline.predict_proba(df)[:, 1]
        return probs

    def evaluate(self, val_df, target_col, set_name="Validation"):
        """
        Evaluates the model on a validation set using ROC AUC.

        Args:
            val_df (pd.DataFrame): Validation data containing features and target.
            target_col (str): Name of the target column.
            set_name (str): Name of the dataset for logging purposes.

        Returns:
            float: The ROC AUC score.
        """
        X_val = val_df.drop(columns=[target_col], errors="ignore")
        y_val = val_df[target_col]

        # Generate predictions
        y_pred_proba = self.predict_proba(X_val)

        # Calculate and print metrics using the utility function
        print_metrics(y_val, y_pred_proba, set_name=set_name)

        return y_pred_proba
