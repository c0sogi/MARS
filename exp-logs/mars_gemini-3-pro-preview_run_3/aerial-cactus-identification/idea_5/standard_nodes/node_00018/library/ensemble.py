import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config


class EnsembleStacker:
    """
    Manages the stacking ensemble logic using a Logistic Regression meta-learner.
    Fuses predictions from heterogeneous base models (e.g., ResNet, DenseNet, EfficientNet).
    """

    def __init__(self):
        """
        Initialize the stacker with a Logistic Regression model.
        Uses 'liblinear' solver which is suitable for the dataset size and binary classification.
        """
        self.meta_learner = LogisticRegression(
            random_state=Config.SEED,
            solver="liblinear",
            C=1.0,  # Default regularization
        )
        self.model_names = []  # Stores the sorted order of models for consistency

    def fit_meta_learner(self, oof_preds_dict, ground_truth_dict):
        """
        Trains the Logistic Regression meta-learner on Out-Of-Fold (OOF) predictions.

        Args:
            oof_preds_dict (dict): Dictionary where keys are model names (e.g., 'resnet')
                                   and values are dictionaries of {image_id: probability}.
            ground_truth_dict (dict): Dictionary of {image_id: true_label}.

        Returns:
            sklearn.linear_model.LogisticRegression: The trained meta-learner.
        """
        # 1. Establish consistent model order
        self.model_names = sorted(oof_preds_dict.keys())
        print(f"Training Meta-Learner on models: {self.model_names}")

        # 2. Align data
        # Iterate over the ground truth IDs to ensure we cover the training set
        train_ids = sorted(list(ground_truth_dict.keys()))

        X_list = []
        y_list = []
        missing_ids = 0

        for img_id in train_ids:
            # Verify ID exists in all model OOF predictions
            if all(img_id in oof_preds_dict[m] for m in self.model_names):
                # Extract features in the fixed order
                features = [oof_preds_dict[m][img_id] for m in self.model_names]
                target = ground_truth_dict[img_id]

                X_list.append(features)
                y_list.append(target)
            else:
                missing_ids += 1

        if missing_ids > 0:
            print(
                f"Warning: {missing_ids} IDs from ground truth were missing in OOF predictions."
            )

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Meta-Learner Training Data Shape: {X.shape}")

        # 3. Fit the Logistic Regression model
        self.meta_learner.fit(X, y)

        # 4. Evaluate OOF Performance (Ensemble Score)
        # Since we are training on OOF predictions, the score on this set
        # represents the cross-validated performance of the ensemble.
        ensemble_probs = self.meta_learner.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, ensemble_probs)

        # Print full precision as requested
        print(f"Ensemble OOF AUC: {auc_score}")

        print("Meta-Learner Coefficients:")
        for name, coef in zip(self.model_names, self.meta_learner.coef_[0]):
            print(f"  {name}: {coef}")
        print(f"  Intercept: {self.meta_learner.intercept_[0]}")

        return self.meta_learner

    def predict_ensemble(self, test_preds_dict):
        """
        Generates final predictions for the test set using the trained meta-learner.

        Args:
            test_preds_dict (dict): Dictionary where keys are model names and values
                                    are dictionaries of {image_id: probability}.

        Returns:
            dict: Dictionary of {image_id: final_probability}.
        """
        if not self.model_names:
            raise RuntimeError("Meta-learner must be fitted before prediction.")

        # Verify input models match trained models
        input_models = sorted(test_preds_dict.keys())
        if input_models != self.model_names:
            raise ValueError(
                f"Model mismatch. Trained on {self.model_names}, "
                f"but received predictions for {input_models}."
            )

        # 1. Align Test Data
        # Get IDs from the first model (assuming all models predicted on the same test set)
        first_model = self.model_names[0]
        test_ids = sorted(list(test_preds_dict[first_model].keys()))

        X_test_list = []
        valid_ids = []

        for img_id in test_ids:
            if all(img_id in test_preds_dict[m] for m in self.model_names):
                features = [test_preds_dict[m][img_id] for m in self.model_names]
                X_test_list.append(features)
                valid_ids.append(img_id)
            else:
                print(f"Warning: ID {img_id} missing in some model predictions.")

        X_test = np.array(X_test_list)

        # 2. Predict
        # predict_proba returns [prob_class_0, prob_class_1]
        final_probs = self.meta_learner.predict_proba(X_test)[:, 1]

        # 3. Map back to IDs
        results = dict(zip(valid_ids, final_probs))

        return results


def save_submission(predictions, output_path):
    """
    Saves the predictions to a CSV file in the format required by the competition.

    Args:
        predictions (dict): Dictionary mapping image_id to probability.
        output_path (str): File path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to DataFrame
    # Sort by ID to ensure deterministic order
    ids = sorted(predictions.keys())
    probs = [predictions[i] for i in ids]

    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} rows.")
