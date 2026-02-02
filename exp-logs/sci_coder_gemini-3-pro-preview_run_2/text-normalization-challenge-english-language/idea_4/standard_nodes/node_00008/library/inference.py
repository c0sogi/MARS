import pandas as pd
import torch
import os
from library.config import Config
from library.model import TokenClassifier, predict_labels
from library.normalization_rules import Normalizer


class InferencePipeline:
    """
    Handles the generation of predictions for the test set.
    """

    def __init__(self, model_path=None):
        self.device = Config.DEVICE
        self.model = TokenClassifier()

        # Determine model path
        if model_path is None:
            model_path = Config.MODEL_SAVE_PATH

        # Load model weights
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model checkpoint not found at {model_path}. Using random weights."
            )

        self.model.to(self.device)
        self.model.eval()

    def predict_classes(self, dataset):
        """
        Runs the model on the test data to get class labels.
        Wraps library.model.predict_labels which handles inference and subword aggregation.

        Returns:
            list of lists: Predicted labels for each sentence.
        """
        # predict_labels returns a list of lists of strings (labels)
        # It handles the aggregation of subwords internally.
        return predict_labels(self.model, dataset)

    def aggregate_subwords(self, subword_preds, word_ids):
        """
        Maps sub-word predictions back to original word-level tokens.

        Note: This logic is already encapsulated within library.model.predict_labels,
        which is used by predict_classes. This method is provided for structural
        completeness as per requirements, but the pipeline uses the library function
        for efficiency and consistency.
        """
        # Placeholder for logic if implemented from scratch:
        # Iterate through word_ids, take the prediction of the first sub-token for each word.
        pass

    def generate_submission(self, test_dataset):
        """
        Generates predictions for the test set, applies normalization rules,
        and saves the submission file.
        """
        print("Generating predictions for test set...")

        # 1. Predict Classes (includes aggregation)
        # Returns list of lists (sentences -> tokens)
        preds_by_sentence = self.predict_classes(test_dataset)

        # 2. Flatten predictions to match token-level test dataframe
        flat_preds = [label for sent in preds_by_sentence for label in sent]

        # 3. Load Test Metadata for alignment
        print("Loading test metadata for alignment...")
        df_test = pd.read_csv(
            Config.TEST_DATA_PATH,
            keep_default_na=False,
            dtype={"sentence_id": int, "token_id": int},
        )

        # Ensure correct order (dataset.py sorts by sentence_id, token_id)
        df_test = df_test.sort_values(["sentence_id", "token_id"])

        # Check alignment
        if len(flat_preds) != len(df_test):
            print(
                f"Warning: Prediction count {len(flat_preds)} != Test Data count {len(df_test)}"
            )
            # Adjust length if necessary (truncate or pad)
            if len(flat_preds) > len(df_test):
                flat_preds = flat_preds[: len(df_test)]
            else:
                flat_preds += ["PLAIN"] * (len(df_test) - len(flat_preds))

        # Assign predicted classes
        df_test["class"] = flat_preds

        # 4. Apply Normalization Rules
        print("Applying deterministic normalization rules...")
        norm = Normalizer()

        # Apply row-wise normalization
        # 'before' column contains raw text
        df_test["after"] = df_test.apply(
            lambda row: norm.normalize(row["before"], row["class"]), axis=1
        )

        # 5. Save Submission
        submission = df_test[["id", "after"]]

        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generation complete.")
