import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


class MetricOptimizer:
    """
    Optimizes classification thresholds to maximize Matthews Correlation Coefficient (MCC)
    and generates the final submission file.
    """

    def __init__(self):
        self.submission_path = Config.SUBMISSION_PATH

    def optimize_thresholds(self, models, val_data):
        """
        Finds the optimal probability threshold for each stream independently
        by maximizing the MCC on the validation set.

        Args:
            models (dict): Dictionary of trained models for 'stream_a' and 'stream_b'.
            val_data (dict): Dictionary containing validation data ('X', 'y') for both streams.

        Returns:
            dict: Optimal thresholds for 'stream_a' and 'stream_b'.
        """
        thresholds = {}
        search_space = np.linspace(0.01, 0.99, 99)

        for stream in ["stream_a", "stream_b"]:
            # Check if model and data exist for this stream
            if stream not in models or models[stream] is None:
                print(f"[{stream.upper()}] No model found. Defaulting threshold to 0.5")
                thresholds[stream] = 0.5
                continue

            if stream not in val_data or len(val_data[stream]["X"]) == 0:
                print(
                    f"[{stream.upper()}] No validation data. Defaulting threshold to 0.5"
                )
                thresholds[stream] = 0.5
                continue

            print(f"[{stream.upper()}] Optimizing threshold...")

            # Get Data
            X_val = val_data[stream]["X"]
            y_true = val_data[stream]["y"]

            # Predict Probabilities
            # Note: XGBClassifier.predict_proba returns (n_samples, n_classes)
            model = models[stream]
            y_prob = model.predict_proba(X_val)[:, 1]

            # Linear Search for Best MCC
            best_mcc = -1.0
            best_thresh = 0.5

            for thresh in search_space:
                y_pred = (y_prob >= thresh).astype(int)
                # matthews_corrcoef handles zero denominator gracefully (returns 0)
                mcc = matthews_corrcoef(y_true, y_pred)

                if mcc > best_mcc:
                    best_mcc = mcc
                    best_thresh = thresh

            print(
                f"[{stream.upper()}] Best Threshold: {best_thresh:.2f}, MCC: {best_mcc:.16f}"
            )
            thresholds[stream] = best_thresh

        return thresholds

    def generate_submission(self, predictions_df, thresholds):
        """
        Applies the optimized thresholds to the test predictions and saves the submission file.
        Differentiates between Stream A (Player-Player) and Stream B (Player-Ground)
        based on the contact_id.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'contact_id' and 'score' (probability).
            thresholds (dict): Dictionary containing optimal thresholds for 'stream_a' and 'stream_b'.
        """
        print("Generating submission file...")

        if predictions_df.empty:
            print("Warning: Predictions DataFrame is empty. Creating empty submission.")
            # Create empty file with correct header
            df_sub = pd.DataFrame(columns=["contact_id", "contact"])
            os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
            df_sub.to_csv(self.submission_path, index=False)
            return

        # Work on a copy to ensure safety
        df = predictions_df.copy()

        # Identify Stream B (Ground Contact) rows
        # Format: game_play_step_p1_p2. If p2 is 'G', the ID ends with '_G'.
        is_stream_b = df["contact_id"].str.endswith("_G")

        # Retrieve thresholds (default to 0.5 if missing)
        thresh_a = thresholds.get("stream_a", 0.5)
        thresh_b = thresholds.get("stream_b", 0.5)

        print(f"Applying Thresholds -> Stream A: {thresh_a}, Stream B: {thresh_b}")

        # Initialize contact column
        df["contact"] = 0

        # Apply Stream A Threshold (Player-Player)
        mask_a = ~is_stream_b
        if mask_a.any():
            df.loc[mask_a, "contact"] = (df.loc[mask_a, "score"] >= thresh_a).astype(
                int
            )

        # Apply Stream B Threshold (Player-Ground)
        mask_b = is_stream_b
        if mask_b.any():
            df.loc[mask_b, "contact"] = (df.loc[mask_b, "score"] >= thresh_b).astype(
                int
            )

        # Select required columns
        submission = df[["contact_id", "contact"]]

        # Save to disk
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        submission.to_csv(self.submission_path, index=False)

        print(f"Submission saved to {self.submission_path}")
        print(f"Submission shape: {submission.shape}")
        print(f"Predicted Contacts: {submission['contact'].sum()}")
