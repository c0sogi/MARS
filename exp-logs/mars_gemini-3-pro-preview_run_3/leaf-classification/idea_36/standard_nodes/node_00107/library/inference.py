import os
import numpy as np
import pandas as pd
import logging
from library.config import Config
from library.utils import load_pickle, seed_everything
from library.feature_extraction import DualStreamExtractor
from library.densification import ManifoldDensifier


class InferenceEngine:
    """
    Handles the inference process for the Convex-Densified Selective-Topology LDA.
    Orchestrates data loading, densification, ensemble prediction, and aggregation.
    """

    def __init__(self, config: Config):
        """
        Initialize the InferenceEngine.

        Args:
            config (Config): Configuration object containing paths and parameters.
        """
        self.config = config
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        self.classes_path = os.path.join(self.models_dir, "classes.pkl")
        seed_everything(self.config.SEED)

    def aggregate_predictions(self, ids, model_probs_list, class_names):
        """
        Aggregates predictions by averaging first across centroids (per model),
        and then across all models in the ensemble.

        Args:
            ids (np.ndarray): Array of image IDs corresponding to the densified samples.
            model_probs_list (list): List of probability arrays (one per model),
                                     each of shape (N_dense_samples, N_classes).
            class_names (list): List of class names.

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated probabilities with 'id' column.
        """
        cumulative_df = None
        n_models = len(model_probs_list)

        if n_models == 0:
            raise ValueError("No models provided for aggregation.")

        logging.info(f"Aggregating predictions across {n_models} models...")

        for probs in model_probs_list:
            # Create DataFrame for this model's predictions
            df = pd.DataFrame(probs, columns=class_names)
            df["id"] = ids

            # Step 1: Average across centroids for each image
            # Group by 'id' and compute the mean.
            # sort=False preserves the order of appearance (which is deterministic in our pipeline)
            df_agg = df.groupby("id", sort=False).mean()

            # Step 2: Accumulate the per-image averages
            if cumulative_df is None:
                cumulative_df = df_agg
            else:
                cumulative_df = cumulative_df.add(df_agg)

        # Step 3: Average across models
        final_df = cumulative_df / n_models

        # Reset index to make 'id' a column again
        final_df = final_df.reset_index()

        return final_df

    def predict_all(self, test_metadata_path, dataset_name="test"):
        """
        Executes the full inference pipeline:
        1. Extracts features (Visual + Tabular).
        2. Densifies the manifold (Generates 6 centroids per image).
        3. Generates predictions using the loaded ensemble.
        4. Aggregates predictions.

        Args:
            test_metadata_path (str): Path to the test metadata CSV.
            dataset_name (str): Name of the dataset for caching purposes.

        Returns:
            pd.DataFrame: The final aggregated predictions.
        """
        logging.info(f"Starting inference on {dataset_name}...")

        # 1. Feature Extraction
        # Uses DualStreamExtractor to get 12-view visual features and tabular data
        extractor = DualStreamExtractor(self.config)
        raw_data = extractor.process_dataset(
            test_metadata_path, dataset_name, load_cached_data=True
        )

        # 2. Manifold Densification
        # Generates 6 centroids (3 Primary + 3 Secondary) per image
        densifier = ManifoldDensifier(self.config)
        dense_data = densifier.densify_dataset(
            raw_data, dataset_name, load_cached_data=True
        )

        # 3. Prepare Input Matrix
        # Concatenate DINO, ConvNeXt, and Tabular features
        X = np.hstack([dense_data["dino"], dense_data["conv"], dense_data["tab"]])
        ids = dense_data["ids"]

        # Load class names
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(
                f"Classes file not found at {self.classes_path}. Train the model first."
            )
        classes = load_pickle(self.classes_path)

        # 4. Ensemble Prediction
        model_probs_list = []

        for fold in range(self.config.N_FOLDS):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")

            if not os.path.exists(model_path):
                logging.warning(
                    f"Model for fold {fold} not found at {model_path}. Skipping."
                )
                continue

            logging.info(f"Predicting with model fold {fold}...")
            pipeline = load_pickle(model_path)

            # Predict probabilities for all densified samples (centroids)
            probs = pipeline.predict_proba(X)
            model_probs_list.append(probs)

        # 5. Aggregation
        final_predictions = self.aggregate_predictions(ids, model_probs_list, classes)

        return final_predictions

    def generate_submission(self, test_metadata_path, output_path):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_metadata_path (str): Path to the test metadata CSV.
            output_path (str): Path to save the submission CSV.
        """
        predictions_df = self.predict_all(test_metadata_path, dataset_name="test")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        predictions_df.to_csv(output_path, index=False)
        logging.info(f"Submission saved to {output_path}")
