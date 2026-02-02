import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import library.config as cfg
import library.utils as utils
from library.feature_extractor import DualStreamExtractor
from library.data_processor import CentroidGenerator
from library.model_factory import SelectiveFeaturePipeline


class CrossValidationRunner:
    """
    Orchestrates the Stratified K-Fold Cross-Validation and training of the
    Orthogonal-Expert Ensemble.
    """

    def __init__(self):
        self.n_folds = cfg.N_FOLDS
        self.seed = cfg.SEED
        self.n_experts = cfg.N_EXPERTS
        utils.seed_everything(self.seed)

    def run(self, load_cached_data: bool = True):
        """
        Executes the full training and evaluation pipeline.

        Args:
            load_cached_data (bool): Whether to load features/centroids from cache.
        """
        print("Initializing Cross-Validation Runner...")

        # 1. Feature Extraction
        # Retrieves raw features (N, 36, D) and tabular data
        extractor = DualStreamExtractor()
        raw_data = extractor.extract_all_rotations(load_cached_data=load_cached_data)

        train_img = raw_data["train_img"]
        train_ids = raw_data["train_ids"]
        train_labels_raw = raw_data["train_labels"]
        train_tab = raw_data["train_tab"]

        test_img = raw_data["test_img"]
        test_ids = raw_data["test_ids"]
        test_tab = raw_data["test_tab"]

        # 2. Centroid Generation
        # Computes (N, 9, D_vis) centroids
        processor = CentroidGenerator()
        centroid_data = processor.process_features(
            raw_data=raw_data, load_cached_data=load_cached_data
        )

        train_centroids = centroid_data["train_centroids"]
        test_centroids = centroid_data["test_centroids"]

        # 3. Label Encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(train_labels_raw)
        class_names = list(le.classes_)
        n_classes = len(class_names)

        print(f"Data prepared. Classes: {n_classes}, Samples: {len(train_ids)}")

        # 4. Cross-Validation Setup
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        # Storage for OOF predictions and Test predictions
        oof_preds = np.zeros((len(train_ids), n_classes))
        test_preds_accumulator = np.zeros((len(test_ids), n_classes))

        fold_scores = []

        # 5. Training Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_ids, y_encoded)):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")

            # Split Tabular Data and Labels
            X_tab_train, X_tab_val = train_tab[train_idx], train_tab[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            # Initialize accumulators for this fold's expert ensemble
            fold_val_probs = np.zeros((len(val_idx), n_classes))
            fold_test_probs = np.zeros((len(test_ids), n_classes))

            # Expert Loop
            for expert_k in range(self.n_experts):
                # Prepare Expert Data: [Centroid_k | Tabular]
                # Train
                X_train_k = processor.prepare_expert_dataset(
                    train_centroids[train_idx], X_tab_train, expert_k
                )
                # Val
                X_val_k = processor.prepare_expert_dataset(
                    train_centroids[val_idx], X_tab_val, expert_k
                )
                # Test
                X_test_k = processor.prepare_expert_dataset(
                    test_centroids, test_tab, expert_k
                )

                # Create and Fit Pipeline
                # We assume standard dimensions: DINO(1024) + ConvNeXt(1536) = 2560 visual dim
                # If dimensions differ, they should be passed here. Using defaults from factory.
                pipeline = SelectiveFeaturePipeline().create_expert_pipeline()
                pipeline.fit(X_train_k, y_train)

                # Predict
                val_probs_k = pipeline.predict_proba(X_val_k)
                test_probs_k = pipeline.predict_proba(X_test_k)

                # Accumulate (Arithmetic Mean Strategy)
                fold_val_probs += val_probs_k
                fold_test_probs += test_probs_k

            # Average across experts
            fold_val_probs /= self.n_experts
            fold_test_probs /= self.n_experts

            # Store OOF predictions
            oof_preds[val_idx] = fold_val_probs

            # Accumulate Test predictions for this fold
            test_preds_accumulator += fold_test_probs

            # Calculate Fold Score
            fold_score = utils.calculate_log_loss(y_val, fold_val_probs)
            fold_scores.append(fold_score)
            print(f"Fold {fold + 1} Log Loss: {fold_score}")

        # 6. Final Aggregation
        overall_score = np.mean(fold_scores)
        print(f"\n========================================")
        print(f"Cross-Validation Complete")
        print(f"Average Log Loss: {overall_score}")
        print(f"========================================")

        # Average test predictions across folds
        final_test_preds = test_preds_accumulator / self.n_folds

        # 7. Save Submission
        utils.save_submission(
            ids=test_ids,
            probs=final_test_preds,
            class_names=class_names,
            output_path=cfg.SUBMISSION_PATH,
        )
