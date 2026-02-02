import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_loader import DataLoader
from library.text_encoder import TextEncoder
from library.tabular_processor import TabularProcessor
import library.model_factory as model_factory


class Trainer:
    """
    Orchestrates the training, validation, and submission generation process
    using the Supervised Semantic Projection Ensemble (SSPE) strategy.
    """

    def __init__(self):
        """
        Initialize the Trainer with necessary processors and directory setup.
        """
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.text_encoder = TextEncoder()
        self.tabular_processor = TabularProcessor()

    def train_and_submit(self, load_cached_data: bool = True):
        """
        Executes the full pipeline:
        1. Loads and preprocesses data.
        2. Runs Stratified Cross-Validation with nested Grid Search.
        3. Saves the best model artifacts for each fold.
        4. Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to use cached features if available.
        """
        # ==========================================
        # 1. Data Loading & Preparation
        # ==========================================
        print("Loading data...")
        df_train, df_val, df_test = DataLoader.load_data(
            load_cached_data=load_cached_data
        )

        # Merge Train and Validation sets for 5-Fold CV
        # We use the full labeled dataset to maximize training signal
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        y = df_full["requester_received_pizza"].values

        # ==========================================
        # 2. Feature Extraction
        # ==========================================
        print("Extracting features...")
        # Text Embeddings (MPNet) - Loaded from cache or computed
        # We process original splits first to leverage specific cache paths defined in Config
        X_text_train = self.text_encoder.encode(
            df_train, Config.TRAIN_EMBEDDINGS_PATH, load_cached_data
        )
        X_text_val = self.text_encoder.encode(
            df_val, Config.VAL_EMBEDDINGS_PATH, load_cached_data
        )
        # Stack for CV
        X_text = np.vstack([X_text_train, X_text_val])

        # Tabular Features - Processed on the fly (fast) or cached if implemented in processor
        X_tab_train = self.tabular_processor.process(df_train)
        X_tab_val = self.tabular_processor.process(df_val)
        # Stack for CV
        X_tab = np.vstack([X_tab_train, X_tab_val])

        # ==========================================
        # 3. Stratified Cross-Validation Loop
        # ==========================================
        skf = StratifiedKFold(
            n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED
        )

        oof_preds = np.zeros(len(y))
        fold_scores = []

        print(f"\nStarting {Config.N_SPLITS}-Fold Stratified Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
            print(f"\n--- Fold {fold} ---")

            # Split Data
            X_text_tr, X_text_va = X_text[train_idx], X_text[val_idx]
            X_tab_tr, X_tab_va = X_tab[train_idx], X_tab[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Grid Search Variables
            best_fold_score = -1.0
            best_artifacts = None
            best_params = None

            # --- Grid Search: PLS Components ---
            for n_comp in Config.PLS_N_COMPONENTS_GRID:
                # 1. Supervised Projection (PLS)
                # Fit only on training data to prevent leakage
                pls = model_factory.get_pls_transformer(n_components=n_comp)
                pls.fit(X_text_tr, y_tr)

                # Transform Text
                X_text_tr_pls = pls.transform(X_text_tr)
                X_text_va_pls = pls.transform(X_text_va)

                # 2. Scale PLS Output (StandardScaler)
                pls_scaler = model_factory.get_standard_scaler()
                X_text_tr_pls = pls_scaler.fit_transform(X_text_tr_pls)
                X_text_va_pls = pls_scaler.transform(X_text_va_pls)

                # 3. Scale Tabular Data (RankGauss)
                tab_scaler = model_factory.get_scaler(output_distribution="normal")
                X_tab_tr_scaled = tab_scaler.fit_transform(X_tab_tr)
                X_tab_va_scaled = tab_scaler.transform(X_tab_va)

                # 4. Feature Fusion
                X_tr_final = np.hstack([X_text_tr_pls, X_tab_tr_scaled])
                X_va_final = np.hstack([X_text_va_pls, X_tab_va_scaled])

                # --- Grid Search: Classifier Hyperparameters ---
                for C in Config.LR_C_GRID:
                    for cw in Config.LR_CLASS_WEIGHT_GRID:
                        # Train Classifier
                        clf = model_factory.get_classifier(
                            C=C,
                            class_weight=cw,
                            n_estimators=Config.N_BAGGING_ESTIMATORS,
                            n_jobs=Config.N_JOBS,
                            random_state=Config.SEED,
                        )
                        clf.fit(X_tr_final, y_tr)

                        # Evaluate
                        y_pred = clf.predict_proba(X_va_final)[:, 1]
                        score = roc_auc_score(y_va, y_pred)

                        # Update Best
                        if score > best_fold_score:
                            best_fold_score = score
                            best_params = {"n_comp": n_comp, "C": C, "cw": cw}
                            best_artifacts = {
                                "pls": pls,
                                "pls_scaler": pls_scaler,
                                "tab_scaler": tab_scaler,
                                "clf": clf,
                            }

            print(f"Fold {fold} Best Score: {best_fold_score}")
            print(f"Fold {fold} Best Params: {best_params}")
            fold_scores.append(best_fold_score)

            # Save Best Artifacts for this fold
            self._save_artifacts(fold, best_artifacts)

            # Generate OOF Predictions using best model
            pls = best_artifacts["pls"]
            pls_scaler = best_artifacts["pls_scaler"]
            tab_scaler = best_artifacts["tab_scaler"]
            clf = best_artifacts["clf"]

            # Re-transform validation set with best artifacts
            X_t_va_p = pls_scaler.transform(pls.transform(X_text_va))
            X_tb_va_s = tab_scaler.transform(X_tab_va)
            X_va_f = np.hstack([X_t_va_p, X_tb_va_s])

            oof_preds[val_idx] = clf.predict_proba(X_va_f)[:, 1]

        # ==========================================
        # 4. Overall Evaluation
        # ==========================================
        overall_auc = roc_auc_score(y, oof_preds)
        print("\n--- Cross-Validation Results ---")
        print(f"Mean Fold AUC: {np.mean(fold_scores)}")
        print(f"Overall OOF AUC: {overall_auc}")

        # ==========================================
        # 5. Test Prediction & Submission
        # ==========================================
        print("\nGenerating Test Predictions...")
        self._predict_test(df_test, load_cached_data)

    def _save_artifacts(self, fold, artifacts):
        """Helper to save trained models and scalers."""
        joblib.dump(
            artifacts["pls"],
            os.path.join(self.models_dir, f"pls_fold_{fold}.joblib"),
        )
        joblib.dump(
            artifacts["pls_scaler"],
            os.path.join(self.models_dir, f"pls_scaler_fold_{fold}.joblib"),
        )
        joblib.dump(
            artifacts["tab_scaler"],
            os.path.join(self.models_dir, f"tab_scaler_fold_{fold}.joblib"),
        )
        joblib.dump(
            artifacts["clf"],
            os.path.join(self.models_dir, f"clf_fold_{fold}.joblib"),
        )

    def _predict_test(self, df_test, load_cached_data):
        """Helper to generate predictions on test set using the ensemble of fold models."""
        # 1. Extract Test Features
        X_text_test = self.text_encoder.encode(
            df_test, Config.TEST_EMBEDDINGS_PATH, load_cached_data
        )
        X_tab_test = self.tabular_processor.process(df_test)

        test_preds = np.zeros(len(df_test))

        # 2. Aggregate Predictions from all Folds
        for fold in range(Config.N_SPLITS):
            # Load artifacts
            pls = joblib.load(os.path.join(self.models_dir, f"pls_fold_{fold}.joblib"))
            pls_scaler = joblib.load(
                os.path.join(self.models_dir, f"pls_scaler_fold_{fold}.joblib")
            )
            tab_scaler = joblib.load(
                os.path.join(self.models_dir, f"tab_scaler_fold_{fold}.joblib")
            )
            clf = joblib.load(os.path.join(self.models_dir, f"clf_fold_{fold}.joblib"))

            # Transform
            X_text_p = pls_scaler.transform(pls.transform(X_text_test))
            X_tab_s = tab_scaler.transform(X_tab_test)
            X_final = np.hstack([X_text_p, X_tab_s])

            # Predict
            test_preds += clf.predict_proba(X_final)[:, 1]

        # Average
        test_preds /= Config.N_SPLITS

        # 3. Save Submission
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": test_preds,
            }
        )
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
