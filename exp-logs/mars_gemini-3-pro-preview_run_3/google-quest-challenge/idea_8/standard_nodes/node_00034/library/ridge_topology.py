import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.model_core import BackboneWrapper
from library.data_loader import get_dataloaders, get_tokenizer
from library.feature_store import extract_and_save_features


class TopologyRidgeTrainer:
    """
    Trainer for the Topology-Aware Ridge Regression layer.
    Decouples Question and Answer targets and trains separate Ridge heads
    on top of frozen backbone embeddings.
    """

    def __init__(self, model_tag, fold_idx, base_model_name):
        """
        Args:
            model_tag (str): Identifier for the model (e.g., 'deberta', 'mpnet').
            fold_idx (int): Fold index.
            base_model_name (str): HuggingFace model name for tokenizer/config.
        """
        self.model_tag = model_tag
        self.fold_idx = fold_idx
        self.base_model_name = base_model_name
        self.device = torch.device(Config.DEVICE)

        # Define file paths for features
        self.path_train_q = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_train_features_Q.npy"
        )
        self.path_train_a = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_train_features_A.npy"
        )
        self.path_train_t = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_train_targets.npy"
        )

        self.path_val_q = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_features_Q.npy"
        )
        self.path_val_a = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_features_A.npy"
        )
        self.path_val_t = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_targets.npy"
        )
        self.path_val_ids = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_ids.npy"
        )

        self.path_test_q = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_features_Q.npy"
        )
        self.path_test_a = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_features_A.npy"
        )
        self.path_test_ids = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_ids.npy"
        )

        # Output paths for predictions
        self.path_oof_preds = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_oof_preds.npy"
        )
        self.path_test_preds = os.path.join(
            Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_preds.npy"
        )

    def _extract_train_features(self, checkpoint_path, debug=False):
        """
        Extracts features from the training set using the fine-tuned backbone.
        This is necessary because feature_store.py only handles Val/Test.
        """
        if (
            os.path.exists(self.path_train_q)
            and os.path.exists(self.path_train_a)
            and os.path.exists(self.path_train_t)
        ):
            print(f"[Ridge] Cached training features found for {self.model_tag}.")
            return

        print(f"[Ridge] Extracting training features for {self.model_tag}...")

        # Load Model
        tokenizer = get_tokenizer(self.base_model_name)
        model = BackboneWrapper(
            self.base_model_name, num_labels=len(Config.TARGET_COLS)
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Get Train Loader
        train_loader, _, _ = get_dataloaders(
            tokenizer=tokenizer,
            train_batch_size=Config.VALID_BATCH_SIZE,  # Use larger batch for inference
            valid_batch_size=Config.VALID_BATCH_SIZE,
            load_cached_data=True,
            debug=debug,
        )

        # Inference Loop
        h_q_list = []
        h_a_list = []
        h_cls_list = []
        targets_list = []

        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                q_mask = batch["q_mask"].to(self.device)
                a_mask = batch["a_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    q_mask=q_mask,
                    a_mask=a_mask,
                )

                features = outputs["features"]
                h_cls_list.append(features["h_cls"].cpu().numpy())
                h_q_list.append(features["h_q"].cpu().numpy())
                h_a_list.append(features["h_a"].cpu().numpy())
                targets_list.append(labels.cpu().numpy())

        # Concatenate
        h_cls = np.concatenate(h_cls_list, axis=0)
        h_q = np.concatenate(h_q_list, axis=0)
        h_a = np.concatenate(h_a_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        # Construct Topology Features
        # Q: Just h_q
        train_feat_q = h_q

        # A: [h_cls, h_q, h_a, |h_q - h_a|]
        h_diff = np.abs(h_q - h_a)
        train_feat_a = np.concatenate([h_cls, h_q, h_a, h_diff], axis=1)

        # Save
        np.save(self.path_train_q, train_feat_q)
        np.save(self.path_train_a, train_feat_a)
        np.save(self.path_train_t, targets)

        print(f"[Ridge] Saved training features to {Config.WORKING_DIR}")

        # Cleanup
        del model, h_cls, h_q, h_a, targets
        torch.cuda.empty_cache()

    def train_and_predict(self, checkpoint_path, debug=False, load_cached_data=True):
        """
        Main execution flow:
        1. Ensure all features are extracted.
        2. Train Question-Solver Ridge.
        3. Train Answer-Solver Ridge.
        4. Generate and save predictions.
        """
        seed_everything(Config.SEED)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Ensure Features Exist
        # Val/Test features via feature_store
        extract_and_save_features(
            base_model_name=self.base_model_name,
            checkpoint_path=checkpoint_path,
            fold_idx=self.fold_idx,
            model_tag=self.model_tag,
            debug=debug,
            load_cached_data=load_cached_data,
        )

        # Train features via internal method
        self._extract_train_features(checkpoint_path, debug=debug)

        # 2. Load Data
        print(f"[Ridge] Loading features for {self.model_tag}...")
        X_train_q = np.load(self.path_train_q)
        X_train_a = np.load(self.path_train_a)
        y_train = np.load(self.path_train_t)

        X_val_q = np.load(self.path_val_q)
        X_val_a = np.load(self.path_val_a)
        y_val = np.load(self.path_val_t)

        X_test_q = np.load(self.path_test_q)
        X_test_a = np.load(self.path_test_a)

        # 3. Identify Target Indices
        q_indices = [Config.TARGET_COLS.index(c) for c in Config.QUESTION_TARGETS]
        a_indices = [Config.TARGET_COLS.index(c) for c in Config.ANSWER_TARGETS]

        # 4. Train Question Solver
        print("[Ridge] Training Question Solver (Targets: 21, Features: h_Q)...")
        y_train_q = y_train[:, q_indices]

        ridge_q = MultiOutputRegressor(
            RidgeCV(alphas=Config.RIDGE_ALPHAS, scoring="neg_mean_squared_error")
        )
        ridge_q.fit(X_train_q, y_train_q)

        pred_val_q = ridge_q.predict(X_val_q)
        pred_test_q = ridge_q.predict(X_test_q)

        # 5. Train Answer Solver
        print("[Ridge] Training Answer Solver (Targets: 9, Features: Joint+Diff)...")
        y_train_a = y_train[:, a_indices]

        ridge_a = MultiOutputRegressor(
            RidgeCV(alphas=Config.RIDGE_ALPHAS, scoring="neg_mean_squared_error")
        )
        ridge_a.fit(X_train_a, y_train_a)

        pred_val_a = ridge_a.predict(X_val_a)
        pred_test_a = ridge_a.predict(X_test_a)

        # 6. Reassemble Predictions
        # Initialize empty arrays
        pred_val_full = np.zeros((len(X_val_q), 30))
        pred_test_full = np.zeros((len(X_test_q), 30))

        # Fill columns
        pred_val_full[:, q_indices] = pred_val_q
        pred_val_full[:, a_indices] = pred_val_a

        pred_test_full[:, q_indices] = pred_test_q
        pred_test_full[:, a_indices] = pred_test_a

        # Clip predictions to [0, 1]
        pred_val_full = np.clip(pred_val_full, 0, 1)
        pred_test_full = np.clip(pred_test_full, 0, 1)

        # 7. Evaluation
        score = compute_spearman_metric(y_val, pred_val_full)
        print(
            f"[Ridge] {self.model_tag} Fold {self.fold_idx} OOF Spearman: {score:.16f}"
        )

        # 8. Save Predictions
        np.save(self.path_oof_preds, pred_val_full)
        np.save(self.path_test_preds, pred_test_full)
        print(f"[Ridge] Saved predictions to {Config.WORKING_DIR}")

        return pred_test_full

    def generate_submission_file(self):
        """
        Generates a submission.csv from the current model's test predictions.
        Useful if running this model standalone.
        """
        if not os.path.exists(self.path_test_preds):
            print("No test predictions found. Run train_and_predict first.")
            return

        print("[Ridge] Generating submission file...")
        preds = np.load(self.path_test_preds)
        test_ids = np.load(self.path_test_ids)

        # Create DataFrame
        sub_df = pd.DataFrame(preds, columns=Config.TARGET_COLS)
        sub_df.insert(0, "qa_id", test_ids)

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"[Ridge] Submission saved to {Config.SUBMISSION_PATH}")
