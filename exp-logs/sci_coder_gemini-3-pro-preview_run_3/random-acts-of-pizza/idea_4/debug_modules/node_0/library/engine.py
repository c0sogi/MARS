import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, save_predictions, load_data, get_cached_data
from library.feature_engineering import FeatureEngineer
from library.dataset import _compute_transformer_data, PizzaDataset
from library.models import (
    get_lexical_model,
    get_style_model,
    get_meta_model,
    SemanticFineTuner,
)


class CrossValidationRunner:
    """
    Orchestrates the Tri-View Stacking Ensemble training and evaluation pipeline.
    """

    def __init__(self, n_folds=Config.N_FOLDS, debug=Config.DEBUG):
        self.n_folds = n_folds
        self.debug = debug
        self.fe = FeatureEngineer()
        set_seed(Config.SEED)

    def _get_transformer_data(self):
        """
        Retrieves tokenized data, utilizing caching to avoid re-tokenization.
        """
        # Load raw dataframes required for tokenization
        train_df = load_data(Config.TRAIN_PATH, debug=self.debug)
        val_df = load_data(Config.VAL_PATH, debug=self.debug)
        test_df = load_data(Config.TEST_PATH, debug=self.debug)

        suffix = "_debug" if self.debug else ""
        cache_name = f"transformer_data{suffix}"

        return get_cached_data(
            _compute_transformer_data,
            cache_name,
            load_cached_data=True,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    def run(self):
        print("Initializing Cross-Validation Runner...")

        # -------------------------------------------------------------------
        # 1. Data Loading & Preparation
        # -------------------------------------------------------------------
        # Load engineered features (Lexical, Style, Meta)
        data = self.fe.process_data(load_cached_data=True, debug=self.debug)

        # Load tokenized data for Transformer
        trans_data = self._get_transformer_data()

        # Merge Train and Val splits into a single 'Dev' set for CV
        # This maximizes data usage for the stacking process

        # Lexical View
        X_lex_dev = np.concatenate(
            [data["train"]["lexical"], data["val"]["lexical"]], axis=0
        )
        X_lex_test = data["test"]["lexical"]

        # Style View
        X_style_dev = np.concatenate(
            [data["train"]["style"], data["val"]["style"]], axis=0
        )
        X_style_test = data["test"]["style"]

        # Metadata
        X_meta_dev = np.concatenate(
            [data["train"]["meta"], data["val"]["meta"]], axis=0
        )
        X_meta_test = data["test"]["meta"]

        # Targets
        y_dev = np.concatenate([data["train"]["y"], data["val"]["y"]], axis=0)

        # Transformer Inputs
        input_ids_dev = np.concatenate(
            [trans_data["train_input_ids"], trans_data["val_input_ids"]], axis=0
        )
        masks_dev = np.concatenate(
            [trans_data["train_attention_mask"], trans_data["val_attention_mask"]],
            axis=0,
        )

        input_ids_test = trans_data["test_input_ids"]
        masks_test = trans_data["test_attention_mask"]

        # Concatenate Features for Level 1 Models
        # Lexical Model: TF-IDF + Metadata
        X_lexical_full_dev = np.hstack([X_lex_dev, X_meta_dev])
        X_lexical_full_test = np.hstack([X_lex_test, X_meta_test])

        # Style Model: Style Features + Metadata
        X_style_full_dev = np.hstack([X_style_dev, X_meta_dev])
        X_style_full_test = np.hstack([X_style_test, X_meta_test])

        # -------------------------------------------------------------------
        # 2. Cross-Validation Loop (Level 1 Training)
        # -------------------------------------------------------------------
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        # Placeholders for Out-Of-Fold predictions
        oof_preds = {
            "lexical": np.zeros(len(y_dev)),
            "style": np.zeros(len(y_dev)),
            "semantic": np.zeros(len(y_dev)),
        }

        print(
            f"Starting {self.n_folds}-Fold Cross-Validation on {len(y_dev)} samples..."
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_lexical_full_dev, y_dev)
        ):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")

            # --- A. Lexical Branch (Random Forest) ---
            print("Training Lexical Model (RF)...")
            rf = get_lexical_model()
            rf.fit(X_lexical_full_dev[train_idx], y_dev[train_idx])
            oof_preds["lexical"][val_idx] = rf.predict_proba(
                X_lexical_full_dev[val_idx]
            )[:, 1]

            # --- B. Style Branch (XGBoost) ---
            print("Training Style Model (XGB)...")
            xgb = get_style_model()
            xgb.fit(X_style_full_dev[train_idx], y_dev[train_idx])
            oof_preds["style"][val_idx] = xgb.predict_proba(X_style_full_dev[val_idx])[
                :, 1
            ]

            # --- C. Semantic Branch (DistilBERT) ---
            print("Training Semantic Model (DistilBERT)...")
            train_ds = PizzaDataset(
                input_ids_dev[train_idx], masks_dev[train_idx], y_dev[train_idx]
            )
            val_ds = PizzaDataset(
                input_ids_dev[val_idx], masks_dev[val_idx], y_dev[val_idx]
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            bert = SemanticFineTuner()
            bert.fit(train_loader, val_loader)
            oof_preds["semantic"][val_idx] = bert.predict_proba(val_loader)

        # -------------------------------------------------------------------
        # 3. OOF Evaluation
        # -------------------------------------------------------------------
        print("\n--- Level 1 OOF Performance ---")
        for name, preds in oof_preds.items():
            auc = roc_auc_score(y_dev, preds)
            print(f"{name.capitalize()} OOF AUC: {auc}")

        # -------------------------------------------------------------------
        # 4. Meta-Learner Training (Level 2)
        # -------------------------------------------------------------------
        print("\nTraining Meta-Learner (Logistic Regression) on OOF Predictions...")
        X_oof = np.column_stack(
            [oof_preds["lexical"], oof_preds["style"], oof_preds["semantic"]]
        )

        meta_model = get_meta_model()
        meta_model.fit(X_oof, y_dev)

        # Check Meta-Learner Performance on OOF (sanity check)
        meta_oof_preds = meta_model.predict_proba(X_oof)[:, 1]
        meta_auc = roc_auc_score(y_dev, meta_oof_preds)
        print(f"Meta-Learner OOF AUC: {meta_auc}")

        # -------------------------------------------------------------------
        # 5. Full Retraining & Test Prediction
        # -------------------------------------------------------------------
        print("\nRetraining Level 1 Models on Full Dev Set and Predicting Test Set...")

        # --- A. Lexical Full ---
        rf_full = get_lexical_model()
        rf_full.fit(X_lexical_full_dev, y_dev)
        test_pred_lex = rf_full.predict_proba(X_lexical_full_test)[:, 1]

        # --- B. Style Full ---
        xgb_full = get_style_model()
        xgb_full.fit(X_style_full_dev, y_dev)
        test_pred_style = xgb_full.predict_proba(X_style_full_test)[:, 1]

        # --- C. Semantic Full ---
        # For the deep learning model, we need a validation set for Early Stopping.
        # We split the full dev set 90/10.
        t_idx, v_idx = train_test_split(
            np.arange(len(y_dev)),
            test_size=0.1,
            stratify=y_dev,
            random_state=Config.SEED,
        )

        train_ds_full = PizzaDataset(
            input_ids_dev[t_idx], masks_dev[t_idx], y_dev[t_idx]
        )
        val_ds_full = PizzaDataset(input_ids_dev[v_idx], masks_dev[v_idx], y_dev[v_idx])
        test_ds_full = PizzaDataset(input_ids_test, masks_test, labels=None)

        train_loader_full = DataLoader(
            train_ds_full,
            batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader_full = DataLoader(
            val_ds_full,
            batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        test_loader_full = DataLoader(
            test_ds_full,
            batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        bert_full = SemanticFineTuner()
        bert_full.fit(train_loader_full, val_loader_full)
        test_pred_sem = bert_full.predict_proba(test_loader_full)

        # -------------------------------------------------------------------
        # 6. Final Prediction Generation
        # -------------------------------------------------------------------
        print("Generating Final Predictions...")
        X_test_meta = np.column_stack([test_pred_lex, test_pred_style, test_pred_sem])
        final_preds = meta_model.predict_proba(X_test_meta)[:, 1]

        save_predictions(data["test"]["ids"], final_preds)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
