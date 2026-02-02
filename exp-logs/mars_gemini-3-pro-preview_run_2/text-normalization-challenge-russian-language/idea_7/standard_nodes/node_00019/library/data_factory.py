import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from sklearn.model_selection import KFold
from library.config import Config
from library.hfbb import HFBB
from library.utils import setup_logger, is_semiotic, set_seed


class DataFactory:
    """
    Handles data preparation for the Curriculum-Enriched Residual Hybrid Cascade.
    1. Trains BPE tokenizer on target text.
    2. Generates curriculum-enriched training data (Residuals + Anchors) via Jackknifing.
    3. Prepares validation data relevant to the Transformer.
    """

    def __init__(self):
        self.logger = setup_logger("DataFactory")
        set_seed(Config.SEED)

    def train_bpe_tokenizer(self, df_train: pd.DataFrame):
        """
        Trains a BPE tokenizer on the 'after' column of the training data.

        Args:
            df_train (pd.DataFrame): Training data containing 'after' column.
        """
        model_prefix = Config.BPE_MODEL_PREFIX
        # Check if model already exists
        if os.path.exists(model_prefix + ".model"):
            self.logger.info(f"BPE model already exists at {model_prefix}.model")
            return

        self.logger.info("Training BPE tokenizer...")

        # Create temp file for SentencePiece training
        temp_file = os.path.join(Config.WORKING_DIR, "temp_bpe_train.txt")

        # Extract unique target strings to optimize training speed
        text_data = df_train["after"].dropna().astype(str).unique()

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for line in text_data:
                    f.write(line + "\n")

            # Train BPE model
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=Config.TARGET_VOCAB_SIZE,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="[PAD]",
                unk_piece="[UNK]",
                bos_piece="[BOS]",
                eos_piece="[EOS]",
                user_defined_symbols=[],
            )
            self.logger.info(f"BPE tokenizer trained and saved to {model_prefix}")

        except Exception as e:
            self.logger.error(f"Failed to train BPE tokenizer: {e}")
            raise e
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def generate_curriculum_data(
        self, df_train: pd.DataFrame, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Generates the training dataset for the Transformer via Jackknifing.
        Combines Residuals (HFBB failures) with Anchors (HFBB successes on semiotic tokens).

        Args:
            df_train (pd.DataFrame): Full training dataset.
            load_cached_data (bool): Whether to load from Parquet cache if available.

        Returns:
            pd.DataFrame: The enriched training dataset.
        """
        output_path = Config.ENRICHED_TRAIN_PATH

        if load_cached_data and os.path.exists(output_path):
            self.logger.info(f"Loading enriched training data from {output_path}")
            return pd.read_parquet(output_path)

        self.logger.info("Generating curriculum-enriched data (Jackknifing)...")

        residuals_list = []
        anchors_list = []

        # Get unique sentences for group splitting to prevent leakage
        unique_sentences = df_train["sentence_id"].unique()
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

        fold = 1
        for train_idx, val_idx in kf.split(unique_sentences):
            self.logger.info(f"Processing Fold {fold}/{Config.N_FOLDS}...")

            train_sents = unique_sentences[train_idx]
            val_sents = unique_sentences[val_idx]

            # Filter data for current fold
            fold_train = df_train[df_train["sentence_id"].isin(train_sents)].copy()
            fold_val = df_train[df_train["sentence_id"].isin(val_sents)].copy()

            # Train HFBB on this fold's training data
            # IMPORTANT: load_cached_data=False to force retraining on this specific fold
            hfbb = HFBB()
            hfbb.fit(fold_train, load_cached_data=False)

            # Predict on this fold's validation data
            preds = hfbb.predict_batch(fold_val)

            # Prepare for analysis
            fold_val["pred_after"] = preds.fillna("__NaN__")
            fold_val["after"] = fold_val["after"].astype(str)

            # Identify Residuals: Prediction does not match Target
            is_residual = fold_val["pred_after"] != fold_val["after"]

            # Identify Anchors: Prediction matches Target AND is Semiotic
            # We preserve these to teach the model the grammar of numbers
            is_correct = ~is_residual
            is_sem = fold_val["before"].astype(str).apply(is_semiotic)
            is_anchor = is_correct & is_sem

            residuals_fold = fold_val[is_residual].copy()
            anchors_fold = fold_val[is_anchor].copy()

            residuals_list.append(residuals_fold)
            anchors_list.append(anchors_fold)

            self.logger.info(
                f"Fold {fold}: Residuals={len(residuals_fold)}, Anchors={len(anchors_fold)}"
            )
            fold += 1

        # Combine all folds
        all_residuals = pd.concat(residuals_list, ignore_index=True)
        all_anchors = pd.concat(anchors_list, ignore_index=True)

        self.logger.info(f"Total Residuals: {len(all_residuals)}")
        self.logger.info(f"Total Anchors Available: {len(all_anchors)}")

        # Sample Anchors based on ratio
        n_anchors = int(len(all_residuals) * Config.ANCHOR_RATIO)
        n_anchors = min(n_anchors, len(all_anchors))

        self.logger.info(f"Sampling {n_anchors} anchors to mix with residuals...")
        if n_anchors > 0:
            sampled_anchors = all_anchors.sample(n=n_anchors, random_state=Config.SEED)
            combined = pd.concat([all_residuals, sampled_anchors], ignore_index=True)
        else:
            combined = all_residuals

        # Class Balancing: Upsample rare classes (MONEY, DECIMAL) to match DATE
        self.logger.info("Performing Class Balancing...")
        class_counts = combined["class"].value_counts()

        # Determine target count (Frequency of DATE or max)
        target_count = 0
        if "DATE" in class_counts:
            target_count = class_counts["DATE"]
        elif len(class_counts) > 0:
            target_count = class_counts.max()

        if target_count > 0:
            upsample_classes = ["MONEY", "DECIMAL"]
            dfs_to_concat = [combined]

            for cls in upsample_classes:
                if cls in class_counts:
                    count = class_counts[cls]
                    if count < target_count:
                        n_add = target_count - count
                        cls_df = combined[combined["class"] == cls]
                        if not cls_df.empty:
                            upsampled = cls_df.sample(
                                n=n_add, replace=True, random_state=Config.SEED
                            )
                            dfs_to_concat.append(upsampled)
                            self.logger.info(f"Upsampled {cls}: +{n_add} samples")

            combined = pd.concat(dfs_to_concat, ignore_index=True)

        # Shuffle final dataset
        combined = combined.sample(frac=1, random_state=Config.SEED).reset_index(
            drop=True
        )

        self.logger.info(f"Final Enriched Dataset Size: {len(combined)}")
        combined.to_parquet(output_path, index=False)
        return combined

    def prepare_val_data(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Prepares validation data by training HFBB on full train set and identifying
        residuals/semiotic tokens in the validation set. This simulates the inference
        pipeline to create a relevant validation set for the Transformer.

        Args:
            df_train (pd.DataFrame): Full training dataset.
            df_val (pd.DataFrame): Validation dataset.
            load_cached_data (bool): Whether to load from Parquet cache.

        Returns:
            pd.DataFrame: Filtered validation dataset.
        """
        output_path = Config.ENRICHED_VAL_PATH

        if load_cached_data and os.path.exists(output_path):
            self.logger.info(f"Loading enriched validation data from {output_path}")
            return pd.read_parquet(output_path)

        self.logger.info(
            "Preparing validation data (Full HFBB Train -> Val Predict)..."
        )

        # Train HFBB on full training data
        # Force retraining to ensure the cache corresponds to the full dataset
        hfbb = HFBB()
        hfbb.fit(df_train, load_cached_data=False)

        # Predict on Validation
        preds = hfbb.predict_batch(df_val)

        df_val_proc = df_val.copy()
        df_val_proc["pred_after"] = preds.fillna("__NaN__")
        df_val_proc["after"] = df_val_proc["after"].astype(str)

        # Filter for Residuals OR Semiotic tokens
        # 1. Residuals: Where HFBB is wrong or missing (simulating failure)
        # 2. Semiotic: Where Gate would pass token to Transformer
        is_residual = df_val_proc["pred_after"] != df_val_proc["after"]
        is_sem = df_val_proc["before"].astype(str).apply(is_semiotic)

        mask = is_residual | is_sem
        df_val_filtered = df_val_proc[mask].copy()

        self.logger.info(
            f"Validation Set Filtered: {len(df_val_filtered)} / {len(df_val)} tokens relevant for Transformer."
        )

        df_val_filtered.to_parquet(output_path, index=False)
        return df_val_filtered
