import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, calculate_log_loss, format_submission
from library.mlm_engine import run_mlm_pretraining
from library.stat_engine import StatisticalTrainer
from library.neural_engine import NeuralTrainer
from library.ensemble import EnsembleOptimizer
from library.dataset import load_text_data
from library.features import StylometricExtractor


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("=== Starting Author Identification Pipeline ===")

    # 2. MLM Pre-training (Domain Adaptation)
    print("\n--- Step 1: MLM Pre-training ---")
    # This will skip training if cached models exist
    run_mlm_pretraining(load_cached_data=True)

    # 3. Load Data
    print("\n--- Loading Data ---")
    # We need raw text for Neural models and manual feature extraction for Stage 2 Stat
    train_texts, train_labels, val_texts, val_labels, test_texts, test_ids = (
        load_text_data(load_cached_data=True)
    )

    # 4. Stage 1: Training & CV
    print("\n--- Step 2: Stage 1 Training (Cross-Validation) ---")

    # 4.1 Statistical Model
    print("Training Statistical Model (Stage 1)...")
    stat_trainer = StatisticalTrainer()
    # We let it load its own features from cache (get_data)
    oof_stat, test_stat, y_train_sorted_stat, _ = stat_trainer.run_cv(
        load_cached_data=True
    )

    # 4.2 Neural Models
    neural_oofs = []
    neural_tests = []
    neural_model_names = []

    for backbone in Config.MODEL_BACKBONES:
        print(f"Training Neural Model: {backbone} (Stage 1)...")
        trainer = NeuralTrainer(backbone)
        # run_cv loads data internally, combines Train+Val, and returns OOFs aligned with that combination
        oof_neural, test_neural, y_train_sorted_neural, _ = trainer.run_cv(
            load_cached_data=True
        )

        neural_oofs.append(oof_neural)
        neural_tests.append(test_neural)
        neural_model_names.append(backbone)

    # 5. Stage 1 Ensemble & Pseudo-Labeling
    print("\n--- Step 3: Stage 1 Ensemble & Pseudo-Labeling ---")

    # Check alignment of labels between trainers
    if not np.array_equal(y_train_sorted_stat, y_train_sorted_neural):
        print(
            "Warning: Label alignment mismatch between Statistical and Neural trainers."
        )

    all_oofs = [oof_stat] + neural_oofs
    all_tests = [test_stat] + neural_tests
    model_names = ["Statistical"] + neural_model_names

    optimizer = EnsembleOptimizer(model_names)
    stage1_weights, stage1_score = optimizer.optimize(all_oofs, y_train_sorted_stat)

    stage1_test_preds = optimizer.predict(all_tests)

    # Pseudo-Labeling Logic
    max_probs = np.max(stage1_test_preds, axis=1)
    pseudo_mask = max_probs > Config.PSEUDO_LABEL_THRESHOLD
    pseudo_count = np.sum(pseudo_mask)
    print(f"Pseudo-Labeling: Found {pseudo_count} high-confidence test samples.")

    # Initialize variables for final evaluation
    final_val_preds = None
    final_test_preds = None
    final_val_score = None

    if pseudo_count > 0 and Config.USE_PSEUDO_LABELING:
        pseudo_texts = test_texts[pseudo_mask]
        pseudo_labels = np.argmax(stage1_test_preds[pseudo_mask], axis=1)

        # Augment Training Data (Original Train + Pseudo Test)
        # We keep Original Val separate for Stage 2 monitoring
        aug_train_texts = np.concatenate([train_texts, pseudo_texts])
        aug_train_labels = np.concatenate([train_labels, pseudo_labels])

        print(f"Augmented Training Set Size: {len(aug_train_texts)}")

        # 6. Stage 2: Retraining
        print("\n--- Step 4: Stage 2 Retraining (Student Models) ---")

        # 6.1 Stage 2 Statistical
        print("Retraining Statistical Model (Stage 2)...")
        # Feature Generation for Augmented Data

        # Word TF-IDF
        word_vec = TfidfVectorizer(
            ngram_range=Config.TFIDF_PARAMS["ngram_range_word"],
            max_features=Config.TFIDF_PARAMS["max_features_word"],
            sublinear_tf=Config.TFIDF_PARAMS["sublinear_tf"],
            analyzer="word",
            token_pattern=r"\w{1,}",
        )
        # Char TF-IDF
        char_vec = TfidfVectorizer(
            ngram_range=Config.TFIDF_PARAMS["ngram_range_char"],
            max_features=Config.TFIDF_PARAMS["max_features_char"],
            sublinear_tf=Config.TFIDF_PARAMS["sublinear_tf"],
            analyzer="char",
        )

        # Fit on Augmented Train
        X_aug_word = word_vec.fit_transform(aug_train_texts)
        X_aug_char = char_vec.fit_transform(aug_train_texts)

        # Transform Val and Test
        X_val_word = word_vec.transform(val_texts)
        X_val_char = char_vec.transform(val_texts)
        X_test_word = word_vec.transform(test_texts)
        X_test_char = char_vec.transform(test_texts)

        # Stylometric
        stylo = StylometricExtractor()
        X_aug_stylo = stylo.transform(aug_train_texts)
        X_val_stylo = stylo.transform(val_texts)
        X_test_stylo = stylo.transform(test_texts)

        scaler = StandardScaler()
        X_aug_stylo = scaler.fit_transform(X_aug_stylo)
        X_val_stylo = scaler.transform(X_val_stylo)
        X_test_stylo = scaler.transform(X_test_stylo)

        # Stack
        X_aug_final = scipy.sparse.hstack(
            [X_aug_word, X_aug_char, scipy.sparse.csr_matrix(X_aug_stylo)]
        ).tocsr()
        X_val_final = scipy.sparse.hstack(
            [X_val_word, X_val_char, scipy.sparse.csr_matrix(X_val_stylo)]
        ).tocsr()
        X_test_final = scipy.sparse.hstack(
            [X_test_word, X_test_char, scipy.sparse.csr_matrix(X_test_stylo)]
        ).tocsr()

        # Train
        val_preds_stat_s2, test_preds_stat_s2, _ = stat_trainer.train_on_augmented(
            X_aug_final,
            aug_train_labels,
            X_val_final,
            val_labels,
            X_test_final,
            test_ids,
        )

        # 6.2 Stage 2 Neural
        val_preds_neural_s2 = []
        test_preds_neural_s2 = []

        for backbone in Config.MODEL_BACKBONES:
            print(f"Retraining Neural Model: {backbone} (Stage 2)...")
            trainer = NeuralTrainer(backbone)
            val_p, test_p, _ = trainer.train_on_augmented(
                aug_train_texts,
                aug_train_labels,
                val_texts,
                val_labels,
                test_texts,
                test_ids,
            )
            val_preds_neural_s2.append(val_p)
            test_preds_neural_s2.append(test_p)

        # 7. Final Ensemble (Stage 2)
        print("\n--- Step 5: Final Ensemble Optimization ---")

        # Optimize weights based on the Original Validation Set performance of Stage 2 models
        s2_val_preds_list = [val_preds_stat_s2] + val_preds_neural_s2
        s2_test_preds_list = [test_preds_stat_s2] + test_preds_neural_s2

        optimizer_s2 = EnsembleOptimizer(model_names)
        final_weights, final_val_score = optimizer_s2.optimize(
            s2_val_preds_list, val_labels
        )

        final_test_preds = optimizer_s2.predict(s2_test_preds_list)
        final_val_preds = optimizer_s2.predict(s2_val_preds_list)

    else:
        print(
            "Skipping Stage 2 (No pseudo-labels found or disabled). Using Stage 1 models."
        )
        # Fallback to Stage 1 OOFs for Validation metric
        # Extract Val part from OOFs (OOFs are aligned with concat(Train, Val))
        val_start_idx = len(train_texts)

        blended_oof = optimizer.predict(all_oofs)
        final_val_preds = blended_oof[val_start_idx:]

        # Recalculate score on just this part
        final_val_score = calculate_log_loss(val_labels, final_val_preds)
        final_test_preds = stage1_test_preds

    print(f"Final Validation Metric: {final_val_score}")

    # 8. Failure Analysis
    print("\n--- Step 6: Failure Analysis ---")
    if final_val_preds is not None:
        # Calculate Cross Entropy per sample
        epsilon = 1e-15
        preds_clipped = np.clip(final_val_preds, epsilon, 1 - epsilon)

        # Gather probabilities of true classes
        rows = np.arange(len(val_labels))
        true_probs = preds_clipped[rows, val_labels]

        # Cross Entropy
        ce_loss = -np.log(true_probs)

        # Feature: Text Length
        text_lengths = np.array([len(t) for t in val_texts])

        # Correlation
        corr, _ = pearsonr(ce_loss, text_lengths)
        print(f"Correlation between Error (Cross Entropy) and Text Length: {corr:.10f}")

    # 9. Submission
    print("\n--- Step 7: Submission ---")
    threshold = 0.2435629959371868
    if final_val_score < threshold:
        print(
            f"Validation score ({final_val_score:.6f}) meets threshold ({threshold}). Generating submission..."
        )
        columns = [Config.ID2LABEL[i] for i in range(Config.NUM_CLASSES)]
        format_submission(
            test_ids,
            final_test_preds,
            columns,
            os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
        )
    else:
        print(
            f"Validation score ({final_val_score:.6f}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
