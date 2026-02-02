import os
import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_log_loss, save_submission
from library.data_manager import DataLoader
from library.expert_models import get_lda_expert, get_lr_expert
from library.ensemble_selector import GreedyEnsembleSelector

# Set global seeds for reproducibility
np.random.seed(Config.RANDOM_SEED)


def run():
    print("Initializing DataLoader...")
    loader = DataLoader()

    # =========================================================================
    # PHASE 1: SELECTION
    # =========================================================================
    print("\n=== Phase 1: Expert Selection ===")

    # Load Phase 1 Data (Train/Val Split)
    # This handles feature extraction, caching, and Gaussianization
    data_p1 = loader.get_phase1_data()

    X_anc_tr = data_p1["train"]["anchor"]
    X_ort_tr = data_p1["train"]["orthogonal"]
    X_syn_tr = data_p1["train"]["synergistic"]
    y_tr = data_p1["train"]["y"]

    X_anc_val = data_p1["val"]["anchor"]
    X_ort_val = data_p1["val"]["orthogonal"]
    X_syn_val = data_p1["val"]["synergistic"]
    y_val = data_p1["val"]["y"]
    classes = data_p1["classes"]

    # Train Experts
    print("Training experts on training split...")

    # 1. Anchor (LDA)
    model_anchor = get_lda_expert()
    model_anchor.fit(X_anc_tr, y_tr)

    # 2. Orthogonal (LDA)
    model_ortho = get_lda_expert()
    model_ortho.fit(X_ort_tr, y_tr)

    # 3. Synergistic (LDA)
    model_syn = get_lda_expert()
    model_syn.fit(X_syn_tr, y_tr)

    # 4. Synergistic Variants (Ablation)
    # Split Orthogonal Transformed Features: Hu Moments (0-6) vs Scalars (7-10)
    # This allows the selector to isolate noise from specific feature groups (Cite Lesson 00056)
    X_hu_tr = X_ort_tr[:, :7]
    X_sca_tr = X_ort_tr[:, 7:]

    X_syn_hu_tr = np.hstack([X_anc_tr, X_hu_tr])
    X_syn_sca_tr = np.hstack([X_anc_tr, X_sca_tr])

    model_syn_hu = get_lda_expert()
    model_syn_hu.fit(X_syn_hu_tr, y_tr)

    model_syn_sca = get_lda_expert()
    model_syn_sca.fit(X_syn_sca_tr, y_tr)

    # Generate Validation Predictions
    print("Generating validation predictions...")
    preds_anchor = model_anchor.predict_proba(X_anc_val).astype(np.float64)
    preds_ortho = model_ortho.predict_proba(X_ort_val).astype(np.float64)
    preds_syn = model_syn.predict_proba(X_syn_val).astype(np.float64)

    # Prepare Validation Data for Variants
    X_hu_val = X_ort_val[:, :7]
    X_sca_val = X_ort_val[:, 7:]
    X_syn_hu_val = np.hstack([X_anc_val, X_hu_val])
    X_syn_sca_val = np.hstack([X_anc_val, X_sca_val])

    preds_syn_hu = model_syn_hu.predict_proba(X_syn_hu_val).astype(np.float64)
    preds_syn_sca = model_syn_sca.predict_proba(X_syn_sca_val).astype(np.float64)

    predictions_dict = {
        "Anchor": preds_anchor,
        "Orthogonal": preds_ortho,
        "Synergistic": preds_syn,
        "Synergistic_Hu": preds_syn_hu,
        "Synergistic_Scalars": preds_syn_sca,
    }

    # Run Greedy Selection
    selector = GreedyEnsembleSelector()
    all_labels = np.arange(len(classes))
    weights = selector.fit(predictions_dict, y_val, labels=all_labels)

    print("\nSelected Ensemble Weights:")
    for expert, weight in weights.items():
        print(f"  {expert}: {weight}")

    # Calculate Final Validation Metric
    # Reconstruct the ensemble prediction using the weights
    final_val_preds = np.zeros_like(preds_anchor)
    total_weight = 0

    for expert, weight in weights.items():
        final_val_preds += predictions_dict[expert] * weight
        total_weight += weight

    if total_weight > 0:
        final_val_preds /= total_weight
    else:
        # Fallback if nothing selected (unlikely), use Anchor
        final_val_preds = preds_anchor
        weights = {"Anchor": 1}
        total_weight = 1

    val_loss = calculate_log_loss(y_val, final_val_preds, labels=all_labels)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # =========================================================================
    # FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Failure Analysis ===")
    # Calculate per-sample log loss
    epsilon = 1e-15
    preds_clipped = np.clip(final_val_preds, epsilon, 1 - epsilon)

    # Gather probabilities of true classes
    # y_val is shape (n_samples,)
    true_class_probs = preds_clipped[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # Correlate with features (using Anchor features for analysis)
    # Calculate mean magnitude and variance of features per sample
    feat_mean = np.mean(np.abs(X_anc_val), axis=1)
    feat_std = np.std(X_anc_val, axis=1)

    corr_mean = np.corrcoef(sample_losses, feat_mean)[0, 1]
    corr_std = np.corrcoef(sample_losses, feat_std)[0, 1]

    print(f"Correlation between Error and Feature Magnitude (Mean): {corr_mean:.4f}")
    print(f"Correlation between Error and Feature Variance (Std): {corr_std:.4f}")

    # =========================================================================
    # PHASE 2: SUBMISSION
    # =========================================================================
    # Threshold set to the previous best validation score.
    # We must achieve a lower score to justify a new submission.
    THRESHOLD = 4.3016180315260426e-13

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Get Phase 2 Data (Full Train + Test)
        data_p2 = loader.get_phase2_data()

        X_anc_full = data_p2["train"]["anchor"]
        X_ort_full = data_p2["train"]["orthogonal"]
        X_syn_full = data_p2["train"]["synergistic"]
        y_full = data_p2["train"]["y"]

        X_anc_test = data_p2["test"]["anchor"]
        X_ort_test = data_p2["test"]["orthogonal"]
        X_syn_test = data_p2["test"]["synergistic"]
        ids_test = data_p2["test"]["ids"]

        # Retrain Selected Experts
        test_preds_sum = np.zeros((len(ids_test), len(classes)), dtype=np.float64)

        print("Retraining selected experts on full dataset...")

        if "Anchor" in weights:
            print(f"  Retraining Anchor (x{weights['Anchor']})...")
            model = get_lda_expert()
            model.fit(X_anc_full, y_full)
            p = model.predict_proba(X_anc_test).astype(np.float64)
            test_preds_sum += p * weights["Anchor"]

        if "Orthogonal" in weights:
            print(f"  Retraining Orthogonal (x{weights['Orthogonal']})...")
            model = get_lda_expert()
            model.fit(X_ort_full, y_full)
            p = model.predict_proba(X_ort_test).astype(np.float64)
            test_preds_sum += p * weights["Orthogonal"]

        if "Synergistic" in weights:
            print(f"  Retraining Synergistic (x{weights['Synergistic']})...")
            model = get_lda_expert()
            model.fit(X_syn_full, y_full)
            p = model.predict_proba(X_syn_test).astype(np.float64)
            test_preds_sum += p * weights["Synergistic"]

        if "Synergistic_Hu" in weights:
            print(f"  Retraining Synergistic_Hu (x{weights['Synergistic_Hu']})...")
            # Prepare Full/Test Data
            X_hu_full = X_ort_full[:, :7]
            X_syn_hu_full = np.hstack([X_anc_full, X_hu_full])

            X_hu_test = X_ort_test[:, :7]
            X_syn_hu_test = np.hstack([X_anc_test, X_hu_test])

            model = get_lda_expert()
            model.fit(X_syn_hu_full, y_full)
            p = model.predict_proba(X_syn_hu_test).astype(np.float64)
            test_preds_sum += p * weights["Synergistic_Hu"]

        if "Synergistic_Scalars" in weights:
            print(
                f"  Retraining Synergistic_Scalars (x{weights['Synergistic_Scalars']})..."
            )
            # Prepare Full/Test Data
            X_sca_full = X_ort_full[:, 7:]
            X_syn_sca_full = np.hstack([X_anc_full, X_sca_full])

            X_sca_test = X_ort_test[:, 7:]
            X_syn_sca_test = np.hstack([X_anc_test, X_sca_test])

            model = get_lda_expert()
            model.fit(X_syn_sca_full, y_full)
            p = model.predict_proba(X_syn_sca_test).astype(np.float64)
            test_preds_sum += p * weights["Synergistic_Scalars"]

        # Average
        test_preds_avg = test_preds_sum / total_weight

        # Save
        save_submission(ids_test, test_preds_avg, classes)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation metric ({val_loss}) is NOT lower than threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
