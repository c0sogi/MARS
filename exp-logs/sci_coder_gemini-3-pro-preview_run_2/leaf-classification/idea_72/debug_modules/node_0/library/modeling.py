import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from library.config import RANDOM_SEED, FLOAT_PRECISION, N_CLASSES, SUBMISSION_PATH

# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================


class StratifiedManifoldTransformer(BaseEstimator, TransformerMixin):
    """
    Implements the Stratified Rotational Anchor logic.
    Splits the global feature vector into Margin, Shape, and Texture subsets.
    Applies PowerTransformer -> PCA -> PowerTransformer to each independently
    to align manifolds without one view dominating the variance.
    Concatenates the results.
    """

    def __init__(self):
        # Feature indices based on dataset description (64 features per group)
        self.idx_margin = slice(0, 64)
        self.idx_shape = slice(64, 128)
        self.idx_texture = slice(128, 192)

        # Independent pipelines for each view
        self.pipe_margin = self._make_sub_pipeline()
        self.pipe_shape = self._make_sub_pipeline()
        self.pipe_texture = self._make_sub_pipeline()

    def _make_sub_pipeline(self):
        return Pipeline(
            [
                ("pt1", PowerTransformer(method="yeo-johnson")),
                ("pca", PCA(whiten=False, random_state=RANDOM_SEED)),
                ("pt2", PowerTransformer(method="yeo-johnson")),
            ]
        )

    def fit(self, X, y=None):
        # Expects Global View (192 features)
        X_m = X[:, self.idx_margin]
        X_s = X[:, self.idx_shape]
        X_t = X[:, self.idx_texture]

        self.pipe_margin.fit(X_m)
        self.pipe_shape.fit(X_s)
        self.pipe_texture.fit(X_t)
        return self

    def transform(self, X):
        X_m = self.pipe_margin.transform(X[:, self.idx_margin])
        X_s = self.pipe_shape.transform(X[:, self.idx_shape])
        X_t = self.pipe_texture.transform(X[:, self.idx_texture])
        return np.hstack([X_m, X_s, X_t])


# =============================================================================
# EXPERT LIBRARY FACTORY
# =============================================================================


def get_expert_library():
    """
    Constructs the library of 12 probabilistic experts across 3 tiers.
    Returns a list of dictionaries containing expert configuration.
    """
    experts = []

    # --- Tier 1: Linear Manifold Anchors ---

    # Expert A: Global Marginal Anchor (LDA with OAS)
    experts.append(
        {
            "name": "A_Global_OAS",
            "view": "global",
            "pipeline": Pipeline(
                [
                    ("pt", PowerTransformer()),
                    (
                        "lda",
                        LinearDiscriminantAnalysis(
                            solver="lsqr", covariance_estimator=OAS()
                        ),
                    ),
                ]
            ),
        }
    )

    # Expert B: Stratified Rotational Anchor
    experts.append(
        {
            "name": "B_Stratified_Rotational",
            "view": "global",
            "pipeline": Pipeline(
                [
                    ("strat_manifold", StratifiedManifoldTransformer()),
                    (
                        "lda",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            ),
        }
    )

    # --- Tier 2: Physical Domain Experts ---

    # Expert C: Physical QDA (Morphometrics)
    # Using two regularization levels
    for reg in [0.1, 0.5]:
        experts.append(
            {
                "name": f"C_Physical_QDA_{reg}",
                "view": "morph",
                "pipeline": Pipeline(
                    [
                        ("pt", PowerTransformer()),
                        ("qda", QuadraticDiscriminantAnalysis(reg_param=reg)),
                    ]
                ),
            }
        )

    # --- Tier 3: Full-Rank Interaction Experts ---

    # Experts D, E, F: Intra-Component Polynomial LDAs
    # Views: Margin, Shape, Texture
    # Shrinkage: 0.1, 0.5
    views = [("margin", "D"), ("shape", "E"), ("texture", "F")]
    shrinkages = [0.1, 0.5]

    for view_key, expert_prefix in views:
        for shrink in shrinkages:
            experts.append(
                {
                    "name": f"{expert_prefix}_{view_key.capitalize()}_PolyLDA_{shrink}",
                    "view": view_key,
                    "pipeline": Pipeline(
                        [
                            ("pt1", PowerTransformer()),
                            ("poly", PolynomialFeatures(degree=2)),
                            ("pt2", PowerTransformer()),
                            (
                                "lda",
                                LinearDiscriminantAnalysis(
                                    solver="lsqr", shrinkage=shrink
                                ),
                            ),
                        ]
                    ),
                }
            )

    # Expert G: Global High-Variance Polynomial LDA
    # Uses PCA(0.99) to compress before expansion
    for shrink in shrinkages:
        experts.append(
            {
                "name": f"G_Global_PolyLDA_{shrink}",
                "view": "global",
                "pipeline": Pipeline(
                    [
                        ("pt1", PowerTransformer()),
                        ("pca", PCA(n_components=0.99, random_state=RANDOM_SEED)),
                        ("poly", PolynomialFeatures(degree=2)),
                        ("pt2", PowerTransformer()),
                        (
                            "lda",
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink),
                        ),
                    ]
                ),
            }
        )

    return experts


# =============================================================================
# ENSEMBLE TRAINER
# =============================================================================


class SMFRIE_Trainer:
    """
    Manages the training, selection, and inference lifecycle of the SM-FRIE strategy.
    """

    def __init__(self):
        self.library = get_expert_library()
        self.label_encoder = LabelEncoder()
        self.selected_experts = {}  # name -> weight
        self.classes_ = None

    def fit_and_select(self, data_train, data_val):
        """
        Phase 1: Train all experts on Train, Predict on Val, run Greedy Selection.
        """
        print("\n=== Phase 1: Expert Training & Selection ===")

        # 1. Prepare Targets
        y_train = self.label_encoder.fit_transform(data_train["y"])
        y_val = self.label_encoder.transform(data_val["y"])
        self.classes_ = self.label_encoder.classes_

        # 2. Train and Predict with all experts
        val_predictions = {}

        print(f"Training {len(self.library)} experts on {len(y_train)} samples...")

        for expert in self.library:
            name = expert["name"]
            view = expert["view"]
            pipeline = expert["pipeline"]

            # Get specific view data
            X_train_view = data_train[view]
            X_val_view = data_val[view]

            # Train
            try:
                pipeline.fit(X_train_view, y_train)

                # Predict Probabilities (clip to avoid log errors)
                probs = pipeline.predict_proba(X_val_view)
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                val_predictions[name] = probs

                # Individual Score
                score = log_loss(y_val, probs)
                print(f"  [{name}] Val LogLoss: {score:.5f}")

            except Exception as e:
                print(f"  [{name}] Failed: {e}")
                val_predictions[name] = None

        # 3. Greedy Forward Selection
        print("\nRunning Greedy Forward Selection...")
        self.selected_experts = self._greedy_selection(val_predictions, y_val)

        print("\nSelected Ensemble:")
        for name, weight in self.selected_experts.items():
            print(f"  - {name}: Weight {weight}")

    def _greedy_selection(self, predictions, y_true, iterations=50):
        """
        Iteratively adds the expert that maximizes validation log loss improvement.
        Allows replacement (weighting).
        """
        # Filter out failed experts
        valid_experts = [k for k, v in predictions.items() if v is not None]

        current_ensemble_sum = np.zeros(
            (len(y_true), len(self.classes_)), dtype=FLOAT_PRECISION
        )
        ensemble_counts = 0
        best_log_loss = float("inf")
        selected = []

        # Start with best single expert
        best_single_expert = None
        for name in valid_experts:
            score = log_loss(y_true, predictions[name])
            if score < best_log_loss:
                best_log_loss = score
                best_single_expert = name

        if best_single_expert:
            selected.append(best_single_expert)
            current_ensemble_sum += predictions[best_single_expert]
            ensemble_counts += 1
            print(f"  Iter 1: Added {best_single_expert} (Score: {best_log_loss:.5f})")

        # Hill Climbing
        for i in range(iterations - 1):
            best_iter_expert = None
            best_iter_score = best_log_loss

            for name in valid_experts:
                # Try adding this expert
                temp_sum = current_ensemble_sum + predictions[name]
                temp_prob = temp_sum / (ensemble_counts + 1)
                score = log_loss(y_true, temp_prob)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_expert = name

            if best_iter_expert:
                selected.append(best_iter_expert)
                current_ensemble_sum += predictions[best_iter_expert]
                ensemble_counts += 1
                best_log_loss = best_iter_score
                print(
                    f"  Iter {i+2}: Added {best_iter_expert} (Score: {best_log_loss:.5f})"
                )
            else:
                print("  No improvement found. Stopping selection.")
                break

        # Calculate weights
        weights = {}
        total = len(selected)
        for name in selected:
            weights[name] = weights.get(name, 0) + (1.0 / total)

        return weights

    def retrain_and_predict(self, data_train, data_val, data_test):
        """
        Phase 2: Combine Train+Val, Retrain selected experts, Predict Test.
        """
        print("\n=== Phase 2: Final Retraining & Prediction ===")

        # 1. Combine Data
        combined_data = {}
        views = ["global", "margin", "shape", "texture", "morph"]

        for v in views:
            combined_data[v] = np.vstack([data_train[v], data_val[v]])

        y_combined = np.concatenate([data_train["y"], data_val["y"]])
        y_combined_enc = self.label_encoder.transform(y_combined)

        # 2. Retrain Selected Experts and Accumulate Test Predictions
        n_test = data_test["global"].shape[0]
        n_classes = len(self.classes_)
        final_proba = np.zeros((n_test, n_classes), dtype=FLOAT_PRECISION)

        print(f"Retraining on {len(y_combined)} samples...")

        for expert in self.library:
            name = expert["name"]
            if name not in self.selected_experts:
                continue

            weight = self.selected_experts[name]
            view = expert["view"]
            pipeline = expert["pipeline"]

            print(f"  Retraining {name} (Weight: {weight:.4f})...")

            # Retrain
            pipeline.fit(combined_data[view], y_combined_enc)

            # Predict Test
            probs = pipeline.predict_proba(data_test[view])
            probs = np.clip(probs, 1e-15, 1 - 1e-15)

            # Weighted Add
            final_proba += probs * weight

        return final_proba

    def format_submission(self, probs, test_ids):
        """
        Formats the probability matrix into the submission DataFrame.
        """
        df_sub = pd.DataFrame(probs, columns=self.classes_)
        df_sub.insert(0, "id", test_ids)
        return df_sub


def train_and_predict(data_train, data_val, data_test):
    """
    Main entry point for the modeling module.
    """
    trainer = SMFRIE_Trainer()

    # Phase 1
    trainer.fit_and_select(data_train, data_val)

    # Phase 2
    final_probs = trainer.retrain_and_predict(data_train, data_val, data_test)

    # Format
    submission_df = trainer.format_submission(final_probs, data_test["ids"])

    # Save
    print(f"\nSaving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    return submission_df
