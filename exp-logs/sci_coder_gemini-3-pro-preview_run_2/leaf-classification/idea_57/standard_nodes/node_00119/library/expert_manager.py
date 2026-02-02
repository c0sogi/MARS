import numpy as np
from sklearn.base import clone
from library.pipeline_factory import (
    build_global_linear,
    build_rotational,
    build_robust,
    build_polynomial,
    build_bottleneck_interaction,
)
from library.utils import to_float64


class ExpertLibrary:
    def __init__(self):
        """
        Initializes the ExpertLibrary.
        Defines the configuration of all experts in the ensemble.
        """
        self.expert_configs = []
        self.fitted_experts = {}
        self._initialize_library()

    def _initialize_library(self):
        """
        Populates the expert_configs list with dictionaries defining each expert.
        Structure:
        {
            'name': str,
            'view': str,
            'pipeline': sklearn.pipeline.Pipeline
        }
        """
        # --- Group A: Global Linear Anchors (The Baseline) ---
        # View: global
        # Strategies: Marginal, Rotational, Robust
        # Shrinkage values: 'auto' (Ledoit-Wolf), 0.001, 0.01 (Fixed)

        shrinkage_levels = ["auto", 0.001, 0.01]

        for s in shrinkage_levels:
            s_name = str(s)

            # 1. Marginal: PowerTransformer -> LDA
            self.expert_configs.append(
                {
                    "name": f"global_marginal_shrinkage_{s_name}",
                    "view": "global",
                    "pipeline": build_global_linear(shrinkage=s),
                }
            )

            # 2. Rotational: PT -> PCA -> PT -> LDA
            self.expert_configs.append(
                {
                    "name": f"global_rotational_shrinkage_{s_name}",
                    "view": "global",
                    "pipeline": build_rotational(shrinkage=s, n_pca_components=None),
                }
            )

            # 3. Robust: QuantileTransformer -> LDA
            self.expert_configs.append(
                {
                    "name": f"global_robust_shrinkage_{s_name}",
                    "view": "global",
                    "pipeline": build_robust(shrinkage=s, n_quantiles=50),
                }
            )

        # --- Group B: Physical Polynomial Experts (The Domain Signal) ---
        # View: morpho (Polarity-Corrected Morphometrics)
        # Strategy: PT -> Poly(2) -> PT -> LDA

        self.expert_configs.append(
            {
                "name": "morpho_poly_deg2_auto",
                "view": "morpho",
                "pipeline": build_polynomial(shrinkage="auto", degree=2),
            }
        )

        # --- Group C: Intra-Component Interaction Experts (Local Complexity) ---
        # Views: margin, shape, texture
        # Strategy: PT -> LDA_Trans(n=10) -> Poly(2) -> PT -> LDA

        components = ["margin", "shape", "texture"]
        for comp in components:
            self.expert_configs.append(
                {
                    "name": f"{comp}_interaction_bottleneck10_deg2",
                    "view": comp,
                    "pipeline": build_bottleneck_interaction(
                        shrinkage="auto", n_bottleneck=10, degree=2
                    ),
                }
            )

        # --- Group D: Inter-Component Interaction Experts (Coupling Complexity) ---
        # Views: margin_shape, margin_texture, shape_texture
        # Strategy: PT -> LDA_Trans(n=15) -> Poly(2) -> PT -> LDA

        pairs = ["margin_shape", "margin_texture", "shape_texture"]
        for pair in pairs:
            self.expert_configs.append(
                {
                    "name": f"{pair}_interaction_bottleneck15_deg2",
                    "view": pair,
                    "pipeline": build_bottleneck_interaction(
                        shrinkage="auto", n_bottleneck=15, degree=2
                    ),
                }
            )

    def fit_all(self, data_train):
        """
        Fits all defined experts on the provided training data.

        Args:
            data_train (dict): Dictionary containing 'y' (labels) and 'views' (dict of feature arrays).
        """
        y_train = data_train["y"]
        views = data_train["views"]

        print(f"Fitting {len(self.expert_configs)} experts...")

        for config in self.expert_configs:
            name = config["name"]
            view_name = config["view"]
            pipeline_template = config["pipeline"]

            # Retrieve specific view data
            if view_name not in views:
                print(
                    f"Warning: View '{view_name}' not found for expert '{name}'. Skipping."
                )
                continue

            X = to_float64(views[view_name])

            # Clone pipeline to ensure a fresh start
            model = clone(pipeline_template)

            try:
                model.fit(X, y_train)
                self.fitted_experts[name] = {"model": model, "view": view_name}
            except Exception as e:
                print(f"Error fitting expert '{name}': {e}")

        print(f"Successfully fitted {len(self.fitted_experts)} experts.")

    def predict_all(self, data_test):
        """
        Generates probability predictions for all fitted experts.

        Args:
            data_test (dict): Dictionary containing 'views' (dict of feature arrays).

        Returns:
            dict: A dictionary where keys are expert names and values are probability matrices (float64).
        """
        views = data_test["views"]
        predictions = {}

        for name, expert_info in self.fitted_experts.items():
            model = expert_info["model"]
            view_name = expert_info["view"]

            if view_name not in views:
                print(
                    f"Warning: View '{view_name}' not found for expert '{name}'. Skipping prediction."
                )
                continue

            X_test = to_float64(views[view_name])

            try:
                # Predict probabilities
                probs = model.predict_proba(X_test)
                predictions[name] = to_float64(probs)
            except Exception as e:
                print(f"Error predicting with expert '{name}': {e}")

        return predictions

    def get_expert_names(self):
        """Returns a list of names of all configured experts."""
        return [cfg["name"] for cfg in self.expert_configs]
