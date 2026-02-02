import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


class ExpertFactory:
    """
    Factory class to instantiate Linear Discriminant Analysis (LDA) experts
    configured with specific solvers and shrinkage parameters as defined
    in the CIPGE strategy.
    """

    @staticmethod
    def create_model(view_code, shrinkage):
        """
        Creates a single LDA model instance based on the view configuration
        and the specific shrinkage parameter.

        Args:
            view_code (str): The view identifier ('A', 'B', 'C', 'D').
            shrinkage (float or str): The shrinkage parameter (e.g., 0.01 or 'auto').

        Returns:
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The configured model.
        """
        # Retrieve view specification from config
        if view_code not in config.VIEW_SPECS:
            raise ValueError(f"Invalid view_code: {view_code}")

        spec = config.VIEW_SPECS[view_code]
        solver = spec["lda_solver"]

        # Validate solver/shrinkage compatibility
        # 'svd' solver does not support shrinkage, but our config uses 'lsqr' or 'eigen'
        if solver == "svd" and shrinkage is not None:
            raise ValueError(
                f"Solver 'svd' does not support shrinkage (View {view_code})."
            )

        # Instantiate the model
        # Note: We do not set priors explicitly; they will be estimated from class proportions.
        # store_covariance is False by default.
        # We rely on scikit-learn to handle the float64 precision provided by the data loader.
        model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

        return model


def get_view_experts(view_code):
    """
    Generates a library of expert models for a specific view by iterating
    through the defined shrinkage options in config.

    Args:
        view_code (str): The view identifier.

    Returns:
        list of dict: A list where each dictionary contains:
            - 'model': The instantiated sklearn model.
            - 'shrinkage': The shrinkage value used.
            - 'view': The view code.
            - 'name': A unique name for the expert.
    """
    if view_code not in config.VIEW_SPECS:
        raise ValueError(f"Unknown view code: {view_code}")

    spec = config.VIEW_SPECS[view_code]
    shrinkage_options = spec["shrinkage_options"]

    experts = []

    for s in shrinkage_options:
        # Create the model using the factory
        model = ExpertFactory.create_model(view_code, s)

        # Create a descriptive name for tracking/logging
        s_str = str(s) if s != "auto" else "Auto"
        name = f"{spec['name']}_Shrinkage_{s_str}"

        experts.append(
            {"model": model, "shrinkage": s, "view": view_code, "name": name}
        )

    return experts
