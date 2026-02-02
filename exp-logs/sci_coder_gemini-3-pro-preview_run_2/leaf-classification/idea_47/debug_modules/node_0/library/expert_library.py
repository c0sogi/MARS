from library.config import Config
from library.pipeline_factory import PipelineFactory


class ExpertDefinition:
    """
    Encapsulates the definition of a single expert model within the DMGE.

    Attributes:
        name (str): Unique identifier for the expert.
        view_name (str): The key for the data view required ('X_global' or 'X_morph').
        builder_func (callable): The static method from PipelineFactory to build the pipeline.
        builder_kwargs (dict): Arguments to pass to the builder_func.
    """

    def __init__(self, name, view_name, builder_func, builder_kwargs=None):
        self.name = name
        self.view_name = view_name
        self.builder_func = builder_func
        self.builder_kwargs = builder_kwargs if builder_kwargs is not None else {}

    def build_pipeline(self):
        """
        Instantiates the scikit-learn pipeline for this expert.

        Returns:
            sklearn.pipeline.Pipeline: The constructed pipeline.
        """
        return self.builder_func(**self.builder_kwargs)

    def __repr__(self):
        return f"ExpertDefinition(name='{self.name}', view='{self.view_name}')"


def get_expert_pool():
    """
    Generates the library of candidate experts based on the DMGE topologies.

    Topologies:
    - A: Marginal Statistical Anchors (Global View, Fixed Shrinkage)
    - B: Rotational Statistical Experts (Global View, Fixed Shrinkage)
    - C: Discriminative-Interaction Experts (Global View, Fixed Shrinkage)
    - D: Polynomial Physical Experts (Morphometric View, Auto Shrinkage)
    - E: Robust Distributional Experts (Global View, Auto Shrinkage)

    Returns:
        list[ExpertDefinition]: A list of expert definitions ready for training/selection.
    """
    experts = []
    shrinkage_candidates = Config.TOPOLOGY_A_SHRINKAGE_CANDIDATES

    # ---------------------------------------------------------
    # Topology A: Marginal Statistical Anchors
    # ---------------------------------------------------------
    for shrinkage in shrinkage_candidates:
        experts.append(
            ExpertDefinition(
                name=f"Topology_A_Marginal_Shrinkage_{shrinkage}",
                view_name="X_global",
                builder_func=PipelineFactory.build_topology_a,
                builder_kwargs={"shrinkage": shrinkage},
            )
        )

    # ---------------------------------------------------------
    # Topology B: Rotational Statistical Experts
    # ---------------------------------------------------------
    for shrinkage in shrinkage_candidates:
        experts.append(
            ExpertDefinition(
                name=f"Topology_B_Rotational_Shrinkage_{shrinkage}",
                view_name="X_global",
                builder_func=PipelineFactory.build_topology_b,
                builder_kwargs={"shrinkage": shrinkage},
            )
        )

    # ---------------------------------------------------------
    # Topology C: Discriminative-Interaction Experts
    # ---------------------------------------------------------
    for shrinkage in shrinkage_candidates:
        experts.append(
            ExpertDefinition(
                name=f"Topology_C_Interaction_Shrinkage_{shrinkage}",
                view_name="X_global",
                builder_func=PipelineFactory.build_topology_c,
                builder_kwargs={"shrinkage": shrinkage},
            )
        )

    # ---------------------------------------------------------
    # Topology D: Polynomial Physical Experts
    # ---------------------------------------------------------
    # Uses 'auto' shrinkage (Ledoit-Wolf) internally in the builder
    experts.append(
        ExpertDefinition(
            name="Topology_D_Physical_Poly",
            view_name="X_morph",
            builder_func=PipelineFactory.build_topology_d,
            builder_kwargs={},
        )
    )

    # ---------------------------------------------------------
    # Topology E: Robust Distributional Experts
    # ---------------------------------------------------------
    # Uses 'auto' shrinkage internally in the builder
    experts.append(
        ExpertDefinition(
            name="Topology_E_Robust_Quantile",
            view_name="X_global",
            builder_func=PipelineFactory.build_topology_e,
            builder_kwargs={},
        )
    )

    return experts
