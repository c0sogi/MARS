import torch
import torch.nn as nn
import library.config as config


class PyramidalBlock(nn.Module):
    """
    Residual Block for the Pyramidal Backbone.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate=0.2):
        super(PyramidalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class NRPIRVNet(nn.Module):
    """
    Noise-Robust Pyramidal Invariant Residual-Visual Network (NR-PIRV-Net).

    Dual-stream architecture:
    1. Kinematic Stream: Pyramidal backbone processing wide tracking features.
    2. Visual Stream: Shallow MLP processing helmet bounding box metrics.
    3. Fusion: Additive residual connection of logits.
    """

    def __init__(self, kin_input_dim, vis_input_dim):
        """
        Args:
            kin_input_dim (int): Dimension of the flattened kinematic feature vector.
            vis_input_dim (int): Dimension of the flattened visual feature vector.
        """
        super(NRPIRVNet, self).__init__()

        # --- Kinematic Stream ---
        # Structure: Input -> [Project -> ResBlock] x N -> Logit
        # Uses an Interleaved Pyramidal Backbone
        self.kinematic_layers = nn.ModuleList()
        current_dim = kin_input_dim

        for hidden_dim in config.PYRAMID_DIMS:
            # Projection Layer: Resizes dimension -> BN -> ReLU
            projection = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            self.kinematic_layers.append(projection)

            # Residual Block: Operates at the projected dimension
            res_block = PyramidalBlock(hidden_dim, dropout_rate=config.DROPOUT_RATE)
            self.kinematic_layers.append(res_block)

            current_dim = hidden_dim

        # Final Kinematic Head (Scalar Logit)
        self.kin_head = nn.Linear(current_dim, 1)

        # --- Visual Stream ---
        # Structure: Input -> Linear -> ReLU -> Linear -> Logit
        # Shallow MLP to prevent overfitting to noisy visual proxies
        self.visual_net = nn.Sequential(
            nn.Linear(vis_input_dim, config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.VISUAL_HIDDEN_DIM, 1),
        )

    def forward(self, kinematic_x, visual_x):
        """
        Args:
            kinematic_x (torch.Tensor): Flattened kinematic features.
            visual_x (torch.Tensor): Flattened visual features.

        Returns:
            torch.Tensor: Combined logits (kinematic_logit + visual_logit).
        """
        # --- Kinematic Stream Forward ---
        k = kinematic_x
        for layer in self.kinematic_layers:
            k = layer(k)
        kin_logit = self.kin_head(k)

        # --- Visual Stream Forward ---
        vis_logit = self.visual_net(visual_x)

        # --- Residual Fusion ---
        # Additive fusion of logits: Logit_final = L_kin + lambda * L_vis
        # Here lambda is implicitly 1.0 as per standard residual connections
        return kin_logit + vis_logit
