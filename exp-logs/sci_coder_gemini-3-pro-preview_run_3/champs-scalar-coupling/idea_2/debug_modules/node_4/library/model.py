import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import (
    BesselBasisLayer,
    SphericalBasisLayer,
    EmbeddingBlock,
    InteractionBlock,
    OutputBlock,
)


class DirectionalMPNN(nn.Module):
    """
    Directional Message Passing Neural Network for Scalar Coupling Prediction.

    This model explicitly captures angular dependencies by performing message passing
    on edges (directed atom pairs) and incorporating triplet geometric information
    (angles) via Spherical Basis Functions.
    """

    def __init__(
        self,
        hidden_channels=Config.HIDDEN_CHANNELS,
        num_layers=Config.NUM_LAYERS,
        num_radial=Config.NUM_RBF,
        num_spherical=Config.NUM_SBF,
        cutoff=Config.CUTOFF,
        envelope_exponent=Config.ENVELOPE_EXPONENT,
        num_output_layers=Config.NUM_OUTPUT_LAYERS,
        out_emb_dim=Config.TYPE_EMBEDDING_DIM,
    ):
        super(DirectionalMPNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        # 1. Geometric Basis Layers
        self.rbf_layer = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )

        self.sbf_layer = SphericalBasisLayer(
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        # 2. Initial Embedding Block
        self.emb_block = EmbeddingBlock(
            num_radial=num_radial, hidden_channels=hidden_channels
        )

        # 3. Interaction Blocks (Message Passing)
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_bilinear=hidden_channels,  # Assuming bilinear dim equals hidden
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output Block
        self.output_block = OutputBlock(
            hidden_channels=hidden_channels,
            num_radial=num_radial,
            out_emb_dim=out_emb_dim,
            num_layers=num_output_layers,
        )

    def _calculate_geometry(self, pos, edge_index, idx_kj, idx_ji):
        """
        Calculates distances for edges and angles for triplets.

        Args:
            pos: Atom coordinates (N, 3)
            edge_index: Graph connectivity (2, E)
            idx_kj: Indices of source edges in triplets (T, )
            idx_ji: Indices of target edges in triplets (T, )

        Returns:
            dist: Euclidean distances of edges (E, )
            angle: Angles of triplets (T, ) in radians
        """
        # Calculate distances for all edges j->i
        j, i = edge_index
        dist_vec = pos[i] - pos[j]
        dist = dist_vec.norm(dim=-1)

        # Calculate angles for triplets k->j->i
        # We need vectors for edges k->j and j->i
        # idx_kj points to edge k->j
        # idx_ji points to edge j->i

        # Vector r_ji (j -> i)
        vec_ji = dist_vec[idx_ji]

        # Vector r_jk (j -> k)
        # Note: dist_vec contains r_target_source.
        # edge_index[idx_kj] gives (k, j). dist_vec[idx_kj] is pos[j] - pos[k].
        # We want vector from j to k, which is -(pos[j] - pos[k]) = pos[k] - pos[j].
        # However, standard definition usually uses vectors emanating from central atom j.
        # u = pos[k] - pos[j]
        # v = pos[i] - pos[j]
        # dist_vec stores (pos[dst] - pos[src]).
        # For edge k->j: src=k, dst=j. dist_vec is pos[j] - pos[k].
        # So vector j->k is -dist_vec[idx_kj].

        vec_jk = -dist_vec[idx_kj]

        # Normalize vectors
        # Add epsilon to avoid division by zero
        norm_ji = vec_ji.norm(dim=-1, keepdim=True) + 1e-7
        norm_jk = vec_jk.norm(dim=-1, keepdim=True) + 1e-7

        u = vec_ji / norm_ji
        v = vec_jk / norm_jk

        # Cosine angle
        cosine = (u * v).sum(dim=-1)
        # Clamp for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        angle = torch.acos(cosine)

        return dist, angle

    def forward(
        self,
        z,
        pos,
        edge_index,
        idx_kj,
        idx_ji,
        target_node_0,
        target_node_1,
        target_type,
        target_edge_index_uv=None,
        target_edge_index_vu=None,
    ):
        """
        Forward pass of the model.

        Args:
            z: Atom types (N, )
            pos: Atom coordinates (N, 3)
            edge_index: Graph connectivity (2, E)
            idx_kj: Indices of source edges in triplets (T, )
            idx_ji: Indices of target edges in triplets (T, )
            target_node_0: Indices of first atom in target pairs (B, )
            target_node_1: Indices of second atom in target pairs (B, )
            target_type: Coupling type indices (B, )
            target_edge_index_uv: Indices in edge_index for edge u->v (B, ).
                                  If None, assumes all targets are covered or handled externally.
                                  Values < 0 indicate edge not in graph.
            target_edge_index_vu: Indices in edge_index for edge v->u (B, ).

        Returns:
            out: Predicted scalar coupling constants (B, 1)
        """
        # 1. Calculate Geometry
        dist, angle = self._calculate_geometry(pos, edge_index, idx_kj, idx_ji)

        # 2. Basis Expansion
        rbf = self.rbf_layer(dist)  # (E, Nr)
        sbf = self.sbf_layer(dist, angle, idx_kj)  # (T, Nr*Ns)

        # 3. Initial Embedding
        # edge_index[0] is source (j), edge_index[1] is target (i)
        x = self.emb_block(z, rbf, edge_index[0], edge_index[1])  # (E, H)

        # 4. Interaction Blocks
        for block in self.interaction_blocks:
            x = block(x, rbf, sbf, idx_kj, idx_ji)

        # 5. Output Prediction
        # We need to gather the embeddings for the specific target pairs.

        # A. Calculate exact distance for target pairs (independent of graph cutoff)
        # This ensures we use the precise distance for the final prediction head
        diff_target = pos[target_node_0] - pos[target_node_1]
        dist_target = diff_target.norm(dim=-1)
        rbf_target = self.rbf_layer(dist_target)  # (B, Nr)

        # B. Gather Edge Embeddings
        # We need x_uv (u->v) and x_vu (v->u)
        # If the edge exists in the graph, we pull it. If not (due to cutoff), we use zeros.

        batch_size = target_node_0.size(0)
        x_uv = torch.zeros(
            batch_size, self.hidden_channels, device=x.device, dtype=x.dtype
        )
        x_vu = torch.zeros(
            batch_size, self.hidden_channels, device=x.device, dtype=x.dtype
        )

        if target_edge_index_uv is not None and target_edge_index_vu is not None:
            # Mask for valid edges (index >= 0)
            mask_uv = target_edge_index_uv >= 0
            mask_vu = target_edge_index_vu >= 0

            if mask_uv.any():
                valid_indices_uv = target_edge_index_uv[mask_uv]
                x_uv[mask_uv] = x[valid_indices_uv]

            if mask_vu.any():
                valid_indices_vu = target_edge_index_vu[mask_vu]
                x_vu[mask_vu] = x[valid_indices_vu]
        else:
            # If indices are not provided, we cannot efficiently map without a search.
            # In a real pipeline, these indices should be precomputed.
            # For robustness, we leave them as zeros if not provided,
            # effectively predicting based only on distance and type for missing graph edges.
            pass

        # 6. Final Regression
        out = self.output_block(x_uv, x_vu, rbf_target, target_type)

        return out
