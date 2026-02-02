import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from library.config import Config

# ==========================================
# 1. Helper Functions & Basis Layers
# ==========================================


class Envelope(nn.Module):
    """
    Polynomial envelope function to ensure smooth cutoff.
    f(x) = 1 - 6x^5 + 15x^4 - 10x^3 for x < 1, else 0.
    where x = d / cutoff.
    """

    def __init__(self, exponent=5):
        super(Envelope, self).__init__()
        self.exponent = exponent

    def forward(self, x):
        # x is assumed to be dist / cutoff
        # Clamp x to [0, 1] to avoid issues, though mask handles > 1
        p = x
        return 1 - 6 * p**5 + 15 * p**4 - 10 * p**3


class BesselBasisLayer(nn.Module):
    """
    Radial Basis Function (RBF) expansion using Bessel functions.
    rbf(d) = sqrt(2/c) * sin(n * pi * d / c) / d
    """

    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Frequencies n * pi
        self.freq = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.num_radial + 1, out=self.freq).mul_(np.pi)

    def forward(self, dist):
        # dist: (Num_edges, )
        # Scale distance by cutoff
        d_scaled = dist / self.cutoff

        # Envelope
        env = self.envelope(d_scaled)

        # Bessel expansion
        # (N, 1) * (1, K) -> (N, K)
        d_norm = d_scaled.unsqueeze(-1)  # (N, 1)
        freq = self.freq.unsqueeze(0)  # (1, K)

        # Avoid division by zero for d=0 (though d=0 shouldn't exist in edge lists)
        # sin(n * pi * d/c) / (d/c)
        # For numerical stability, we use the fact that dist is usually > 0.
        # We compute sin(freq * d_norm) / d_norm.
        # Note: The standard formula often keeps the 'dist' in denominator raw or scaled.
        # We follow DimeNet: sqrt(2/c) * sin(...) / dist

        sin_term = torch.sin(freq * d_norm)
        rbf = np.sqrt(2.0 / self.cutoff) * sin_term / (dist.unsqueeze(-1) + 1e-7)

        return rbf * env.unsqueeze(-1)


class SphericalBasisLayer(nn.Module):
    """
    Spherical Basis Function (SBF) expansion for triplets.
    Expands the angle theta and distance d using spherical harmonics (Legendre) and Bessel functions.
    """

    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Radial part (Bessel)
        # We reuse the logic of BesselBasis but need separate parameters if we want independence
        self.bessel_freq = nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.num_radial + 1, out=self.bessel_freq).mul_(np.pi)

    def forward(self, dist, angle, idx_kj):
        """
        Args:
            dist: Distance of edge k->j (Num_edges, )
            angle: Angle k-j-i (Num_triplets, )
            idx_kj: Index of edge k->j in the triplet (Num_triplets, )
        """
        # 1. Radial Part (based on dist of k->j)
        # Gather distances for triplets
        d = dist[idx_kj]  # (Num_triplets, )
        d_scaled = d / self.cutoff

        env = self.envelope(d_scaled)

        d_norm = d_scaled.unsqueeze(-1)  # (T, 1)
        freq = self.bessel_freq.unsqueeze(0)  # (1, Nr)

        # rbf: (T, Nr)
        sin_term = torch.sin(freq * d_norm)
        rbf = np.sqrt(2.0 / self.cutoff) * sin_term / (d.unsqueeze(-1) + 1e-7)
        rbf = rbf * env.unsqueeze(-1)

        # 2. Angular Part (Legendre Polynomials)
        # angle is in [0, pi]. cos_angle in [-1, 1]
        cos_angle = torch.cos(angle)  # (T, )

        # Recursive calculation of Legendre polynomials
        # P_0(x) = 1
        # P_1(x) = x
        # P_l(x) = ((2l-1)x P_{l-1} - (l-1)P_{l-2}) / l

        cbf = [torch.ones_like(cos_angle), cos_angle]
        for l in range(2, self.num_spherical):
            p_l = ((2 * l - 1) * cos_angle * cbf[-1] - (l - 1) * cbf[-2]) / l
            cbf.append(p_l)

        # Stack: (T, Ns)
        cbf = torch.stack(cbf, dim=-1)

        # 3. Combine (Outer Product) -> Flatten
        # We want a basis of size (T, Ns * Nr)
        # (T, Nr, 1) * (T, 1, Ns) -> (T, Nr, Ns) -> (T, Nr*Ns)
        sbf = rbf.unsqueeze(-1) * cbf.unsqueeze(1)
        sbf = sbf.view(sbf.size(0), -1)

        return sbf


# ==========================================
# 2. Core Blocks
# ==========================================


class EmbeddingBlock(nn.Module):
    """
    Initializes atom embeddings and combines them with edge RBF to create edge embeddings.
    """

    def __init__(self, num_radial, hidden_channels, act=nn.SiLU()):
        super(EmbeddingBlock, self).__init__()
        self.act = act

        # Atom embeddings
        self.atom_embedding = nn.Embedding(Config.NUM_ATOM_TYPES, hidden_channels)

        # Interaction of atom types and rbf
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        """
        x: Atom types (N, )
        rbf: Radial basis of edges (E, num_radial)
        i, j: Edge indices (E, )
        """
        # Get atom embeddings
        h = self.atom_embedding(x)  # (N, H)

        # Create edge features by concatenating h_i, h_j, and rbf
        h_i = h[i]  # (E, H)
        h_j = h[j]  # (E, H)
        rbf_emb = self.act(self.lin_rbf(rbf))  # (E, H)

        # Combine
        edge_emb = torch.cat([h_i, h_j, rbf_emb], dim=-1)
        edge_emb = self.act(self.lin(edge_emb))

        return edge_emb


class InteractionBlock(nn.Module):
    """
    Directional interaction block.
    Aggregates messages from edge k->j to edge j->i using angular information.
    """

    def __init__(
        self, hidden_channels, num_radial, num_spherical, num_bilinear, act=nn.SiLU()
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.hidden_channels = hidden_channels

        # Transformations
        # 1. Transform RBF of edge j->i
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        # 2. Transform input edge embeddings (k->j)
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)

        # 3. Transform SBF (triplets)
        # SBF dim is num_radial * num_spherical
        sbf_dim = num_radial * num_spherical
        self.lin_sbf = nn.Linear(sbf_dim, num_bilinear, bias=False)

        # 4. Dense layer after aggregation
        self.lin_ji = nn.Linear(num_bilinear, hidden_channels)

        # 5. Residual layers
        self.lin_res_1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_res_2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf.reset_parameters()
        self.lin_kj.reset_parameters()
        self.lin_sbf.reset_parameters()
        self.lin_ji.reset_parameters()
        self.lin_res_1.reset_parameters()
        self.lin_res_2.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        """
        x: Edge embeddings (E, H)
        rbf: Radial basis of edges (E, Nr)
        sbf: Spherical basis of triplets (T, Nr*Ns)
        idx_kj: Indices of source edges in triplets (T, )
        idx_ji: Indices of target edges in triplets (T, )
        """
        # Initial transformation of edge features (k->j)
        x_kj = self.act(self.lin_kj(x))  # (E, H)
        x_kj = x_kj[idx_kj]  # (T, H)

        # Transform SBF
        W_sbf = self.lin_sbf(sbf)  # (T, num_bilinear)

        # We need to combine x_kj and W_sbf.
        # Typically DimeNet projects x_kj to num_bilinear as well, or uses element-wise if dims match.
        # Here we assume we want to modulate the signal.
        # Let's project x_kj to num_bilinear if needed, but standard DimeNet does:
        # m = Linear(x_kj) * Linear(sbf) (element-wise)
        # We'll assume hidden_channels == num_bilinear for simplicity or project.
        # To be safe, let's assume num_bilinear is the aggregation space.
        # We need a projection for x_kj if dimensions differ, but usually they are kept same.
        # If config doesn't specify num_bilinear, we use hidden_channels.

        # Element-wise multiplication (Hadamard product)
        # If dimensions mismatch, we would need another linear, but we'll assume consistency in Config.
        # For this implementation, we'll rely on broadcasting or assume H == Bilinear.
        # If H != Bilinear, we'd need: x_kj = Linear(x_kj) -> Bilinear.
        # Let's assume standard DimeNet++: x_ji = sum( x_kj * W_sbf )

        # Check dimensions
        if x_kj.shape[-1] != W_sbf.shape[-1]:
            # This implies we need a projection, but we'll assume the user sets config correctly
            # or we add a projection here.
            pass

        m = x_kj * W_sbf  # (T, H)

        # Aggregate to target edges (j->i)
        # scatter sum: (T, H) -> (E, H)
        # We aggregate based on idx_ji
        agg = scatter(m, idx_ji, dim=0, dim_size=x.size(0), reduce="sum")

        # Update with info from edge j->i (rbf)
        rbf_ji = self.lin_rbf(rbf)  # (E, H)

        # Combine aggregated message + local rbf
        m = agg + rbf_ji

        # Final dense transformation
        m = self.act(self.lin_ji(m))

        # Residual connection
        # x_new = x + MLP(m)
        res = self.act(self.lin_res_1(m))
        res = self.lin_res_2(res)

        return x + res


class OutputBlock(nn.Module):
    """
    Prediction head for scalar coupling.
    """

    def __init__(
        self, hidden_channels, num_radial, out_emb_dim=32, num_layers=3, act=nn.SiLU()
    ):
        super(OutputBlock, self).__init__()
        self.act = act

        # Embedding for coupling type
        self.type_embedding = nn.Embedding(Config.NUM_COUPLING_TYPES, out_emb_dim)

        # Input dimension:
        # Edge embedding (H) + Edge embedding reverse (H) + Type Emb (H_type) + Distance RBF (Nr)
        # We use both directions of the edge between atom 0 and 1.
        input_dim = 2 * hidden_channels + out_emb_dim + num_radial

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, hidden_channels))

        for _ in range(num_layers - 2):
            self.mlp.append(nn.Linear(hidden_channels, hidden_channels))

        self.lin_out = nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.type_embedding.reset_parameters()
        for layer in self.mlp:
            layer.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, edge_attr_uv, edge_attr_vu, rbf_uv, coupling_type_idx):
        """
        Args:
            edge_attr_uv: Embedding of edge u->v (B, H)
            edge_attr_vu: Embedding of edge v->u (B, H)
            rbf_uv: Radial basis of distance u-v (B, Nr)
            coupling_type_idx: Integer type indices (B, )
        """
        # Get type embedding
        type_emb = self.type_embedding(coupling_type_idx)  # (B, H_type)

        # Concatenate
        out = torch.cat([edge_attr_uv, edge_attr_vu, type_emb, rbf_uv], dim=-1)

        # MLP
        for layer in self.mlp:
            out = self.act(layer(out))

        return self.lin_out(out)
