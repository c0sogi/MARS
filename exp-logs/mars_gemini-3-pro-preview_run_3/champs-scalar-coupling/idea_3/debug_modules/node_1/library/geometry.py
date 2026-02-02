import torch
import torch.nn as nn
import numpy as np
from library.config import Config

# Attempt to import scipy for precise Bessel roots, with fallback
try:
    from scipy.special import jn_zeros

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class GaussianSmearing(nn.Module):
    """
    Expands scalar distances using a Gaussian Radial Basis Function (RBF) bank.
    Output: exp(-((d - mu) / sigma)^2)
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Widths are set based on the spacing between centers
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        self.register_buffer("offset", offset)
        self.register_buffer("widths", widths)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape [N] or [N, 1]
        Returns:
            Tensor of shape [N, num_gaussians]
        """
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)

        # dist: [N, 1], offset: [num_gaussians]
        # diff: [N, num_gaussians]
        diff = dist - self.offset
        return torch.exp(-torch.pow(diff / self.widths, 2))


class SphericalBasisLayer(nn.Module):
    """
    Expands distances and angles using Spherical Bessel Functions and Legendre Polynomials.
    Used for directional message passing to capture angular dependencies.
    """

    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent

        self.retrieve_roots()

    def retrieve_roots(self):
        """
        Computes or approximates the roots of spherical Bessel functions.
        z_{ln} is the n-th root of the l-th spherical Bessel function.
        """
        roots = []
        for l in range(self.num_spherical):
            if SCIPY_AVAILABLE:
                # jn_zeros(l, n) returns the first n positive zeros of j_l
                r = jn_zeros(l, self.num_radial)
            else:
                # Asymptotic approximation: z_{ln} approx pi * (n + l/2 + 1)
                # n is 1-based index
                n_idx = np.arange(1, self.num_radial + 1)
                r = np.pi * (n_idx + l / 2.0)
            roots.append(r)

        # Shape: [num_spherical, num_radial]
        roots = torch.tensor(np.array(roots), dtype=torch.float32)
        self.register_buffer("roots", roots)

    def envelope(self, dist):
        """
        Polynomial envelope function to ensure smooth cutoff at boundary.
        1 - 6x^5 + 15x^6 - 10x^7
        """
        u = dist / self.cutoff
        # Clamp u to [0, 1] to avoid instability outside cutoff (though dist should be < cutoff)
        u = torch.clamp(u, 0, 1)

        # Polynomial for p=5
        env_val = (
            1.0
            - 6.0 * torch.pow(u, 5)
            + 15.0 * torch.pow(u, 6)
            - 10.0 * torch.pow(u, 7)
        )
        return env_val

    def _bessel_j(self, l, x):
        """
        Recursive implementation of Spherical Bessel Functions j_l(x).
        """
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)

        # Handle small x singularity if necessary, though x > 0 usually
        # Using mask to prevent division by zero if x contains 0
        mask = x < 1e-6
        safe_x = torch.where(mask, torch.ones_like(x), x)

        if l == 0:
            val = sin_x / safe_x
            return torch.where(mask, torch.ones_like(x), val)

        if l == 1:
            val = sin_x / (safe_x**2) - cos_x / safe_x
            return torch.where(mask, torch.zeros_like(x), val)

        # Iterative recursion for l >= 2
        j_prev2 = torch.where(mask, torch.ones_like(x), sin_x / safe_x)  # l=0
        j_prev1 = torch.where(
            mask, torch.zeros_like(x), sin_x / (safe_x**2) - cos_x / safe_x
        )  # l=1

        j_curr = j_prev1
        for i in range(2, l + 1):
            j_new = ((2 * (i - 1) + 1) / safe_x) * j_prev1 - j_prev2
            j_prev2 = j_prev1
            j_prev1 = j_new
            j_curr = j_new

        return j_curr

    def _legendre_p(self, l, x):
        """
        Recursive implementation of Legendre Polynomials P_l(x).
        x is cos(theta).
        """
        if l == 0:
            return torch.ones_like(x)
        if l == 1:
            return x

        p_prev2 = torch.ones_like(x)  # l=0
        p_prev1 = x  # l=1

        p_curr = p_prev1
        for i in range(2, l + 1):
            # (l)P_l = (2l-1)x P_{l-1} - (l-1)P_{l-2}
            p_new = ((2 * i - 1) * x * p_prev1 - (i - 1) * p_prev2) / i
            p_prev2 = p_prev1
            p_prev1 = p_new
            p_curr = p_new

        return p_curr

    def forward(self, dist, angle, idx_kj=None):
        """
        Args:
            dist: Tensor [num_triplets] (Distance d_kj)
            angle: Tensor [num_triplets] (Angle theta_kji)
            idx_kj: Unused, kept for API compatibility if needed.
        Returns:
            Tensor [num_triplets, num_spherical * num_radial]
        """
        # 1. Radial Part: j_l(z * d / c)
        d_scaled = dist / self.cutoff
        rbf = []
        for l in range(self.num_spherical):
            # roots[l]: [num_radial]
            rt = self.roots[l]
            # arg: [num_triplets, num_radial]
            arg = d_scaled.unsqueeze(-1) * rt.unsqueeze(0)
            val = self._bessel_j(l, arg)
            rbf.append(val)

        # 2. Angular Part: P_l(cos theta)
        cos_angle = torch.cos(angle)
        cbf = []
        for l in range(self.num_spherical):
            val = self._legendre_p(l, cos_angle)
            cbf.append(val)

        # 3. Combine: Envelope * R(d) * A(theta)
        out = []
        envelope = self.envelope(dist).unsqueeze(-1)  # [num_triplets, 1]

        for l in range(self.num_spherical):
            # r: [N, num_radial], a: [N]
            r = rbf[l]
            a = cbf[l].unsqueeze(-1)

            # Product
            prod = r * a * envelope
            out.append(prod)

        # Concatenate all features
        return torch.cat(out, dim=-1)


def get_radius_graph(pos, cutoff):
    """
    Computes the radius graph (edges with dist < cutoff).

    Args:
        pos: [N, 3] coordinates
        cutoff: float
    Returns:
        edge_index: [2, E] LongTensor
        edge_dist: [E] FloatTensor
    """
    # Compute pairwise distances [N, N]
    dist_mat = torch.cdist(pos, pos)

    # Mask for valid edges: dist < cutoff AND dist > 0 (remove self-loops)
    mask = (dist_mat < cutoff) & (dist_mat > 1e-6)

    # Get indices [E, 2]
    edge_index = torch.nonzero(mask, as_tuple=False).t()

    # Get distances [E]
    edge_dist = dist_mat[edge_index[0], edge_index[1]]

    return edge_index, edge_dist


def get_triplets(edge_index, num_nodes):
    """
    Finds triplets k->j->i for directional message passing.

    Args:
        edge_index: [2, E]
        num_nodes: int
    Returns:
        triplets: [T, 2] LongTensor.
                  Column 0 is index of incoming edge (k->j).
                  Column 1 is index of outgoing edge (j->i).
    """
    src, dst = edge_index
    triplets = []

    # Iterate over nodes to find connections j
    for j in range(num_nodes):
        # Find edges pointing to j (k -> j)
        incoming = (dst == j).nonzero(as_tuple=False).view(-1)
        # Find edges pointing from j (j -> i)
        outgoing = (src == j).nonzero(as_tuple=False).view(-1)

        if len(incoming) == 0 or len(outgoing) == 0:
            continue

        # Get source nodes k and target nodes i
        k_indices = src[incoming]  # [num_in]
        i_indices = dst[outgoing]  # [num_out]

        # Create mask where k != i (avoid backtracking)
        # [num_in, num_out]
        mask = k_indices.unsqueeze(1) != i_indices.unsqueeze(0)

        # Get valid pairs indices
        valid_in, valid_out = mask.nonzero(as_tuple=True)

        # Map back to edge indices
        edge_idx_k_j = incoming[valid_in]
        edge_idx_j_i = outgoing[valid_out]

        triplets.append(torch.stack([edge_idx_k_j, edge_idx_j_i], dim=1))

    if len(triplets) > 0:
        triplets = torch.cat(triplets, dim=0)
    else:
        triplets = torch.zeros((0, 2), dtype=torch.long)

    return triplets


def compute_angles(pos, edge_index, triplets):
    """
    Computes angles for the triplets.

    Args:
        pos: [N, 3]
        edge_index: [2, E]
        triplets: [T, 2]
    Returns:
        theta: [T] Angle in radians
    """
    idx_kj = triplets[:, 0]
    idx_ji = triplets[:, 1]

    # Nodes
    k = edge_index[0, idx_kj]
    j = edge_index[1, idx_kj]
    i = edge_index[1, idx_ji]

    pos_k = pos[k]
    pos_j = pos[j]
    pos_i = pos[i]

    # Vectors pointing away from j
    v_ji = pos_i - pos_j
    v_jk = pos_k - pos_j

    # Dot product and norms
    dot = (v_ji * v_jk).sum(dim=-1)
    norm_ji = v_ji.norm(dim=-1)
    norm_jk = v_jk.norm(dim=-1)

    # Cosine theta
    cos_theta = dot / (norm_ji * norm_jk + 1e-9)
    # Clamp for numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = torch.acos(cos_theta)
    return theta
