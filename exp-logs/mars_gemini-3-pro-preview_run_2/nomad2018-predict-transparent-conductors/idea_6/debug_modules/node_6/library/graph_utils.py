import torch
import numpy as np
from ase import io
from ase.neighborlist import neighbor_list
from library.config import Config


def load_structure(file_path):
    """
    Loads an atomic structure from a file using ASE.

    Args:
        file_path (str): Path to the geometry file (e.g., .xyz).

    Returns:
        ase.Atoms: The loaded atomic structure.
    """
    try:
        # Explicitly specify 'aims' format because the file extension .xyz is misleading.
        # The content follows FHI-aims format (lattice_vector, atom keywords).
        # Cite debug_lesson_1
        atoms = io.read(file_path, format="aims")
        return atoms
    except Exception as e:
        print(f"Error loading structure from {file_path}: {e}")
        return None


def get_pbc_neighbor_graph(atoms, cutoff=Config.CUTOFF_RADIUS):
    """
    Constructs a graph from an ASE Atoms object using a cutoff radius under PBC.

    Args:
        atoms (ase.Atoms): The atomic structure.
        cutoff (float): The cutoff radius for neighbor search.

    Returns:
        tuple: (edge_index, edge_distances, atom_numbers)
            - edge_index (torch.LongTensor): Shape (2, num_edges), source and target indices.
            - edge_distances (torch.Tensor): Shape (num_edges, 1), distances between connected atoms.
            - atom_numbers (torch.LongTensor): Shape (num_atoms,), atomic numbers of the nodes.
    """
    # Use ASE's neighbor_list to find neighbors within cutoff
    # 'i': source index, 'j': target index, 'd': distance
    # self_interaction=False excludes the atom interacting with itself at the same image,
    # but allows interaction with itself in periodic images if within cutoff.
    i_indices, j_indices, d_values = neighbor_list(
        "ijd", atoms, cutoff, self_interaction=False
    )

    # Convert to numpy arrays first (ASE returns numpy arrays or lists)
    i_indices = np.array(i_indices)
    j_indices = np.array(j_indices)
    d_values = np.array(d_values)

    # Create edge_index tensor (2, E)
    # Ensure type is Long for indices
    edge_index = torch.tensor(np.vstack((i_indices, j_indices)), dtype=torch.long)

    # Create edge_distances tensor (E, 1)
    # Ensure type is Float
    edge_distances = torch.tensor(d_values, dtype=torch.float32).unsqueeze(1)

    # Get atomic numbers for node features
    atom_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    return edge_index, edge_distances, atom_numbers


def get_global_features(atoms):
    """
    Extracts global features from the atomic structure.
    Features: 3 Lattice Lengths, 3 Lattice Angles, 4 Atomic Fractions (Al, Ga, In, O).

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        torch.Tensor: Shape (10,), the global feature vector.
    """
    # 1. Lattice Parameters
    # cell_lengths_and_angles returns (a, b, c, alpha, beta, gamma)
    cell_params = atoms.get_cell_lengths_and_angles()
    lattice_features = torch.tensor(cell_params, dtype=torch.float32)

    # 2. Atomic Fractions
    # Elements of interest: Al (13), Ga (31), In (49), O (8)
    atomic_nums = atoms.get_atomic_numbers()
    total_atoms = len(atomic_nums)

    # Count occurrences
    # We can use numpy for speed
    counts = {
        13: np.sum(atomic_nums == 13),
        31: np.sum(atomic_nums == 31),
        49: np.sum(atomic_nums == 49),
        8: np.sum(atomic_nums == 8),
    }

    # Calculate fractions
    # Order: Al, Ga, In, O (arbitrary but must be consistent)
    # Config comment says: "4 Atomic Fractions (Al, Ga, In, O)"
    fractions = [
        counts[13] / total_atoms,
        counts[31] / total_atoms,
        counts[49] / total_atoms,
        counts[8] / total_atoms,
    ]

    composition_features = torch.tensor(fractions, dtype=torch.float32)

    # Concatenate
    global_features = torch.cat([lattice_features, composition_features], dim=0)

    return global_features
