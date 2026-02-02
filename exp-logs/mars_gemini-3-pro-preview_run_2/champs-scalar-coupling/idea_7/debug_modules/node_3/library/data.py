import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from library.config import Config
from library.features import RBFExpansion, SBFExpansion

# Mapping for coupling types to integers
COUPLING_TYPES = sorted(
    ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
)
TYPE_TO_ID = {t: i for i, t in enumerate(COUPLING_TYPES)}

# Atomic number mapping (H, C, N, O, F) - prevalent in this dataset
# We map them to 0-based indices for embedding layers
ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}


class MolecularGraphDataset(Dataset):
    def __init__(self, metadata_path, split_name, load_cached_data=True):
        """
        Args:
            metadata_path: Path to the metadata CSV (train/val/test).
            split_name: Name of the split (e.g., 'train', 'val', 'test') for naming cache files.
            load_cached_data: Whether to try loading from cache first.
        """
        self.split_name = split_name
        self.cache_path = os.path.join(Config.CACHE_DIR, f"cached_{split_name}.npz")

        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading {split_name} data from cache: {self.cache_path}")
            self.load_cache()
        else:
            print(f"Processing {split_name} data from scratch...")
            self.process_and_cache(metadata_path)

    def process_and_cache(self, metadata_path):
        # 1. Load Data
        print("Loading metadata and structures...")
        df_meta = pd.read_csv(metadata_path)

        # In Debug mode, reduce dataset size
        if Config.DEBUG:
            print(
                f"DEBUG mode: reducing dataset to {Config.DEBUG_SAMPLE_SIZE} molecules."
            )
            unique_mols = df_meta["molecule_name"].unique()[: Config.DEBUG_SAMPLE_SIZE]
            df_meta = df_meta[df_meta["molecule_name"].isin(unique_mols)]

        # Load structures
        # We load the full structures.csv but filter for relevant molecules to save memory/time
        df_struct = pd.read_csv(os.path.join(Config.INPUT_DIR, "structures.csv"))
        relevant_mols = set(df_meta["molecule_name"].unique())
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Group data for efficient access
        # Sort to ensure order
        df_struct = df_struct.sort_values(["molecule_name", "atom_index"])
        struct_groups = df_struct.groupby("molecule_name")

        df_meta = df_meta.sort_values(["molecule_name"])
        meta_groups = df_meta.groupby("molecule_name")

        # Initialize Feature Extractors (run on CPU for preprocessing)
        # We use the library classes to ensure consistency
        rbf_expansion = RBFExpansion(num_rbf=Config.NUM_RBF, cutoff=Config.CUTOFF).to(
            "cpu"
        )
        sbf_expansion = SBFExpansion(
            num_sbf=Config.NUM_SBF, num_rbf=Config.NUM_RBF, cutoff=Config.CUTOFF
        ).to("cpu")

        # Storage containers for jagged arrays
        # We will store everything in lists and then concatenate
        all_atom_z = []
        all_pos = []

        all_edge_index = []
        all_edge_rbf = []

        all_triplet_indices = []  # (2, T) -> (incoming_edge_idx, outgoing_edge_idx)
        all_triplet_sbf = []

        all_coupling_atom_indices = []
        all_coupling_types = []
        all_coupling_values = []
        all_coupling_ids = []
        all_coupling_edge_indices = (
            []
        )  # Index of the edge u->v in the graph, -1 if not exists

        # Slices to reconstruct molecules
        # Format: (start_index, count)
        slices = {"atoms": [], "edges": [], "triplets": [], "couplings": []}

        mol_names = []

        # Counters
        cnt_atoms = 0
        cnt_edges = 0
        cnt_triplets = 0
        cnt_couplings = 0

        # Processing Loop
        print(f"Processing {len(relevant_mols)} molecules...")

        # We iterate over the molecules present in metadata
        # Note: meta_groups keys are sorted unique molecule names
        for mol_name, group_meta in meta_groups:
            if mol_name not in struct_groups.groups:
                continue  # Should not happen based on logic above

            # --- 1. Node Features & Geometry ---
            group_struct = struct_groups.get_group(mol_name)

            # Atomic numbers and positions
            atoms_z = (
                group_struct["atom"]
                .map(
                    lambda x: ATOM_MAP.get(
                        pd.api.types.infer_dtype([x]) == "string"
                        and {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}.get(x, 0)
                        or x,
                        0,
                    )
                )
                .values
            )
            # Re-map standard atomic numbers to 0-based index if needed, or just use atomic number
            # The prompt's ATOM_MAP logic is slightly ambiguous, let's stick to standard atomic numbers
            # and map them during model forward or here. Let's map here for embedding lookup.
            # Actually, let's just use the raw atomic number from the file (H=1, C=6 etc)
            # and let the model handle the embedding mapping.
            # But wait, 'atom' column in structures.csv is 'H', 'C', etc.
            atom_symbol_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
            atoms_z = group_struct["atom"].map(atom_symbol_to_z).values.astype(np.int64)

            pos = group_struct[["x", "y", "z"]].values.astype(np.float32)
            n_atoms = len(atoms_z)

            # --- 2. Edge Construction (Cutoff) ---
            # Use KDTree for fast neighbor search
            tree = cKDTree(pos)
            # query_pairs returns set of (i, j) where i < j and dist < cutoff
            # We need directed edges, so we take (i, j) and (j, i)
            pairs = tree.query_pairs(Config.CUTOFF)

            if len(pairs) == 0:
                # Handle isolated atoms case (unlikely in molecules)
                edge_index = np.empty((2, 0), dtype=np.int64)
                dist_vec = np.empty((0,), dtype=np.float32)
            else:
                pairs = np.array(list(pairs), dtype=np.int64)
                # Create bidirectional edges
                src = np.concatenate([pairs[:, 0], pairs[:, 1]])
                dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
                edge_index = np.stack([src, dst], axis=0)  # (2, E)

                # Compute distances
                diff = pos[dst] - pos[src]
                dist_vec = np.linalg.norm(diff, axis=1)

            n_edges = edge_index.shape[1]

            # --- 3. Edge Features (RBF) ---
            if n_edges > 0:
                dist_tensor = torch.from_numpy(dist_vec).float()
                # RBF Expansion
                rbf_feats = (
                    rbf_expansion(dist_tensor).detach().numpy().astype(np.float32)
                )
            else:
                rbf_feats = np.empty((0, Config.NUM_RBF), dtype=np.float32)

            # --- 4. Triplet Construction & SBF ---
            # We need triplets k -> j -> i (where k != i)
            # edge_index[0] is source, edge_index[1] is target
            # For an edge e_ji (j->i), we need incoming edges e_kj (k->j)

            triplet_indices = np.empty((2, 0), dtype=np.int64)
            sbf_feats = np.empty(
                (0, Config.NUM_SBF * Config.NUM_RBF), dtype=np.float16
            )  # Save as fp16

            if n_edges > 0:
                # Create an adjacency list for edges: target_node -> list of edge_indices
                # Sort by target node to make finding incoming edges fast
                # Or use pandas/numpy logic

                # Let's use a robust numpy method
                # We want to match edge_index[1] (target of incoming) with edge_index[0] (source of outgoing)

                # Broadcast approach might be memory heavy if edges are many, but here E ~ 100
                # E x E comparison
                # incoming (k->j): target is edge_index[1, :]
                # outgoing (j->i): source is edge_index[0, :]

                # Find pairs of edges (e1, e2) such that target(e1) == source(e2)
                # and source(e1) != target(e2) (k != i)

                targets = edge_index[1]
                sources = edge_index[0]

                # Find indices where targets[e1] == sources[e2]
                # Using searchsorted or simple equality broadcasting
                # Given E is small (~100), broadcasting is fine.

                # (E, 1) == (1, E) -> (E, E) boolean matrix
                adj_mat = targets[:, None] == sources[None, :]

                # Filter k != i: source(e1) != target(e2)
                # sources[:, None] != targets[None, :]
                non_backtrack = sources[:, None] != targets[None, :]

                valid_triplets = adj_mat & non_backtrack

                # Indices of (incoming, outgoing)
                # row indices are incoming edge indices, col indices are outgoing
                inc_idx, out_idx = np.where(valid_triplets)

                if len(inc_idx) > 0:
                    triplet_indices = np.stack([inc_idx, out_idx], axis=0)  # (2, T)

                    # Compute Angles
                    # Vector k->j: pos[j] - pos[k]
                    # Vector j->i: pos[i] - pos[j]
                    # Note: dist_vec corresponds to the edge vector.
                    # edge k->j vector is pos[j] - pos[k]
                    # edge j->i vector is pos[i] - pos[j]

                    # We need angle between k->j and j->i?
                    # Usually bond angle is defined by vectors j->k and j->i.
                    # Vector j->k is -(pos[j] - pos[k]) = -vec_kj
                    # Vector j->i is vec_ji

                    # Get vectors
                    vecs = pos[dst] - pos[src]  # (E, 3)

                    v1 = -vecs[inc_idx]  # j->k
                    v2 = vecs[out_idx]  # j->i

                    # Cosine angle
                    # dot product
                    dot = np.sum(v1 * v2, axis=1)
                    # norms
                    n1 = dist_vec[inc_idx]
                    n2 = dist_vec[out_idx]

                    cosine = dot / (n1 * n2 + 1e-6)
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angles = np.arccos(cosine)  # in radians

                    # SBF Calculation
                    # library.features.SBFExpansion(dist, angle, idx_kj)
                    # dist: Tensor of all edge lengths
                    # angle: Tensor of triplet angles
                    # idx_kj: Indices of the "kj" edge (incoming edge) in the dist array

                    dist_th = torch.from_numpy(dist_vec).float()
                    angle_th = torch.from_numpy(angles).float()
                    idx_kj_th = torch.from_numpy(inc_idx).long()

                    # Compute SBF
                    sbf_out = sbf_expansion(dist_th, angle_th, idx_kj_th)
                    sbf_feats = sbf_out.detach().numpy().astype(np.float16)

            n_triplets = triplet_indices.shape[1]

            # --- 5. Couplings ---
            # Targets
            c_atom_0 = group_meta["atom_index_0"].values
            c_atom_1 = group_meta["atom_index_1"].values
            c_types = group_meta["type"].map(TYPE_TO_ID).values
            c_ids = group_meta["id"].values

            if "scalar_coupling_constant" in group_meta.columns:
                c_values = group_meta["scalar_coupling_constant"].values.astype(
                    np.float32
                )
            else:
                c_values = np.zeros(len(c_ids), dtype=np.float32)  # Test set

            # Map coupling pairs to directed edges u->v for readout
            # We need to find the index of edge u->v in 'edge_index'
            # If it doesn't exist (dist > cutoff), we assign -1

            c_edge_indices = np.full(len(c_ids), -1, dtype=np.int64)

            if n_edges > 0:
                # Create a map from (u, v) -> edge_idx
                # Since E is small, we can just iterate or use a dict
                # Using a dict for O(1) lookup per coupling
                edge_lookup = {(s, d): i for i, (s, d) in enumerate(zip(src, dst))}

                for k, (u, v) in enumerate(zip(c_atom_0, c_atom_1)):
                    if (u, v) in edge_lookup:
                        c_edge_indices[k] = edge_lookup[(u, v)]
                    # Note: We only look for u->v. The coupling is symmetric but the readout
                    # strategy defined in the idea says "Local directed edge embedding u->v".
                    # We could also look for v->u, but let's stick to the definition.
                    # If u->v exists, v->u also exists in our graph construction.

            n_couplings = len(c_ids)

            # --- 6. Append to Lists ---
            all_atom_z.append(atoms_z)
            all_pos.append(pos)

            all_edge_index.append(edge_index)
            all_edge_rbf.append(rbf_feats)

            all_triplet_indices.append(triplet_indices)
            all_triplet_sbf.append(sbf_feats)

            all_coupling_atom_indices.append(np.stack([c_atom_0, c_atom_1], axis=1))
            all_coupling_types.append(c_types)
            all_coupling_values.append(c_values)
            all_coupling_ids.append(c_ids)
            all_coupling_edge_indices.append(c_edge_indices)

            mol_names.append(mol_name)

            # Update slices
            slices["atoms"].append((cnt_atoms, n_atoms))
            slices["edges"].append((cnt_edges, n_edges))
            slices["triplets"].append((cnt_triplets, n_triplets))
            slices["couplings"].append((cnt_couplings, n_couplings))

            cnt_atoms += n_atoms
            cnt_edges += n_edges
            cnt_triplets += n_triplets
            cnt_couplings += n_couplings

        # Concatenate
        print("Concatenating arrays...")
        data_dict = {
            "atom_z": np.concatenate(all_atom_z, axis=0).astype(np.int64),
            "pos": np.concatenate(all_pos, axis=0).astype(np.float32),
            "edge_index": np.concatenate(all_edge_index, axis=1).astype(np.int64),
            "edge_rbf": np.concatenate(all_edge_rbf, axis=0).astype(np.float32),
            "triplet_indices": np.concatenate(all_triplet_indices, axis=1).astype(
                np.int64
            ),
            "triplet_sbf": np.concatenate(all_triplet_sbf, axis=0).astype(np.float16),
            "coupling_atom_index": np.concatenate(
                all_coupling_atom_indices, axis=0
            ).astype(np.int64),
            "coupling_type": np.concatenate(all_coupling_types, axis=0).astype(
                np.int64
            ),
            "coupling_value": np.concatenate(all_coupling_values, axis=0).astype(
                np.float32
            ),
            "coupling_id": np.concatenate(all_coupling_ids, axis=0).astype(np.int64),
            "coupling_edge_index": np.concatenate(
                all_coupling_edge_indices, axis=0
            ).astype(np.int64),
            "mol_names": np.array(mol_names, dtype=str),
            "slices_atoms": np.array(slices["atoms"], dtype=np.int64),
            "slices_edges": np.array(slices["edges"], dtype=np.int64),
            "slices_triplets": np.array(slices["triplets"], dtype=np.int64),
            "slices_couplings": np.array(slices["couplings"], dtype=np.int64),
        }

        # Save
        print(f"Saving to {self.cache_path}...")
        np.savez(self.cache_path, **data_dict)

        # Load into memory
        self.data = data_dict
        self.slices = slices  # This is redundant but convenient, will be overwritten by load_cache logic
        self.load_cache()  # Ensure consistency

    def load_cache(self):
        # Load the npz file
        # mmap_mode='r' allows us to not load everything if RAM is tight,
        # but with 220GB we can load fully.
        loaded = np.load(self.cache_path)

        # We convert to dict of arrays
        self.data = {k: loaded[k] for k in loaded.files}

        # Parse slices back to convenient structure
        self.slices = {
            "atoms": self.data["slices_atoms"],
            "edges": self.data["slices_edges"],
            "triplets": self.data["slices_triplets"],
            "couplings": self.data["slices_couplings"],
        }

        self.num_molecules = len(self.data["mol_names"])
        print(f"Loaded {self.num_molecules} molecules from cache.")

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        # Retrieve slices
        sl_a = self.slices["atoms"][idx]
        sl_e = self.slices["edges"][idx]
        sl_t = self.slices["triplets"][idx]
        sl_c = self.slices["couplings"][idx]

        # Slice data
        # Atoms
        atom_z = torch.from_numpy(self.data["atom_z"][sl_a[0] : sl_a[0] + sl_a[1]])
        pos = torch.from_numpy(self.data["pos"][sl_a[0] : sl_a[0] + sl_a[1]])

        # Edges
        # Note: edge_index refers to local atom indices (0 to N-1)
        # In the concatenated array, they are 0 to N-1 because we constructed them per molecule.
        # So we just slice.
        edge_index = torch.from_numpy(
            self.data["edge_index"][:, sl_e[0] : sl_e[0] + sl_e[1]]
        )
        edge_rbf = torch.from_numpy(self.data["edge_rbf"][sl_e[0] : sl_e[0] + sl_e[1]])

        # Triplets
        triplet_indices = torch.from_numpy(
            self.data["triplet_indices"][:, sl_t[0] : sl_t[0] + sl_t[1]]
        )
        # Cast SBF back to float32
        triplet_sbf = torch.from_numpy(
            self.data["triplet_sbf"][sl_t[0] : sl_t[0] + sl_t[1]]
        ).float()

        # Couplings
        coupling_atom_index = torch.from_numpy(
            self.data["coupling_atom_index"][sl_c[0] : sl_c[0] + sl_c[1]]
        )
        coupling_type = torch.from_numpy(
            self.data["coupling_type"][sl_c[0] : sl_c[0] + sl_c[1]]
        )
        coupling_value = torch.from_numpy(
            self.data["coupling_value"][sl_c[0] : sl_c[0] + sl_c[1]]
        )
        coupling_id = torch.from_numpy(
            self.data["coupling_id"][sl_c[0] : sl_c[0] + sl_c[1]]
        )
        coupling_edge_index = torch.from_numpy(
            self.data["coupling_edge_index"][sl_c[0] : sl_c[0] + sl_c[1]]
        )

        mol_name = str(self.data["mol_names"][idx])

        return {
            "atom_z": atom_z,
            "pos": pos,
            "edge_index": edge_index,
            "edge_rbf": edge_rbf,
            "triplet_indices": triplet_indices,
            "triplet_sbf": triplet_sbf,
            "coupling_atom_index": coupling_atom_index,
            "coupling_type": coupling_type,
            "coupling_value": coupling_value,
            "coupling_id": coupling_id,
            "coupling_edge_index": coupling_edge_index,
            "mol_name": mol_name,
            "num_atoms": sl_a[1],
            "num_edges": sl_e[1],
        }


def collate_graphs(batch):
    """
    Custom collate function to batch variable-sized molecular graphs.
    Concatenates features and increments indices (edge_index, coupling_atom_index, etc.)
    """

    # Initialize lists
    atom_z_list = []
    pos_list = []
    edge_index_list = []
    edge_rbf_list = []
    triplet_indices_list = []
    triplet_sbf_list = []

    c_atom_index_list = []
    c_type_list = []
    c_value_list = []
    c_id_list = []
    c_edge_index_list = []

    mol_name_list = []

    # Offsets
    atom_offset = 0
    edge_offset = 0

    # Batch index vectors (optional, for scatter operations if needed)
    batch_atom = []
    batch_edge = []

    for i, data in enumerate(batch):
        num_atoms = data["num_atoms"]
        num_edges = data["num_edges"]

        # Nodes
        atom_z_list.append(data["atom_z"])
        pos_list.append(data["pos"])
        batch_atom.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Edges
        # Increment atom indices in edge_index
        edge_index_list.append(data["edge_index"] + atom_offset)
        edge_rbf_list.append(data["edge_rbf"])
        batch_edge.append(torch.full((num_edges,), i, dtype=torch.long))

        # Triplets
        # Increment edge indices in triplet_indices
        triplet_indices_list.append(data["triplet_indices"] + edge_offset)
        triplet_sbf_list.append(data["triplet_sbf"])

        # Couplings
        # Increment atom indices
        c_atom_index_list.append(data["coupling_atom_index"] + atom_offset)
        c_type_list.append(data["coupling_type"])
        c_value_list.append(data["coupling_value"])
        c_id_list.append(data["coupling_id"])

        # Increment edge indices for coupling readout
        # Note: -1 indicates no edge, so we only increment non-negative values
        c_edge_idx = data["coupling_edge_index"].clone()
        mask = c_edge_idx >= 0
        c_edge_idx[mask] += edge_offset
        c_edge_index_list.append(c_edge_idx)

        mol_name_list.append(data["mol_name"])

        # Update offsets
        atom_offset += num_atoms
        edge_offset += num_edges

    # Concatenate
    batch_dict = {
        "atom_z": torch.cat(atom_z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "batch_atom": torch.cat(batch_atom, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_rbf": torch.cat(edge_rbf_list, dim=0),
        "batch_edge": torch.cat(batch_edge, dim=0),
        "triplet_indices": torch.cat(triplet_indices_list, dim=1),
        "triplet_sbf": torch.cat(triplet_sbf_list, dim=0),
        "coupling_atom_index": torch.cat(c_atom_index_list, dim=0),
        "coupling_type": torch.cat(c_type_list, dim=0),
        "coupling_value": torch.cat(c_value_list, dim=0),
        "coupling_id": torch.cat(c_id_list, dim=0),
        "coupling_edge_index": torch.cat(c_edge_index_list, dim=0),
        "mol_names": mol_name_list,
        "batch_size": len(batch),
    }

    return batch_dict
