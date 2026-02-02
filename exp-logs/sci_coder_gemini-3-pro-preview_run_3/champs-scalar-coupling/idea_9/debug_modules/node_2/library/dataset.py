import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import Config
from library.preprocessing import process_dataset


class FlattenedMoleculeDataset(Dataset):
    """
    A PyTorch Dataset that wraps the flattened, monolithic numpy arrays.
    It efficiently slices the arrays to return individual molecules.
    """

    def __init__(self, split="train", load_cached=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to load cached numpy arrays.
        """
        self.split = split

        # Load the monolithic data dictionary
        # This contains arrays like 'node_z', 'edge_index', 'node_batch', etc.
        self.data = process_dataset(split, load_cached_data=load_cached)

        # Determine the number of molecules
        # node_batch contains molecule indices 0..N-1
        if len(self.data["node_batch"]) > 0:
            self.num_mols = self.data["node_batch"][-1] + 1
        else:
            self.num_mols = 0

        # --- Pre-compute Index Pointers ---
        # We use searchsorted to find the start/end indices for each molecule
        # in the various arrays. This is much faster than grouping.

        # Create a range of molecule indices to search for: [0, 1, ..., num_mols]
        mol_range = np.arange(self.num_mols + 1)

        # 1. Node Pointers
        # Maps mol_idx -> range in node arrays
        self.node_ptr = np.searchsorted(self.data["node_batch"], mol_range)

        # 2. Edge Pointers
        # Maps mol_idx -> range in edge arrays
        self.edge_ptr = np.searchsorted(self.data["edge_batch"], mol_range)

        # 3. Target Pointers
        # Maps mol_idx -> range in target arrays
        self.target_ptr = np.searchsorted(
            self.data["target_molecule_indices"], mol_range
        )

        # 4. Triplet Pointers
        # Maps mol_idx -> range in triplet arrays.
        # Triplets are not explicitly batched by molecule index in the file,
        # but they are stored in blocks corresponding to molecules.
        # triplet_indices[0] contains the index of the incoming edge.
        # The edges for molecule i are in range [edge_ptr[i], edge_ptr[i+1]).
        # Therefore, triplets for molecule i correspond to where triplet_indices[0]
        # falls within that edge range.
        if "triplet_indices" in self.data and self.data["triplet_indices"].shape[1] > 0:
            # We search for the start of each edge block in the triplet incoming-edge array
            self.triplet_ptr = np.searchsorted(
                self.data["triplet_indices"][0], self.edge_ptr
            )
        else:
            self.triplet_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)

    def __len__(self):
        return self.num_mols

    def __getitem__(self, idx):
        """
        Returns a dictionary representing a single molecule.
        Indices are normalized to be 0-based for this molecule.
        """
        # Get start and end indices for this molecule
        n_start, n_end = self.node_ptr[idx], self.node_ptr[idx + 1]
        e_start, e_end = self.edge_ptr[idx], self.edge_ptr[idx + 1]
        t_start, t_end = self.triplet_ptr[idx], self.triplet_ptr[idx + 1]
        tgt_start, tgt_end = self.target_ptr[idx], self.target_ptr[idx + 1]

        # --- Extract Nodes ---
        node_z = torch.from_numpy(self.data["node_z"][n_start:n_end]).long()
        node_pos = torch.from_numpy(self.data["node_pos"][n_start:n_end]).float()

        # --- Extract Edges ---
        # Normalize edge_index: subtract n_start so indices point to local nodes 0..num_atoms-1
        edge_index = (
            torch.from_numpy(self.data["edge_index"][:, e_start:e_end]) - n_start
        )
        edge_index = edge_index.long()

        edge_dist = torch.from_numpy(self.data["edge_dist"][e_start:e_end]).float()
        edge_vec = torch.from_numpy(self.data["edge_vec"][e_start:e_end]).float()

        # --- Extract Triplets ---
        if t_end > t_start:
            # Normalize triplet_indices: subtract e_start so indices point to local edges 0..num_edges-1
            triplet_indices = (
                torch.from_numpy(self.data["triplet_indices"][:, t_start:t_end])
                - e_start
            )
            triplet_indices = triplet_indices.long()
            triplet_angles = torch.from_numpy(
                self.data["triplet_angles"][t_start:t_end]
            ).float()
        else:
            triplet_indices = torch.empty((2, 0), dtype=torch.long)
            triplet_angles = torch.empty((0,), dtype=torch.float32)

        # --- Extract Targets ---
        # Normalize target_indices: subtract e_start so they point to local edges
        target_indices = (
            torch.from_numpy(self.data["target_indices"][tgt_start:tgt_end]) - e_start
        )
        target_indices = target_indices.long()
        target_types = torch.from_numpy(
            self.data["target_types"][tgt_start:tgt_end]
        ).long()

        item = {
            "node_z": node_z,
            "node_pos": node_pos,
            "edge_index": edge_index,
            "edge_dist": edge_dist,
            "edge_vec": edge_vec,
            "triplet_indices": triplet_indices,
            "triplet_angles": triplet_angles,
            "target_indices": target_indices,
            "target_types": target_types,
            "num_nodes": n_end - n_start,
            "num_edges": e_end - e_start,
        }

        # --- Extract Labels / Aux Data (Train/Val only) ---
        if self.split in ["train", "val"]:
            item["target_values"] = torch.from_numpy(
                self.data["target_values"][tgt_start:tgt_end]
            ).float()

            # Aux data is aligned with nodes
            item["aux_shielding"] = torch.from_numpy(
                self.data["aux_shielding"][n_start:n_end]
            ).float()
            item["aux_charges"] = torch.from_numpy(
                self.data["aux_charges"][n_start:n_end]
            ).float()

            # Aux data per molecule
            item["aux_dipole"] = torch.tensor(
                self.data["aux_dipole"][idx], dtype=torch.float32
            )
            item["aux_potential"] = torch.tensor(
                self.data["aux_potential"][idx], dtype=torch.float32
            )

        return item


def collate_batch(batch_list):
    """
    Collates a list of molecule dictionaries into a single batched dictionary.
    Re-indexes indices to create a disjoint graph.
    """
    # Lists to hold the batched data
    node_z_list = []
    node_pos_list = []
    node_batch_list = []

    edge_index_list = []
    edge_dist_list = []
    edge_vec_list = []
    edge_batch_list = []

    triplet_indices_list = []
    triplet_angles_list = []

    target_indices_list = []
    target_types_list = []
    target_batch_list = []

    # Optional lists for targets/aux
    target_values_list = []
    aux_shielding_list = []
    aux_charges_list = []
    aux_dipole_list = []
    aux_potential_list = []

    # Cumulative counters for re-indexing
    cum_nodes = 0
    cum_edges = 0

    has_targets = "target_values" in batch_list[0]

    for i, item in enumerate(batch_list):
        num_nodes = item["num_nodes"]
        num_edges = item["num_edges"]

        # --- Nodes ---
        node_z_list.append(item["node_z"])
        node_pos_list.append(item["node_pos"])
        # Create batch index vector for nodes
        node_batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        # --- Edges ---
        # Shift edge_index to point to the correct nodes in the batch
        edge_index_list.append(item["edge_index"] + cum_nodes)
        edge_dist_list.append(item["edge_dist"])
        edge_vec_list.append(item["edge_vec"])
        edge_batch_list.append(torch.full((num_edges,), i, dtype=torch.long))

        # --- Triplets ---
        if item["triplet_indices"].numel() > 0:
            # Shift triplet_indices to point to the correct edges in the batch
            triplet_indices_list.append(item["triplet_indices"] + cum_edges)
            triplet_angles_list.append(item["triplet_angles"])

        # --- Targets ---
        # Shift target_indices to point to the correct edges in the batch
        target_indices_list.append(item["target_indices"] + cum_edges)
        target_types_list.append(item["target_types"])
        target_batch_list.append(
            torch.full((item["target_types"].shape[0],), i, dtype=torch.long)
        )

        # --- Aux / Labels ---
        if has_targets:
            target_values_list.append(item["target_values"])
            aux_shielding_list.append(item["aux_shielding"])
            aux_charges_list.append(item["aux_charges"])
            aux_dipole_list.append(item["aux_dipole"])
            aux_potential_list.append(item["aux_potential"])

        cum_nodes += num_nodes
        cum_edges += num_edges

    # --- Concatenate ---
    batch = {
        "node_z": torch.cat(node_z_list),
        "node_pos": torch.cat(node_pos_list),
        "node_batch": torch.cat(node_batch_list),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_dist": torch.cat(edge_dist_list),
        "edge_vec": torch.cat(edge_vec_list),
        "edge_batch": torch.cat(edge_batch_list),
        "target_indices": torch.cat(target_indices_list),
        "target_types": torch.cat(target_types_list),
        "target_batch": torch.cat(target_batch_list),
        "batch_size": len(batch_list),
    }

    if triplet_indices_list:
        batch["triplet_indices"] = torch.cat(triplet_indices_list, dim=1)
        batch["triplet_angles"] = torch.cat(triplet_angles_list)
    else:
        batch["triplet_indices"] = torch.empty((2, 0), dtype=torch.long)
        batch["triplet_angles"] = torch.empty((0,), dtype=torch.float32)

    if has_targets:
        batch["target_values"] = torch.cat(target_values_list)
        batch["aux_shielding"] = torch.cat(aux_shielding_list)
        batch["aux_charges"] = torch.cat(aux_charges_list)
        batch["aux_dipole"] = torch.stack(aux_dipole_list)
        batch["aux_potential"] = torch.stack(aux_potential_list)

    return batch
