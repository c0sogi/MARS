import torch
import numpy as np
import os
from torch.utils.data import Dataset
from library.data_factory import DataFactory
from library.config import Config


class FlattenedGraphDataset(Dataset):
    """
    A Dataset wrapper for the flattened, contiguous arrays produced by DataFactory.
    It manages index pointers to slice individual molecular graphs from the monolithic arrays.
    """

    def __init__(self, split="train", load_cached_data=True):
        self.split = split
        self.config = Config()

        # Load the monolithic data dictionary
        factory = DataFactory()
        self.data = factory.process_dataset(
            split=split, load_cached_data=load_cached_data
        )

        # Build index pointers to allow O(1) slicing of molecules
        self._build_index_pointers()

    def _build_index_pointers(self):
        """
        Computes start and end indices for nodes, edges, triplets, and couplings
        for each molecule based on the batch/id arrays.
        """
        # Node Pointers
        # node_batch contains molecule indices (0, 0, 0, 1, 1, ..., N, N)
        node_batch = self.data["node_batch"]
        if len(node_batch) == 0:
            self.num_mols = 0
            return

        # Determine number of molecules
        self.num_mols = node_batch[-1] + 1

        # Calculate node offsets
        node_counts = np.bincount(node_batch, minlength=self.num_mols)
        self.node_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)
        self.node_ptr[1:] = np.cumsum(node_counts)

        # Edge Pointers
        edge_batch = self.data["edge_batch"]
        edge_counts = np.bincount(edge_batch, minlength=self.num_mols)
        self.edge_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)
        self.edge_ptr[1:] = np.cumsum(edge_counts)

        # Triplet Pointers
        # Triplets are defined by indices into the edge array.
        # We determine which molecule a triplet belongs to by looking up the molecule of its first edge.
        triplet_index = self.data["triplet_index"]
        if triplet_index.shape[1] > 0:
            # triplet_index[0] is the index of the first edge in the triplet (global edge index)
            # edge_batch maps global edge index -> molecule index
            triplet_mol_idx = edge_batch[triplet_index[0]]
            triplet_counts = np.bincount(triplet_mol_idx, minlength=self.num_mols)
            self.triplet_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)
            self.triplet_ptr[1:] = np.cumsum(triplet_counts)
        else:
            self.triplet_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)

        # Coupling Pointers
        # Couplings are defined by an edge index.
        coupling_edge_index = self.data["coupling_edge_index"]
        if len(coupling_edge_index) > 0:
            coupling_mol_idx = edge_batch[coupling_edge_index]
            coupling_counts = np.bincount(coupling_mol_idx, minlength=self.num_mols)
            self.coupling_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)
            self.coupling_ptr[1:] = np.cumsum(coupling_counts)
        else:
            self.coupling_ptr = np.zeros(self.num_mols + 1, dtype=np.int64)

    def __len__(self):
        return getattr(self, "num_mols", 0)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the subgraph for molecule `idx`.
        Indices are shifted to be 0-based relative to the molecule.
        """
        # Get start and end pointers
        n_start, n_end = self.node_ptr[idx], self.node_ptr[idx + 1]
        e_start, e_end = self.edge_ptr[idx], self.edge_ptr[idx + 1]
        t_start, t_end = self.triplet_ptr[idx], self.triplet_ptr[idx + 1]
        c_start, c_end = self.coupling_ptr[idx], self.coupling_ptr[idx + 1]

        # --- Extract Nodes ---
        node_x = self.data["node_x"][n_start:n_end]
        node_pos = self.data["node_pos"][n_start:n_end]
        aux_shield = self.data["aux_shielding"][n_start:n_end]
        aux_charge = self.data["aux_charge"][n_start:n_end]

        # --- Extract Edges ---
        # edge_index is (2, M). Global indices.
        # Subtract n_start to make them local (0 to num_nodes-1)
        if e_end > e_start:
            edge_index = self.data["edge_index"][:, e_start:e_end] - n_start
            edge_attr = self.data["edge_attr"][e_start:e_end]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0,), dtype=np.float32)

        # --- Extract Triplets ---
        # triplet_index is (2, K). Global edge indices.
        # Subtract e_start to make them local (0 to num_edges-1)
        if t_end > t_start:
            triplet_index = self.data["triplet_index"][:, t_start:t_end] - e_start
            triplet_attr = self.data["triplet_attr"][t_start:t_end]
        else:
            triplet_index = np.empty((2, 0), dtype=np.int64)
            triplet_attr = np.empty((0,), dtype=np.float32)

        # --- Extract Couplings ---
        # coupling_edge_index is (C,). Global edge indices.
        # Subtract e_start to make them local (0 to num_edges-1)
        if c_end > c_start:
            coupling_edge_index = (
                self.data["coupling_edge_index"][c_start:c_end] - e_start
            )
            coupling_type = self.data["coupling_type"][c_start:c_end]
            coupling_value = self.data["coupling_value"][c_start:c_end]
            coupling_id = self.data["coupling_id"][c_start:c_end]
        else:
            coupling_edge_index = np.empty((0,), dtype=np.int64)
            coupling_type = np.empty((0,), dtype=np.int64)
            coupling_value = np.empty((0,), dtype=np.float32)
            coupling_id = np.empty((0,), dtype=np.int64)

        return {
            "node_x": node_x,
            "node_pos": node_pos,
            "aux_shield": aux_shield,
            "aux_charge": aux_charge,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "triplet_index": triplet_index,
            "triplet_attr": triplet_attr,
            "coupling_edge_index": coupling_edge_index,
            "coupling_type": coupling_type,
            "coupling_value": coupling_value,
            "coupling_id": coupling_id,
            "num_nodes": n_end - n_start,
            "num_edges": e_end - e_start,
        }


class GraphCollator:
    """
    Collates a list of graph dictionaries into a single batched graph.
    Handles cumulative re-indexing of nodes, edges, and triplets.
    """

    def __call__(self, batch):
        # Initialize containers
        node_x_list, node_pos_list = [], []
        aux_shield_list, aux_charge_list = [], []
        edge_index_list, edge_attr_list = [], []
        triplet_index_list, triplet_attr_list = [], []
        c_edge_idx_list, c_type_list, c_val_list, c_id_list = [], [], [], []

        batch_node_batch = []
        batch_edge_batch = []

        # Cumulative offsets for re-indexing
        node_offset = 0
        edge_offset = 0

        for i, sample in enumerate(batch):
            n_nodes = sample["num_nodes"]
            n_edges = sample["num_edges"]

            # --- Nodes ---
            node_x_list.append(torch.from_numpy(sample["node_x"]).long())
            node_pos_list.append(torch.from_numpy(sample["node_pos"]).float())
            aux_shield_list.append(torch.from_numpy(sample["aux_shield"]).float())
            aux_charge_list.append(torch.from_numpy(sample["aux_charge"]).float())
            batch_node_batch.append(torch.full((n_nodes,), i, dtype=torch.long))

            # --- Edges ---
            if n_edges > 0:
                # Re-index edges: local_node_idx + node_offset
                e_idx = torch.from_numpy(sample["edge_index"]).long() + node_offset
                edge_index_list.append(e_idx)
                edge_attr_list.append(torch.from_numpy(sample["edge_attr"]).float())
                batch_edge_batch.append(torch.full((n_edges,), i, dtype=torch.long))

                # --- Triplets ---
                if sample["triplet_index"].shape[1] > 0:
                    # Re-index triplets: local_edge_idx + edge_offset
                    t_idx = (
                        torch.from_numpy(sample["triplet_index"]).long() + edge_offset
                    )
                    triplet_index_list.append(t_idx)
                    triplet_attr_list.append(
                        torch.from_numpy(sample["triplet_attr"]).float()
                    )

                # --- Couplings ---
                if sample["coupling_edge_index"].shape[0] > 0:
                    # Re-index couplings: local_edge_idx + edge_offset
                    c_idx = (
                        torch.from_numpy(sample["coupling_edge_index"]).long()
                        + edge_offset
                    )
                    c_edge_idx_list.append(c_idx)
                    c_type_list.append(torch.from_numpy(sample["coupling_type"]).long())
                    c_val_list.append(
                        torch.from_numpy(sample["coupling_value"]).float()
                    )
                    c_id_list.append(torch.from_numpy(sample["coupling_id"]).long())

            # Update offsets
            node_offset += n_nodes
            edge_offset += n_edges

        # Concatenate all lists into tensors
        batch_data = {}

        batch_data["node_x"] = torch.cat(node_x_list)
        batch_data["node_pos"] = torch.cat(node_pos_list)
        batch_data["node_batch"] = torch.cat(batch_node_batch)
        batch_data["aux_shield"] = torch.cat(aux_shield_list)
        batch_data["aux_charge"] = torch.cat(aux_charge_list)

        if edge_index_list:
            batch_data["edge_index"] = torch.cat(edge_index_list, dim=1)
            batch_data["edge_attr"] = torch.cat(edge_attr_list)
            batch_data["edge_batch"] = torch.cat(batch_edge_batch)
        else:
            batch_data["edge_index"] = torch.empty((2, 0), dtype=torch.long)
            batch_data["edge_attr"] = torch.empty((0,), dtype=torch.float)
            batch_data["edge_batch"] = torch.empty((0,), dtype=torch.long)

        if triplet_index_list:
            batch_data["triplet_index"] = torch.cat(triplet_index_list, dim=1)
            batch_data["triplet_attr"] = torch.cat(triplet_attr_list)
        else:
            batch_data["triplet_index"] = torch.empty((2, 0), dtype=torch.long)
            batch_data["triplet_attr"] = torch.empty((0,), dtype=torch.float)

        if c_edge_idx_list:
            batch_data["coupling_edge_index"] = torch.cat(c_edge_idx_list)
            batch_data["coupling_type"] = torch.cat(c_type_list)
            batch_data["coupling_value"] = torch.cat(c_val_list)
            batch_data["coupling_id"] = torch.cat(c_id_list)
        else:
            batch_data["coupling_edge_index"] = torch.empty((0,), dtype=torch.long)
            batch_data["coupling_type"] = torch.empty((0,), dtype=torch.long)
            batch_data["coupling_value"] = torch.empty((0,), dtype=torch.float)
            batch_data["coupling_id"] = torch.empty((0,), dtype=torch.long)

        return batch_data
