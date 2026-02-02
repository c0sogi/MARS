import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from library.config import Config
from library.utils import setup_logger, Standardizer

logger = setup_logger("data_prep")


class DataProcessor:
    """
    Handles the processing of raw CSV and XYZ data into a flattened Structure-of-Arrays (SoA) format.
    Implements caching, parallel processing, and target standardization.
    """

    def __init__(self):
        self.processed_dir = Config.PROCESSED_DATA_DIR
        os.makedirs(self.processed_dir, exist_ok=True)
        self.standardizer = Standardizer()

    def process_all(self, load_cached_data=True):
        """
        Main entry point. Checks for cached data; if valid, loads it.
        Otherwise, processes raw data from scratch.
        """
        # Define the set of critical files that must exist
        expected_files = [
            Config.CACHE_NODES_PATH,
            Config.CACHE_COORDS_PATH,
            Config.CACHE_EDGES_PATH,
            Config.CACHE_EDGE_ATTRS_PATH,
            Config.CACHE_TRIPLETS_PATH,
            Config.CACHE_MOL_INDICES_PATH,
            Config.CACHE_TRAIN_TARGETS_PATH,
            Config.CACHE_VAL_TARGETS_PATH,
            Config.STATS_PATH,
        ]

        # Check if all files exist
        cache_exists = all(os.path.exists(f) for f in expected_files)

        if load_cached_data and cache_exists:
            logger.info("Cache found. Loading processed data from disk...")
            return self._load_cache()

        if load_cached_data and not cache_exists:
            logger.info("Cache incomplete or missing. Reprocessing data...")

        return self._build_dataset()

    def _load_cache(self):
        """Loads numpy arrays from the processed directory."""
        data = {
            "nodes": np.load(Config.CACHE_NODES_PATH),
            "coords": np.load(Config.CACHE_COORDS_PATH),
            "edge_indices": np.load(Config.CACHE_EDGES_PATH),
            "edge_attrs": np.load(Config.CACHE_EDGE_ATTRS_PATH),
            "triplets": np.load(Config.CACHE_TRIPLETS_PATH),
            "mol_indices": np.load(Config.CACHE_MOL_INDICES_PATH),
            # Aux
            "aux_charges": np.load(os.path.join(self.processed_dir, "aux_charge.npy")),
            "aux_shielding": np.load(
                os.path.join(self.processed_dir, "aux_shielding.npy")
            ),
            # Couplings (Metadata + Values)
            "coupling_meta": np.load(
                os.path.join(self.processed_dir, "coupling_meta.npy")
            ),
            "coupling_values": np.load(
                os.path.join(self.processed_dir, "coupling_values.npy")
            ),
            # Split indices for couplings
            "train_indices": np.load(
                os.path.join(self.processed_dir, "train_indices.npy")
            ),
            "val_indices": np.load(os.path.join(self.processed_dir, "val_indices.npy")),
            "test_indices": np.load(
                os.path.join(self.processed_dir, "test_indices.npy")
            ),
        }

        # Load standardizer stats
        self.standardizer.load()

        logger.info("Data loaded successfully.")
        return data

    def _build_dataset(self):
        """
        Orchestrates the parallel processing of raw data.
        """
        logger.info("Loading raw metadata and structures...")

        # 1. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # 2. Load Structures
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)

        # 3. Load Aux Data
        df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
        df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
        # Dipole/Potential not strictly used in node/edge features but useful for analysis.
        # We focus on node/edge features here.

        # 4. Prepare Data for Parallel Workers
        # Group structures by molecule_name for O(1) access
        # To avoid passing huge DataFrames to workers, we convert to dictionaries of numpy arrays
        logger.info("Organizing data for parallel processing...")

        # Structures: {mol_name: {'atoms': [], 'coords': []}}
        # We use a more memory-efficient approach: dict of DF groups is okay for 1.3M rows
        struct_dict = {
            n: g[["atom", "x", "y", "z"]].values
            for n, g in df_struct.groupby("molecule_name")
        }

        # Couplings: Combine all and group
        # Add split identifier: 0=Train, 1=Val, 2=Test
        df_train["split"] = 0
        df_val["split"] = 1
        df_test["split"] = 2
        df_test["scalar_coupling_constant"] = 0.0  # Placeholder

        df_all_couplings = pd.concat(
            [
                df_train[
                    [
                        "molecule_name",
                        "atom_index_0",
                        "atom_index_1",
                        "type",
                        "scalar_coupling_constant",
                        "id",
                        "split",
                    ]
                ],
                df_val[
                    [
                        "molecule_name",
                        "atom_index_0",
                        "atom_index_1",
                        "type",
                        "scalar_coupling_constant",
                        "id",
                        "split",
                    ]
                ],
                df_test[
                    [
                        "molecule_name",
                        "atom_index_0",
                        "atom_index_1",
                        "type",
                        "scalar_coupling_constant",
                        "id",
                        "split",
                    ]
                ],
            ],
            ignore_index=True,
        )

        coupling_dict = {
            n: g.values for n, g in df_all_couplings.groupby("molecule_name")
        }

        # Aux Data
        # Charges: {mol_name: [charges...]} (sorted by atom index)
        charge_dict = {
            n: g["mulliken_charge"].values
            for n, g in df_charge.groupby("molecule_name")
        }

        # Shielding: {mol_name: [[XX, YX...], ...]}
        shield_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
        shield_dict = {
            n: g[shield_cols].values for n, g in df_shield.groupby("molecule_name")
        }

        # List of all unique molecules to process
        all_molecules = sorted(list(struct_dict.keys()))

        # 5. Parallel Processing
        logger.info(
            f"Processing {len(all_molecules)} molecules with {Config.NUM_WORKERS} workers..."
        )

        # Split into chunks
        chunk_size = len(all_molecules) // (Config.NUM_WORKERS * 4) + 1
        chunks = [
            all_molecules[i : i + chunk_size]
            for i in range(0, len(all_molecules), chunk_size)
        ]

        results = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
            delayed(process_chunk)(
                chunk, struct_dict, coupling_dict, charge_dict, shield_dict
            )
            for chunk in chunks
        )

        # 6. Aggregation
        logger.info("Aggregating processed chunks...")

        # Initialize containers
        nodes_all = []
        coords_all = []
        aux_charge_all = []
        aux_shield_all = []

        edge_indices_all = []
        edge_attrs_all = []
        triplets_all = []

        coupling_meta_all = []  # atom0, atom1, type, id, split, mol_idx
        coupling_vals_all = []

        mol_indices_all = (
            []
        )  # [node_start, node_count, edge_start, edge_count, triplet_start, triplet_count, c_start, c_count]

        # Offsets
        node_offset = 0
        edge_offset = 0
        triplet_offset = 0
        coupling_offset = 0

        for res in results:
            # Unpack chunk results
            (
                c_nodes,
                c_coords,
                c_charges,
                c_shields,
                c_edges,
                c_edge_attrs,
                c_triplets,
                c_c_meta,
                c_c_vals,
                c_mol_counts,
            ) = res

            # Update molecule indices with global offsets
            num_mols_in_chunk = len(c_mol_counts)
            chunk_mol_indices = np.zeros((num_mols_in_chunk, 8), dtype=np.int32)

            current_local_node = 0
            current_local_edge = 0
            current_local_trip = 0
            current_local_coup = 0

            for i, (n_cnt, e_cnt, t_cnt, c_cnt) in enumerate(c_mol_counts):
                chunk_mol_indices[i] = [
                    node_offset + current_local_node,
                    n_cnt,
                    edge_offset + current_local_edge,
                    e_cnt,
                    triplet_offset + current_local_trip,
                    t_cnt,
                    coupling_offset + current_local_coup,
                    c_cnt,
                ]
                current_local_node += n_cnt
                current_local_edge += e_cnt
                current_local_trip += t_cnt
                current_local_coup += c_cnt

            mol_indices_all.append(chunk_mol_indices)

            # Append data
            nodes_all.append(c_nodes)
            coords_all.append(c_coords)
            aux_charge_all.append(c_charges)
            aux_shield_all.append(c_shields)

            edge_indices_all.append(c_edges)
            edge_attrs_all.append(c_edge_attrs)
            triplets_all.append(c_triplets)

            # Update coupling mol_idx to point to global molecule index?
            # Actually, we process sequentially, so the mol_idx in c_c_meta is local to chunk?
            # No, c_c_meta stores raw data. We need to add the global molecule index.
            # We can reconstruct it from the loop.
            # Let's just store the data and use mol_indices to map back if needed.
            # But for training, we iterate molecules.

            coupling_meta_all.append(c_c_meta)
            coupling_vals_all.append(c_c_vals)

            # Update global offsets
            node_offset += len(c_nodes)
            edge_offset += len(c_edges[0]) if len(c_edges) > 0 else 0
            triplet_offset += len(c_triplets[0]) if len(c_triplets) > 0 else 0
            coupling_offset += len(c_c_vals)

        # Concatenate
        logger.info("Concatenating arrays...")
        nodes = np.concatenate(nodes_all).astype(np.int8)
        coords = np.concatenate(coords_all).astype(np.float32)
        aux_charges = np.concatenate(aux_charge_all).astype(np.float32)
        aux_shielding = np.concatenate(aux_shield_all).astype(np.float32)

        # Edges: (2, N)
        # Note: c_edges in chunks were (2, N_local). We just concat them.
        # They are local indices (0..atoms_in_mol). This is correct for SoA.
        edge_indices = np.concatenate(edge_indices_all, axis=1).astype(np.int32)
        edge_attrs = np.concatenate(edge_attrs_all).astype(np.float32)

        # Triplets: (2, N)
        triplets = np.concatenate(triplets_all, axis=1).astype(np.int32)

        mol_indices = np.concatenate(mol_indices_all).astype(np.int32)

        coupling_meta = np.concatenate(coupling_meta_all).astype(np.int32)
        coupling_values = np.concatenate(coupling_vals_all).astype(np.float32)

        # 7. Standardization
        logger.info("Standardizing targets...")

        # Create temporary DF for fitting standardizer
        # coupling_meta cols: 0:atom0, 1:atom1, 2:type, 3:id, 4:split
        train_mask = coupling_meta[:, 4] == 0
        val_mask = coupling_meta[:, 4] == 1
        test_mask = coupling_meta[:, 4] == 2

        # Extract training data for fitting
        train_types = coupling_meta[train_mask, 2]
        train_values = coupling_values[train_mask]

        df_train_fit = pd.DataFrame(
            {"type": train_types, "scalar_coupling_constant": train_values}
        )

        self.standardizer.fit(df_train_fit)

        # Transform all coupling values
        # We need to reconstruct a DF or just use the arrays manually
        # Standardizer expects a DF with 'type' and 'scalar_coupling_constant'
        # But we can also just use the internal stats
        all_types = coupling_meta[:, 2]
        mu = self.standardizer.means[all_types]
        sigma = self.standardizer.stds[all_types]
        coupling_values_norm = (coupling_values - mu) / sigma

        # Aux Standardization
        # Filter out NaNs (from test set) for fitting
        valid_aux_mask = ~np.isnan(aux_charges)
        self.standardizer.fit_aux(
            aux_shielding[valid_aux_mask], aux_charges[valid_aux_mask]
        )

        aux_shield_norm, aux_charge_norm = self.standardizer.transform_aux(
            aux_shielding, aux_charges
        )
        # Fill NaNs with 0 after standardization (mean imputation)
        aux_shield_norm = np.nan_to_num(aux_shield_norm)
        aux_charge_norm = np.nan_to_num(aux_charge_norm)

        # 8. Save to Disk
        logger.info("Saving to disk...")
        np.save(Config.CACHE_NODES_PATH, nodes)
        np.save(Config.CACHE_COORDS_PATH, coords)
        np.save(Config.CACHE_EDGES_PATH, edge_indices)
        np.save(Config.CACHE_EDGE_ATTRS_PATH, edge_attrs)
        np.save(Config.CACHE_TRIPLETS_PATH, triplets)
        np.save(Config.CACHE_MOL_INDICES_PATH, mol_indices)

        np.save(os.path.join(self.processed_dir, "aux_charge.npy"), aux_charge_norm)
        np.save(os.path.join(self.processed_dir, "aux_shielding.npy"), aux_shield_norm)

        np.save(os.path.join(self.processed_dir, "coupling_meta.npy"), coupling_meta)
        np.save(
            os.path.join(self.processed_dir, "coupling_values.npy"),
            coupling_values_norm.astype(np.float32),
        )

        # Save indices for splits
        # We need indices into the 'coupling_...' arrays
        all_indices = np.arange(len(coupling_meta))
        np.save(
            os.path.join(self.processed_dir, "train_indices.npy"),
            all_indices[train_mask],
        )
        np.save(
            os.path.join(self.processed_dir, "val_indices.npy"), all_indices[val_mask]
        )
        np.save(
            os.path.join(self.processed_dir, "test_indices.npy"), all_indices[test_mask]
        )

        # Save standardizer stats
        self.standardizer.save()

        logger.info("Data processing complete.")

        return self._load_cache()


def process_chunk(molecules, struct_dict, coupling_dict, charge_dict, shield_dict):
    """
    Processes a list of molecules into arrays.
    """
    # Local containers
    nodes_list = []
    coords_list = []
    charges_list = []
    shields_list = []

    edges_list = []
    edge_attrs_list = []
    triplets_list = []

    c_meta_list = []
    c_vals_list = []

    mol_counts = []  # (n_nodes, n_edges, n_triplets, n_couplings)

    atom_map = Config.ATOM_MAP
    type_map = Config.TYPE_MAP

    for mol in molecules:
        # 1. Nodes & Coords
        # struct_data is [[atom, x, y, z], ...]
        struct_data = struct_dict[mol]
        atoms = struct_data[:, 0]
        coords = struct_data[:, 1:].astype(np.float32)

        # Encode atoms
        atom_ids = np.array([atom_map.get(a, 0) for a in atoms], dtype=np.int8)
        n_atoms = len(atom_ids)

        # 2. Aux Data
        # If missing (Test set), fill with NaN
        if mol in charge_dict:
            charges = charge_dict[mol].astype(np.float32)
            shields = shield_dict[mol].astype(np.float32)
        else:
            charges = np.full(n_atoms, np.nan, dtype=np.float32)
            shields = np.full((n_atoms, 9), np.nan, dtype=np.float32)

        # 3. Edges (Radius Graph)
        # Use KDTree
        tree = cKDTree(coords)
        # radius=5.0
        pairs = tree.query_pairs(r=Config.RADIUS_CUTOFF)
        # pairs is set of (i, j) with i < j

        # Convert to directed edges (i->j and j->i)
        # Also include self-loops? No.
        src = []
        dst = []
        dists = []
        vecs = []

        # We need to map (u, v) to edge index for triplet construction
        # Key: (u, v), Value: local_edge_index
        edge_map = {}

        # Process edges
        # Convert pairs to list
        pairs_list = list(pairs)
        if len(pairs_list) > 0:
            u_idx = np.array([p[0] for p in pairs_list])
            v_idx = np.array([p[1] for p in pairs_list])

            # Calculate vectors and distances
            diffs = coords[v_idx] - coords[u_idx]  # u -> v
            d_sq = np.sum(diffs**2, axis=1)
            d = np.sqrt(d_sq)

            # Normalize vectors
            # Avoid div by zero
            d_safe = d.copy()
            d_safe[d_safe < 1e-6] = 1.0
            vecs_uv = diffs / d_safe[:, None]

            # Add u->v
            for k in range(len(u_idx)):
                idx = len(src)
                src.append(u_idx[k])
                dst.append(v_idx[k])
                dists.append(d[k])
                vecs.append(vecs_uv[k])
                edge_map[(u_idx[k], v_idx[k])] = idx

            # Add v->u
            for k in range(len(u_idx)):
                idx = len(src)
                src.append(v_idx[k])
                dst.append(u_idx[k])
                dists.append(d[k])
                vecs.append(-vecs_uv[k])
                edge_map[(v_idx[k], u_idx[k])] = idx

        n_edges = len(src)
        if n_edges > 0:
            edges = np.vstack([src, dst]).astype(np.int32)
            edge_attrs = np.column_stack([dists, np.array(vecs)]).astype(np.float32)
        else:
            edges = np.zeros((2, 0), dtype=np.int32)
            edge_attrs = np.zeros((0, 4), dtype=np.float32)

        # 4. Triplets (Directional Message Passing)
        # For each edge j->i (target edge), we need incoming edges k->j (source edges) where k != i
        # We can group edges by their target node
        triplets_local = []

        if n_edges > 0:
            # Create adjacency list: node -> list of incoming edge indices
            incoming_edges = [[] for _ in range(n_atoms)]
            for e_idx, (u, v) in enumerate(zip(src, dst)):
                incoming_edges[v].append((u, e_idx))

            # Iterate over all edges (j -> i) as the "outgoing" edge
            for out_idx, (j, i) in enumerate(zip(src, dst)):
                # Look for k -> j
                for k, in_idx in incoming_edges[j]:
                    if k != i:
                        # Found a triplet k -> j -> i
                        triplets_local.append([in_idx, out_idx])

        if len(triplets_local) > 0:
            triplets = np.array(triplets_local, dtype=np.int32).T  # (2, N_triplets)
        else:
            triplets = np.zeros((2, 0), dtype=np.int32)

        # 5. Couplings
        # coupling_data: [mol, a0, a1, type, val, id, split]
        if mol in coupling_dict:
            c_data = coupling_dict[mol]
            # Map type string to int
            types = np.array([type_map[t] for t in c_data[:, 3]], dtype=np.int32)

            # Construct meta: atom0, atom1, type, id, split
            # c_data columns: 0:name, 1:a0, 2:a1, 3:type, 4:val, 5:id, 6:split
            c_meta = np.column_stack(
                [
                    c_data[:, 1].astype(np.int32),  # atom0
                    c_data[:, 2].astype(np.int32),  # atom1
                    types,
                    c_data[:, 5].astype(np.int32),  # id
                    c_data[:, 6].astype(np.int32),  # split
                ]
            )
            c_vals = c_data[:, 4].astype(np.float32)
        else:
            c_meta = np.zeros((0, 5), dtype=np.int32)
            c_vals = np.zeros((0,), dtype=np.float32)

        # Append to lists
        nodes_list.append(atom_ids)
        coords_list.append(coords)
        charges_list.append(charges)
        shields_list.append(shields)

        edges_list.append(edges)
        edge_attrs_list.append(edge_attrs)
        triplets_list.append(triplets)

        c_meta_list.append(c_meta)
        c_vals_list.append(c_vals)

        mol_counts.append((n_atoms, n_edges, triplets.shape[1], len(c_vals)))

    # Return chunk data
    # Note: edges and triplets are local to molecule. We just return list of arrays.
    # Aggregator will concat.

    return (
        np.concatenate(nodes_list) if nodes_list else np.array([], dtype=np.int8),
        np.concatenate(coords_list) if coords_list else np.array([], dtype=np.float32),
        (
            np.concatenate(charges_list)
            if charges_list
            else np.array([], dtype=np.float32)
        ),
        (
            np.concatenate(shields_list)
            if shields_list
            else np.array([], dtype=np.float32)
        ),
        # Edges need to be kept separate or concatenated with offset?
        # The aggregator expects list of arrays where each array is (2, N_local).
        # Wait, if we concat here, we lose the boundary unless we return counts.
        # We return the lists of arrays directly? No, pickle overhead.
        # Better to concat here but keep them local (0..N-1).
        # Actually, `np.concatenate(edges_list, axis=1)` creates one big array of local indices.
        # This is fine because `mol_counts` tells us how to slice it.
        (
            np.concatenate(edges_list, axis=1)
            if edges_list
            else np.zeros((2, 0), dtype=np.int32)
        ),
        (
            np.concatenate(edge_attrs_list)
            if edge_attrs_list
            else np.zeros((0, 4), dtype=np.float32)
        ),
        (
            np.concatenate(triplets_list, axis=1)
            if triplets_list
            else np.zeros((2, 0), dtype=np.int32)
        ),
        (
            np.concatenate(c_meta_list)
            if c_meta_list
            else np.zeros((0, 5), dtype=np.int32)
        ),
        (
            np.concatenate(c_vals_list)
            if c_vals_list
            else np.zeros((0,), dtype=np.float32)
        ),
        mol_counts,
    )
