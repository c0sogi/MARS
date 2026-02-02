import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from library.config import Config

# =========================================================================
# Feature Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}


def get_node_features(sequence, loop_type, structure):
    """
    Generates One-Hot encoded node features for an RNA sequence.

    Args:
        sequence (str): RNA sequence (A, G, U, C).
        loop_type (str): Predicted loop type (S, M, I, B, H, E, X).
        structure (str): Dot-bracket structure (., (, )).

    Returns:
        torch.Tensor: Node features of shape (Seq_Len, 14).
    """
    seq_len = len(sequence)

    # Initialize feature tensors
    # Dimensions: 4 (Seq) + 7 (Loop) + 3 (Struct) = 14
    features = np.zeros((seq_len, Config.NUM_NODE_FEATURES), dtype=np.float32)

    for i in range(seq_len):
        # 1. Sequence One-Hot (Indices 0-3)
        char_seq = sequence[i]
        if char_seq in SEQ_MAP:
            features[i, SEQ_MAP[char_seq]] = 1.0

        # 2. Loop Type One-Hot (Indices 4-10)
        char_loop = loop_type[i]
        if char_loop in LOOP_MAP:
            features[i, 4 + LOOP_MAP[char_loop]] = 1.0

        # 3. Structure One-Hot (Indices 11-13)
        char_struct = structure[i]
        if char_struct in STRUCT_MAP:
            features[i, 11 + STRUCT_MAP[char_struct]] = 1.0

    return torch.tensor(features, dtype=torch.float)


def structure_to_edge_index(structure):
    """
    Parses a dot-bracket structure string to create a graph edge index.
    Includes backbone edges (i <-> i+1) and pairing edges (base pairs).

    Args:
        structure (str): Dot-bracket structure string.

    Returns:
        torch.LongTensor: Edge index of shape (2, Num_Edges).
    """
    seq_len = len(structure)
    edges = []

    # 1. Backbone Edges (Linear chain)
    # Connect i to i+1 and i+1 to i
    for i in range(seq_len - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    # 2. Pairing Edges (Hydrogen bonds)
    # Use a stack to find matching parentheses
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Add bidirectional edge between i and j
                edges.append([i, j])
                edges.append([j, i])

    if not edges:
        # Fallback for single node or weird case, though unlikely given seq_len=107
        return torch.empty((2, 0), dtype=torch.long)

    # Convert to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class RNAGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for RNA Degradation.
    Loads data from Parquet files, converts to Graph objects, and caches them.
    """

    def __init__(
        self,
        parquet_path,
        cache_name,
        load_cached_data=True,
        is_test=False,
        root=Config.WORKING_DIR,
    ):
        self.parquet_path = parquet_path
        self.cache_path = os.path.join(root, f"{cache_name}.pt")
        self.is_test = is_test
        self.load_cached_data = load_cached_data

        # Ensure working directory exists
        os.makedirs(root, exist_ok=True)

        # Initialize InMemoryDataset
        super().__init__(root)

        # Load or Process
        if self.load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path}")
            self.data, self.slices = torch.load(self.cache_path)
        else:
            print(f"Processing dataset from {self.parquet_path}")
            self.process_data()

    def process_data(self):
        # Load raw data
        df = pd.read_parquet(self.parquet_path)

        data_list = []

        for idx, row in df.iterrows():
            # Extract inputs
            sequence = row["sequence"]
            structure = row["structure"]
            loop_type = row["predicted_loop_type"]

            # Generate Graph Features
            x = get_node_features(sequence, loop_type, structure)
            edge_index = structure_to_edge_index(structure)

            # Generate Targets (if not test)
            y = None
            if not self.is_test:
                # Extract target lists
                # Each column is a list of floats (length seq_scored=68)
                targets = []
                for col in Config.TARGET_COLS:
                    val = row[col]
                    # Ensure it's a list/array
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    targets.append(val)

                # Shape: (5, 68) -> Transpose to (68, 5)
                targets_array = np.array(targets, dtype=np.float32).T

                # Pad to seq_length (107)
                # We pad with 0.0. The loss function handles slicing, so these values won't be used.
                seq_len = Config.SEQ_LENGTH
                seq_scored = Config.SEQ_SCORED

                if targets_array.shape[0] != seq_scored:
                    # Safety check
                    # In case data is malformed, truncate or pad blindly
                    # But assuming clean data based on metadata script
                    pass

                padding_len = seq_len - seq_scored
                if padding_len > 0:
                    padding = np.zeros(
                        (padding_len, Config.NUM_TARGETS), dtype=np.float32
                    )
                    y_full = np.vstack([targets_array, padding])
                else:
                    y_full = targets_array

                y = torch.tensor(y_full, dtype=torch.float)

            # Create Data object
            # We store 'id' to map predictions back to sample IDs later
            data = Data(x=x, edge_index=edge_index, y=y, id=row["id"])
            data_list.append(data)

        # Collate and Save
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        print(f"Saving processed dataset to {self.cache_path}")
        torch.save((self.data, self.slices), self.cache_path)

    @property
    def processed_file_names(self):
        return [os.path.basename(self.cache_path)]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Set
    train_dataset = RNAGraphDataset(
        parquet_path=Config.TRAIN_DATA_PATH,
        cache_name="train_graph_data",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Validation Set
    val_dataset = RNAGraphDataset(
        parquet_path=Config.VAL_DATA_PATH,
        cache_name="val_graph_data",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Test Set
    test_dataset = RNAGraphDataset(
        parquet_path=Config.TEST_DATA_PATH,
        cache_name="test_graph_data",
        load_cached_data=load_cached_data,
        is_test=True,
    )

    # Create Loaders
    # shuffle=True for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # shuffle=False for val/test
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
