import torch
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import load_dataset


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    This dataset wraps the preprocessed feature matrices and targets, serving
    them as PyTorch tensors. It handles the alignment of features, targets,
    and the auxiliary 'u_out' control signal required for the weighted loss function.
    """

    def __init__(self, X, y=None, u_out=None):
        """
        Args:
            X (np.ndarray): Input features. Shape (Num_Breaths, Seq_Len, Input_Dim).
            y (np.ndarray, optional): Target pressure. Shape (Num_Breaths, Seq_Len).
                                      Can be None for test set.
            u_out (np.ndarray): Binary control input (0 for insp, 1 for exp).
                                Shape (Num_Breaths, Seq_Len).
        """
        # Convert numpy arrays to PyTorch tensors (Float32)
        self.X = torch.tensor(X, dtype=torch.float32)

        # u_out is required for the WeightedL1Loss.
        # We ensure it is float32 for mathematical operations in the loss function.
        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            # Fallback: Create zeros if not provided (should not happen in valid pipeline)
            self.u_out = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.float32)

        # y contains the ground truth pressure values
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        """Returns the total number of breath sequences in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the data for a single breath sequence.

        Returns:
            tuple: (features, target, u_out)
                - features: Tensor of shape (Seq_Len, Input_Dim)
                - target: Tensor of shape (Seq_Len,) or zeros if y is None
                - u_out: Tensor of shape (Seq_Len,)
        """
        x_sample = self.X[idx]
        u_out_sample = self.u_out[idx]

        if self.y is not None:
            y_sample = self.y[idx]
        else:
            # Return a dummy target tensor of zeros for consistency during inference
            # This allows the DataLoader to stack batches without errors
            y_sample = torch.zeros(x_sample.shape[0], dtype=torch.float32)

        return x_sample, y_sample, u_out_sample


def get_ventilator_dataset(split, debug=Config.DEBUG, load_cached_data=True):
    """
    Factory function to create a VentilatorDataset for a specific split.

    This function leverages the data_processing library to load/process data
    and then wraps it in the PyTorch Dataset class.

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, loads a small subset for debugging.
        load_cached_data (bool): If True, attempts to load preprocessed data from disk.

    Returns:
        VentilatorDataset: The instantiated dataset ready for a DataLoader.
    """
    # Load arrays using the provided library function
    X, y, u_out = load_dataset(split, debug=debug, load_cached_data=load_cached_data)

    # Instantiate and return the dataset
    return VentilatorDataset(X, y, u_out)
