import os
import random
import numpy as np
import torch
import copy
from collections import OrderedDict


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_metrics(metrics):
    """
    Prints validation metrics with full precision.

    Args:
        metrics (dict): Dictionary containing metric names and values.
    """
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def average_model_weights(state_dicts):
    """
    Computes the average of a list of model state dictionaries.

    Args:
        state_dicts (list): A list of state_dict objects (OrderedDict).

    Returns:
        OrderedDict: A new state_dict containing the averaged weights.
    """
    if not state_dicts:
        raise ValueError("No state_dicts provided to average.")

    num_models = len(state_dicts)

    # Initialize the averaged state_dict with the first model's weights (deep copy to avoid modification)
    avg_state_dict = copy.deepcopy(state_dicts[0])

    # Iterate over all keys in the state_dict
    for key in avg_state_dict.keys():
        # We perform the summation on the CPU to avoid potential GPU memory issues if accumulating many large tensors
        # though usually doing it in place is fine. Here we ensure we handle tensors correctly.

        # Start with the value from the first model
        # Note: avg_state_dict[key] is already a copy of state_dicts[0][key]

        # Add values from the remaining models
        for i in range(1, num_models):
            current_val = state_dicts[i][key]
            # Ensure types match (e.g., if one is on GPU and other on CPU, though typically they should be same)
            if isinstance(current_val, torch.Tensor):
                avg_state_dict[key] += current_val
            else:
                # Handle non-tensor values (like batch norm tracking stats if they are not tensors, though usually they are)
                pass

        # Divide by the number of models to get the average
        if isinstance(avg_state_dict[key], torch.Tensor):
            if avg_state_dict[key].is_floating_point():
                # Use in-place division
                avg_state_dict[key].div_(num_models)
            else:
                # Use floor division for integers (e.g. num_batches_tracked)
                # This prevents RuntimeError: result type Float can't be cast to the desired output type Long
                avg_state_dict[key].div_(num_models, rounding_mode="floor")

    return avg_state_dict


class SWAHelper:
    """
    Helper class to manage Stochastic Weight Averaging (SWA).
    Collects model states at specified intervals and computes the average.
    """

    def __init__(self):
        self.captured_states = []

    def update(self, model):
        """
        Captures the current state of the model.
        Moves weights to CPU to save GPU memory.

        Args:
            model (torch.nn.Module): The model to capture.
        """
        # Get state dict and move to CPU immediately to save GPU memory
        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.captured_states.append(state)

    def get_averaged_weights(self):
        """
        Computes and returns the averaged weights from the captured states.

        Returns:
            OrderedDict: The averaged state_dict.
        """
        if not self.captured_states:
            return None
        return average_model_weights(self.captured_states)

    def reset(self):
        """
        Clears the captured states.
        """
        self.captured_states = []

    def __len__(self):
        return len(self.captured_states)
