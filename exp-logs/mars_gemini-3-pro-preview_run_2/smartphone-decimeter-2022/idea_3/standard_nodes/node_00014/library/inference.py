import os
import numpy as np
import pandas as pd
import torch
from library.model import WindowedMLP
from library.data_loader import get_dataloaders


class Predictor:
    """
    Wrapper class for loading the trained model and running inference.
    """

    def __init__(self, model_class, checkpoint_path, device=None, **model_kwargs):
        """
        Initialize the Predictor.

        Args:
            model_class (nn.Module): The model class to instantiate (DSTResNet).
            checkpoint_path (str): Path to the saved model weights.
            device (torch.device, optional): Device to run inference on.
            **model_kwargs: Arguments to pass to the model constructor.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model_class(**model_kwargs)

        if os.path.exists(checkpoint_path):
            # Load weights
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, dataloader):
        """
        Run inference on a dataloader.

        Args:
            dataloader (DataLoader): DataLoader containing test data.

        Returns:
            np.ndarray: Array of predictions (N_samples, Output_dim).
        """
        preds_list = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds_list.append(outputs.cpu().numpy())

        if not preds_list:
            return np.array([])

        return np.concatenate(preds_list, axis=0)


def generate_submission(
    batch_size=256,
    window_size=11,
    checkpoint_path="./working/idea_3/best_model.pth",
    output_path="./submission/submission.csv",
    load_cached_data=True,
    hidden_dim=128,
):
    """
    Generates the submission file for the test set by running the trained model.

    Args:
        batch_size (int): Batch size for inference.
        window_size (int): Temporal window size used during training.
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        load_cached_data (bool): Whether to use cached preprocessed data.
        hidden_dim (int): Hidden dimension size of the model.
    """
    print("Initializing submission generation...")

    # 1. Load Test Data
    # get_dataloaders handles the caching and preprocessing logic internally via library.model.process_data
    # We ignore train/val loaders for inference
    _, _, test_loader, meta_test = get_dataloaders(
        batch_size=batch_size,
        window_size=window_size,
        load_cached_data=load_cached_data,
    )

    if len(test_loader) == 0:
        print("Warning: Test loader is empty. No predictions generated.")
        return

    # 2. Initialize Predictor
    num_features = 8
    input_dim = window_size * num_features

    try:
        predictor = Predictor(
            model_class=WindowedMLP,
            checkpoint_path=checkpoint_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Run Inference
    print("Running inference...")
    # preds_enu shape: (N_samples, 2) -> [DeltaEast, DeltaNorth] in meters
    preds_enu = predictor.predict(test_loader)

    if len(preds_enu) != len(meta_test):
        print(
            f"Error: Mismatch between predictions ({len(preds_enu)}) and metadata ({len(meta_test)})."
        )
        return

    # 4. Reconstruct Coordinates
    print("Reconstructing coordinates...")
    # Extract baseline WLS coordinates
    wls_lat = meta_test["WlsLat"].values
    wls_lon = meta_test["WlsLon"].values

    d_east = preds_enu[:, 0]
    d_north = preds_enu[:, 1]

    # Convert metric residuals (East, North) to geodetic offsets (Lat, Lon)
    # Using local linear approximation on WGS84 ellipsoid
    r_earth = 6378137.0
    d_lat_rad = d_north / r_earth
    # Adjust longitude delta based on latitude (cosine scaling)
    d_lon_rad = d_east / (r_earth * np.cos(np.radians(wls_lat)))

    pred_lat = wls_lat + np.degrees(d_lat_rad)
    pred_lon = wls_lon + np.degrees(d_lon_rad)

    # 5. Save Submission
    submission = pd.DataFrame(
        {
            "tripId": meta_test["tripId"],
            "UnixTimeMillis": meta_test["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Generated {len(submission)} predictions.")
