import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.model import ResUNet1D
from library.data import load_data
from library.utils import enu_to_geodetic, set_seed

# Constants
METADATA_DIR = "./metadata"
TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")
MODEL_PATH = "./working/best_model.pth"
SUBMISSION_DIR = "./submission"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceRunner:
    def __init__(self, model_path=MODEL_PATH, output_dir=SUBMISSION_DIR):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = DEVICE
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self, in_channels, base_channels=32):
        """
        Initializes the model architecture and loads weights.
        """
        model = ResUNet1D(
            in_channels=in_channels, out_channels=2, base_channels=base_channels
        )

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from {self.model_path}")
        else:
            print(
                f"Warning: Model file not found at {self.model_path}. Using random weights."
            )

        model.to(self.device)
        model.eval()
        return model

    def run_inference(self, load_cached_data=True, batch_size=1):
        """
        Loads test data, runs inference, and generates the submission file.
        """
        set_seed(42)
        print("Loading test data...")

        # Load Test Dataset
        if not os.path.exists(TEST_META):
            raise FileNotFoundError(f"Test metadata not found at {TEST_META}")

        test_dataset = load_data(
            TEST_META, split="test", load_cached_data=load_cached_data
        )

        if len(test_dataset) == 0:
            print("No test data found.")
            return

        # Determine input dimensions
        sample_x, _ = test_dataset[0]
        in_channels = sample_x.shape[1]
        print(f"Input Feature Dimension: {in_channels}")

        # Initialize Model
        model = self.load_model(in_channels)

        # DataLoader
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        results = []

        print(f"Starting inference on {self.device}...")

        with torch.no_grad():
            for batch_idx, (x, meta) in enumerate(test_loader):
                # x shape: (Batch, SeqLen, Features)
                x = x.to(self.device)

                # Forward pass
                # Output shape: (Batch, SeqLen, 2) -> (East, North) offsets
                outputs = model(x)

                # Move to CPU for processing
                outputs_np = outputs.cpu().numpy()

                # Process each sample in the batch (batch_size is usually 1 here)
                for i in range(x.size(0)):
                    # Extract predictions
                    pred_e = outputs_np[i, :, 0]  # (SeqLen,)
                    pred_n = outputs_np[i, :, 1]  # (SeqLen,)
                    pred_u = np.zeros_like(pred_e)  # 2D task, Up offset = 0

                    # Extract Metadata
                    # Meta items are tensors or lists. Since batch_size=1, we access index i
                    wls_lat = meta["wls_lat"][i].numpy()
                    wls_lon = meta["wls_lon"][i].numpy()
                    wls_alt = meta["wls_alt"][i].numpy()
                    timestamps = meta["UnixTimeMillis"][i].numpy()

                    # Drive ID and Phone Name are strings, DataLoader usually returns them as tuples of strings
                    drive_id = meta["drive_id"][i]
                    phone_name = meta["phone_name"][i]

                    # Convert ENU offsets to Geodetic
                    # enu_to_geodetic is vectorized
                    pred_lat, pred_lon, _ = enu_to_geodetic(
                        pred_e, pred_n, pred_u, wls_lat, wls_lon, wls_alt
                    )

                    # Construct Trip ID
                    # Format: drive_id-phone_name
                    trip_id = f"{drive_id}-{phone_name}"

                    # Append to results
                    for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                        results.append(
                            {
                                "tripId": trip_id,
                                "UnixTimeMillis": t,
                                "LatitudeDegrees": lat,
                                "LongitudeDegrees": lon,
                            }
                        )

        # Create Submission DataFrame
        submission_df = pd.DataFrame(results)

        # Ensure column order
        cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        submission_df = submission_df[cols]

        # Save
        output_path = os.path.join(self.output_dir, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Total predictions: {len(submission_df)}")


def generate_submission(load_cached_data=True):
    """
    Wrapper function to run the inference pipeline.
    """
    runner = InferenceRunner()
    runner.run_inference(load_cached_data=load_cached_data)
