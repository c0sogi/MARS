import torch
import torch.nn as nn
import torchaudio
from library.config import Config


class DifferentiableFrontend(nn.Module):
    """
    A differentiable audio frontend that performs waveform augmentation,
    feature extraction (Mel Spectrogram), and spectrogram augmentation (SpecAugment)
    entirely on the GPU.
    """

    def __init__(self, background_noise=None):
        super().__init__()

        # ==========================================
        # 1. Feature Extraction Setup
        # ==========================================
        # High-resolution Mel Spectrogram configuration
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            n_mels=Config.N_MELS,
            power=2.0,  # Power spectrogram
            normalized=False,
        )

        # Convert power to dB (Log scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

        # ==========================================
        # 2. Augmentation Setup
        # ==========================================
        # SpecAugment transforms
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPECAUG_TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPECAUG_FREQ_MASK_PARAM
        )

        # ==========================================
        # 3. Background Noise Buffer
        # ==========================================
        # We register the noise bank as a buffer so it moves to GPU with the model.
        # persistent=False ensures it is not saved in the model checkpoint (state_dict).
        if background_noise is None:
            # Initialize with a dummy tensor to establish the buffer
            background_noise = torch.zeros(1, Config.NUM_SAMPLES)

        self.register_buffer("background_noise", background_noise, persistent=False)

    def set_background_noise(self, noise_tensor):
        """
        Updates the internal background noise buffer.
        Args:
            noise_tensor (torch.Tensor): Tensor of shape (M, Samples) containing noise clips.
        """
        if noise_tensor.ndim == 1:
            noise_tensor = noise_tensor.unsqueeze(0)

        # Update the buffer
        self.register_buffer("background_noise", noise_tensor, persistent=False)

    def forward(self, x):
        """
        Forward pass of the frontend.
        Args:
            x (torch.Tensor): Input waveforms of shape (Batch, Samples).
        Returns:
            torch.Tensor: Normalized Log-Mel Spectrograms of shape (Batch, 1, Freq, Time).
        """
        # Ensure input is on the correct device (same as the registered buffer)
        x = x.to(self.background_noise.device)

        # ==========================================
        # 1. Waveform Augmentation (Training Only)
        # ==========================================
        # Dynamic Background Noise Mixing
        if self.training and self.background_noise.size(0) > 1:
            batch_size = x.size(0)

            # Determine which samples to augment
            probs = torch.rand(batch_size, device=x.device)
            mask = probs < Config.NOISE_INJECTION_PROB

            if mask.any():
                num_aug = mask.sum()

                # Randomly select noise clips from the buffer
                noise_indices = torch.randint(
                    0, self.background_noise.size(0), (num_aug,), device=x.device
                )
                noise_clips = self.background_noise[noise_indices]

                # Generate random noise weights (0.0 to 0.1)
                # This simulates varying SNR levels
                noise_weights = torch.rand(num_aug, 1, device=x.device) * 0.1

                # Add noise to the signal
                x[mask] = x[mask] + (noise_clips * noise_weights)

                # Optional: Clamp to valid audio range [-1, 1] to prevent numerical instability
                x = torch.clamp(x, -1.0, 1.0)

        # ==========================================
        # 2. Feature Extraction
        # ==========================================
        # Compute Mel Spectrogram: (Batch, n_mels, time)
        spec = self.mel_spectrogram(x)

        # Convert to Log Scale (dB)
        spec = self.amplitude_to_db(spec)

        # ==========================================
        # 3. Instance Normalization
        # ==========================================
        # Normalize each spectrogram independently to Mean=0, Std=1
        # Dimensions: (Batch, Freq, Time) -> Mean/Std over (Freq, Time)
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)

        # Add epsilon for numerical stability
        spec = (spec - mean) / (std + 1e-5)

        # ==========================================
        # 4. Spectrogram Augmentation (Training Only)
        # ==========================================
        # SpecAugment: Time and Frequency Masking
        if self.training:
            # Since data is normalized to mean 0, masking with 0 is effective masking with mean
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # ==========================================
        # 5. Formatting
        # ==========================================
        # Add channel dimension for CNN input: (Batch, 1, Freq, Time)
        spec = spec.unsqueeze(1)

        return spec
