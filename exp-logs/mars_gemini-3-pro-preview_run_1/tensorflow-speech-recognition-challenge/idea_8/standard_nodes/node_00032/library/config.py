import os


class AudioConfig:
    """Configuration for audio processing and spectrogram generation."""

    sample_rate = 16000
    duration = 1.0  # seconds
    n_fft = 1024  # High spectral resolution (64ms window)
    hop_length = 160  # High temporal resolution (10ms step)
    n_mels = 128  # Number of Mel bands
    fmin = 0
    fmax = 8000

    # Waveform Augmentation
    noise_snr_min = 10  # Minimum SNR (dB) for noise injection
    noise_snr_max = 30  # Maximum SNR (dB) for noise injection

    # Spectrogram Augmentation (SpecAugment)
    freq_mask_param = 20
    time_mask_param = 20


class ModelConfig:
    """Configuration for the Dilated EfficientNet-B2."""

    model_name = "efficientnet_b2"
    pretrained = True
    in_channels = 1  # Log-Mel Spectrogram (1 channel)

    # Classification Head
    # 30 words + 1 silence = 31 fine-grained classes
    num_classes = 31


class TrainConfig:
    """Configuration for the training loop."""

    seed = 42
    batch_size = 64  # Optimized for A100 (40GB)
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-2
    mixup_alpha = 1.0  # Strong Mixup regularization

    # Scheduler (Cosine Annealing)
    T_max = 50
    eta_min = 1e-6

    # Directories
    work_dir = "./working/idea_8/"
    checkpoint_path = os.path.join(work_dir, "best_model.pth")
    submission_path = "./submission/submission.csv"

    # Balancing
    target_samples = 2000  # Upsample target classes to this count


class LabelConfig:
    """Taxonomy and mapping for fine-grained classification."""

    # The 10 target commands + silence
    target_labels = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "silence",
    }

    # The complete list of 31 fine-grained classes (Speech Commands V1 + Silence)
    # This list acts as the vocabulary for the model's output layer.
    fine_grained_labels = sorted(
        [
            "bed",
            "bird",
            "cat",
            "dog",
            "down",
            "eight",
            "five",
            "four",
            "go",
            "happy",
            "house",
            "left",
            "marvin",
            "nine",
            "no",
            "off",
            "on",
            "one",
            "right",
            "seven",
            "sheila",
            "silence",
            "six",
            "stop",
            "three",
            "tree",
            "two",
            "up",
            "wow",
            "yes",
            "zero",
        ]
    )

    @staticmethod
    def get_label_map():
        """
        Returns a dictionary mapping fine-grained labels to the 12 submission labels.
        Logic:
        - Target labels -> themselves
        - Silence -> "silence"
        - All other fine-grained labels -> "unknown"
        """
        label_map = {}
        for label in LabelConfig.fine_grained_labels:
            if label in LabelConfig.target_labels:
                label_map[label] = label
            else:
                label_map[label] = "unknown"
        return label_map
