import os


class CFG:
    """
    Configuration class for Audio Tagging Task.
    Implements the strategy: ResNeSt-50d + Attention Pooling + Mixup.
    """

    # =======================
    # General Config
    # =======================
    seed = 42
    num_workers = 4
    debug = False  # Set to True for quick debugging runs

    # =======================
    # File Paths
    # =======================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for saving models and cache
    WORKING_DIR = "./working/idea_7"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =======================
    # Audio Parameters
    # =======================
    # Sampling rate: 32kHz is standard for high-quality audio tagging (Nyquist 16kHz)
    sample_rate = 32000
    # Duration for training crops (random crop)
    train_duration = 5.0  # seconds
    # FFT parameters for Mel Spectrogram
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 16000

    # =======================
    # Model Architecture
    # =======================
    # Backbone: ResNeSt-50d (Deep Stem)
    model_name = "resnest50d"
    pretrained = True
    # Input channels: 1 (Mono audio).
    # Note: Model logic will repeat this to 3 channels for the backbone.
    in_channels = 1
    num_classes = 80

    # =======================
    # Training Hyperparameters
    # =======================
    epochs = 30
    batch_size = 48  # Tuned for A100 40GB

    # Optimizer (AdamW)
    lr = 1e-3
    weight_decay = 1e-2

    # Scheduler (Cosine Annealing)
    min_lr = 1e-6
    T_max = 30  # Matches epochs

    # =======================
    # Augmentation
    # =======================
    # Mixup settings
    mixup_alpha = 0.4
    mixup_prob = 1.0  # Apply to 100% of batches

    # SpecAugment parameters (applied on spectrogram)
    freq_mask_param = 24
    time_mask_param = 80

    # =======================
    # Inference
    # =======================
    inference_batch_size = 32

    # =======================
    # Labels
    # =======================
    # Ordered list of classes as per sample_submission.csv
    target_columns = [
        "Accelerating_and_revving_and_vroom",
        "Accordion",
        "Acoustic_guitar",
        "Applause",
        "Bark",
        "Bass_drum",
        "Bass_guitar",
        "Bathtub_(filling_or_washing)",
        "Bicycle_bell",
        "Burping_and_eructation",
        "Bus",
        "Buzz",
        "Car_passing_by",
        "Cheering",
        "Chewing_and_mastication",
        "Child_speech_and_kid_speaking",
        "Chink_and_clink",
        "Chirp_and_tweet",
        "Church_bell",
        "Clapping",
        "Computer_keyboard",
        "Crackle",
        "Cricket",
        "Crowd",
        "Cupboard_open_or_close",
        "Cutlery_and_silverware",
        "Dishes_and_pots_and_pans",
        "Drawer_open_or_close",
        "Drip",
        "Electric_guitar",
        "Fart",
        "Female_singing",
        "Female_speech_and_woman_speaking",
        "Fill_(with_liquid)",
        "Finger_snapping",
        "Frying_(food)",
        "Gasp",
        "Glockenspiel",
        "Gong",
        "Gurgling",
        "Harmonica",
        "Hi-hat",
        "Hiss",
        "Keys_jangling",
        "Knock",
        "Male_singing",
        "Male_speech_and_man_speaking",
        "Marimba_and_xylophone",
        "Mechanical_fan",
        "Meow",
        "Microwave_oven",
        "Motorcycle",
        "Printer",
        "Purr",
        "Race_car_and_auto_racing",
        "Raindrop",
        "Run",
        "Scissors",
        "Screaming",
        "Shatter",
        "Sigh",
        "Sink_(filling_or_washing)",
        "Skateboard",
        "Slam",
        "Sneeze",
        "Squeak",
        "Stream",
        "Strum",
        "Tap",
        "Tick-tock",
        "Toilet_flush",
        "Traffic_noise_and_roadway_noise",
        "Trickle_and_dribble",
        "Walk_and_footsteps",
        "Water_tap_and_faucet",
        "Waves_and_surf",
        "Whispering",
        "Writing",
        "Yell",
        "Zipper_(clothing)",
    ]
