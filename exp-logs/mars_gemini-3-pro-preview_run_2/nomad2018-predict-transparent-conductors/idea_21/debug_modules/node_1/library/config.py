import os


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (using .npz for numpy data or .pth for torch models)
    TRAIN_GRAPHS_CACHE = os.path.join(WORKING_DIR, "train_graphs.npz")
    VAL_GRAPHS_CACHE = os.path.join(WORKING_DIR, "val_graphs.npz")
    TEST_GRAPHS_CACHE = os.path.join(WORKING_DIR, "test_graphs.npz")
    TARGET_SCALER_CACHE = os.path.join(WORKING_DIR, "target_scaler.npz")

    # Model Checkpoint
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Radius for constructing the crystal graph (neighbor search)
    GRAPH_CUTOFF = 5.0  # Angstroms

    # Maximum number of neighbors to consider per atom to manage memory
    MAX_NEIGHBORS = 50

    # Target columns to predict
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Dimension of node embeddings (h_0 and subsequent layers)
    ATOM_EMBEDDING_DIM = 128

    # Number of Gaussian Radial Basis Functions for edge expansion
    NUM_RBF = 60

    # Number of Interaction Blocks (GNN layers)
    NUM_GNN_LAYERS = 4

    # Dropout rate applied within interaction blocks and prediction heads
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Random seed for reproducibility
    SEED = 42

    # Batch size for training and evaluation
    BATCH_SIZE = 48

    # Initial learning rate for AdamW optimizer
    LEARNING_RATE = 1e-3

    # Weight decay for regularization
    WEIGHT_DECAY = 1e-4

    # Maximum number of training epochs
    NUM_EPOCHS = 100

    # Early stopping patience
    PATIENCE = 15
