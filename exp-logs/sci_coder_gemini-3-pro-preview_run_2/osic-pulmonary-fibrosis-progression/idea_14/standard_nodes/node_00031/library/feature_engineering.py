import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config
from library.image_processing import ImageProcessor


class FeatureProcessor:
    def __init__(self):
        """
        Initializes the FeatureProcessor with necessary transformers.
        """
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.scaler_clinical = StandardScaler()
        self.scaler_morph = StandardScaler()
        self.encoder_clinical = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        self.fitted = False
        self.image_processor = ImageProcessor()

    def _get_raw_image_features(self, df, load_cached_data=True):
        """
        Iterates through the dataframe and extracts raw image features using ImageProcessor.
        Returns arrays for morphology and texture features aligned with the dataframe rows.
        """
        morph_list = []
        texture_list = []

        # Optimize by processing unique patients only once
        unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()
        patient_features = {}

        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            path = row["dcm_path"]

            # ImageProcessor handles disk caching for heavy image ops
            feats = self.image_processor.process_patient(
                pid, path, load_cached_data=load_cached_data
            )
            patient_features[pid] = feats

        # Map unique patient features back to the dataframe rows (which are Patient_Week)
        for pid in df["Patient"]:
            f = patient_features.get(
                pid,
                {
                    "morph": np.zeros(8, dtype=np.float32),
                    "texture": np.zeros(1280 * 3, dtype=np.float32),
                },
            )
            morph_list.append(f["morph"])
            texture_list.append(f["texture"])

        return np.array(morph_list, dtype=np.float32), np.array(
            texture_list, dtype=np.float32
        )

    def fit(self, train_df, train_morph, train_texture):
        """
        Fits PCA and Scalers on training data.
        """
        # 1. Fit PCA on Texture Features
        # Cite debug_lesson_7: Adapt Static Hyperparameters to Runtime Data Dimensions
        n_samples = train_texture.shape[0]
        if n_samples < self.pca.n_components:
            print(
                f"Adjusting PCA components from {self.pca.n_components} to {n_samples} due to sample size."
            )
            self.pca.n_components = n_samples

        self.pca.fit(train_texture)

        # 2. Fit Scaler on Morphological Features
        self.scaler_morph.fit(train_morph)

        # 3. Fit Clinical Preprocessors
        # Continuous: Age, Percent
        clinical_cont = train_df[["Age", "Percent"]].values
        self.scaler_clinical.fit(clinical_cont)

        # Categorical: Sex, SmokingStatus
        clinical_cat = train_df[["Sex", "SmokingStatus"]].values
        self.encoder_clinical.fit(clinical_cat)

        self.fitted = True

    def transform(self, df, morph_raw, texture_raw, mode="fvc"):
        """
        Transforms raw features into specific model inputs.

        Args:
            df (pd.DataFrame): Metadata containing clinical info and Weeks.
            morph_raw (np.array): Raw morphological coefficients.
            texture_raw (np.array): Raw deep texture features.
            mode (str): 'fvc' for quantile regression features, 'uncertainty' for elastic net features.

        Returns:
            np.array: The constructed feature matrix.
        """
        if not self.fitted:
            raise RuntimeError("FeatureProcessor must be fitted before transform.")

        # Cite debug_lesson_2: Harmonize Data Schemas via Runtime Feature Derivation
        if "Percent" not in df.columns and "Baseline_Percent" in df.columns:
            df = df.copy()
            df["Percent"] = df["Baseline_Percent"]

        # 1. Process Texture (PCA)
        texture_pca = self.pca.transform(texture_raw)  # (N, 30)

        # 2. Process Morphology (Scale)
        morph_scaled = self.scaler_morph.transform(morph_raw)  # (N, 8)

        # 3. Process Clinical
        clinical_cont = self.scaler_clinical.transform(df[["Age", "Percent"]].values)
        clinical_cat = self.encoder_clinical.transform(
            df[["Sex", "SmokingStatus"]].values
        )
        clinical_features = np.hstack([clinical_cont, clinical_cat])

        # 4. Concatenate Static Base Vector
        # X_static = [PCA_Texture, Morph_Scaled, Clinical]
        X_static = np.hstack([texture_pca, morph_scaled, clinical_features])

        # 5. Handle Time (Weeks)
        # Calculate Time Delta relative to Baseline CT
        if "Baseline_Weeks" in df.columns:
            # Test set: Weeks is target week, Baseline_Weeks is CT week
            time_delta = (df["Weeks"] - df["Baseline_Weeks"]).values.reshape(-1, 1)
        else:
            # Train/Val set: Weeks is already relative to baseline (0)
            time_delta = df["Weeks"].values.reshape(-1, 1)

        if mode == "fvc":
            # FVC Model: Varying-Coefficient Design
            # Inputs: [X_static, Time_Delta, X_static * Time_Delta]
            interactions = X_static * time_delta
            X_final = np.hstack([X_static, time_delta, interactions])

        elif mode == "uncertainty":
            # Uncertainty Model: Heteroscedasticity based on Horizon
            # Inputs: [X_static, |Time_Delta|]
            horizon = np.abs(time_delta)
            X_final = np.hstack([X_static, horizon])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return X_final.astype(np.float32)

    def process_pipelines(self, load_cached_data=True):
        """
        Main entry point. Loads metadata, processes features, fits transformers,
        and returns dictionaries for train, val, and test sets.

        Implements caching at the dataset level to avoid re-running PCA/assembly.
        """
        # Define cache paths
        cache_files = {
            "train": os.path.join(Config.CACHE_DIR, "dataset_train.npz"),
            "val": os.path.join(Config.CACHE_DIR, "dataset_val.npz"),
            "test": os.path.join(Config.CACHE_DIR, "dataset_test.npz"),
            "preprocessor": os.path.join(Config.CACHE_DIR, "tabular_preprocessor.pkl"),
        }

        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Check if cache exists
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading assembled datasets from cache...")
            train_data = np.load(cache_files["train"])
            val_data = np.load(cache_files["val"])
            test_data = np.load(cache_files["test"])

            # Attempt to restore fitted state
            try:
                state = joblib.load(cache_files["preprocessor"])
                self.pca = state["pca"]
                self.scaler_clinical = state["scaler_clinical"]
                self.scaler_morph = state["scaler_morph"]
                self.encoder_clinical = state["encoder_clinical"]
                self.fitted = True
            except Exception:
                pass

            return dict(train_data), dict(val_data), dict(test_data)

        print("Processing features from scratch...")

        # 1. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        df_val = pd.read_csv(Config.VAL_METADATA)
        df_test = pd.read_csv(Config.TEST_METADATA)

        if Config.DEBUG:
            print(f"DEBUG MODE: limiting to {Config.DEBUG_SAMPLE_SIZE} samples.")
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Extract Raw Image Features
        print("Extracting training images...")
        train_morph, train_texture = self._get_raw_image_features(
            df_train, load_cached_data
        )

        print("Extracting validation images...")
        val_morph, val_texture = self._get_raw_image_features(df_val, load_cached_data)

        print("Extracting test images...")
        test_morph, test_texture = self._get_raw_image_features(
            df_test, load_cached_data
        )

        # 3. Fit Preprocessors on Training Data
        print("Fitting preprocessors...")
        self.fit(df_train, train_morph, train_texture)

        # Save preprocessor state
        state = {
            "pca": self.pca,
            "scaler_clinical": self.scaler_clinical,
            "scaler_morph": self.scaler_morph,
            "encoder_clinical": self.encoder_clinical,
        }
        joblib.dump(state, cache_files["preprocessor"])

        # 4. Transform Features
        print("Transforming features...")

        # Train
        X_train_fvc = self.transform(df_train, train_morph, train_texture, mode="fvc")
        X_train_unc = self.transform(
            df_train, train_morph, train_texture, mode="uncertainty"
        )
        y_train = df_train["FVC"].values.astype(np.float32)

        # Val
        X_val_fvc = self.transform(df_val, val_morph, val_texture, mode="fvc")
        X_val_unc = self.transform(df_val, val_morph, val_texture, mode="uncertainty")
        y_val = df_val["FVC"].values.astype(np.float32)

        # Test
        X_test_fvc = self.transform(df_test, test_morph, test_texture, mode="fvc")
        X_test_unc = self.transform(
            df_test, test_morph, test_texture, mode="uncertainty"
        )
        test_ids = df_test["Patient_Week"].values

        # 5. Save to Cache
        train_dict = {
            "X_fvc": X_train_fvc,
            "X_unc": X_train_unc,
            "y": y_train,
            "weeks": df_train["Weeks"].values,
            "patient": df_train["Patient"].values,
        }
        val_dict = {
            "X_fvc": X_val_fvc,
            "X_unc": X_val_unc,
            "y": y_val,
            "weeks": df_val["Weeks"].values,
            "patient": df_val["Patient"].values,
        }
        test_dict = {"X_fvc": X_test_fvc, "X_unc": X_test_unc, "patient_week": test_ids}

        np.savez(cache_files["train"], **train_dict)
        np.savez(cache_files["val"], **val_dict)
        np.savez(cache_files["test"], **test_dict)

        return train_dict, val_dict, test_dict
