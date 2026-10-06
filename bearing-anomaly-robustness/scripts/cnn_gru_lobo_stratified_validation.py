# =============================================================================
# STEP 155 - CORRECTED CNN-GRU WITH RECORDING-LEVEL STRATIFIED VALIDATION
#
# Strict 10-bearing LOBO
# Correct Paderborn channel: Y[6] / vibration_1
# Nominal sampling rate: 64 kHz
#
# CORRECTION FROM STEP 151:
# Original:
#     validation_split=0.15
#
# Corrected:
#     recording-level stratified 15% validation split
#
# All four windows belonging to a recording remain in the same
# train/validation/test partition.
# =============================================================================

import os
from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    MaxPooling1D,
    GRU,
    Dense,
    Dropout,
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

ROOT = os.environ.get("BEARING_DATA_ROOT", str(Path(__file__).resolve().parent))

OUTPUT_DIR = os.path.join(
    ROOT,
    "step155_cnn_gru_lobo_stratified_validation_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# -------------------------------------------------------------------------
# Signal configuration
# -------------------------------------------------------------------------

FS = 64000

WINDOW_SIZE = 64000

N_WINDOWS_PER_RECORDING = 4

REQUIRED_SAMPLES = (
    WINDOW_SIZE * N_WINDOWS_PER_RECORDING
)

# -------------------------------------------------------------------------
# Temporal chunk configuration
# -------------------------------------------------------------------------

CHUNK_SIZE = 500

N_CHUNKS = (
    WINDOW_SIZE // CHUNK_SIZE
)

# -------------------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------------------

EXPECTED_RECORDINGS = 800

EXPECTED_RECORDINGS_PER_BEARING = 80

BEARINGS = [
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
    "KA01",
    "KA03",
    "KA04",
    "KI05",
]

HEALTHY_BEARINGS = {
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
}

DAMAGED_BEARINGS = {
    "KA01",
    "KA03",
    "KA04",
    "KI05",
}

# -------------------------------------------------------------------------
# CNN-GRU configuration
# -------------------------------------------------------------------------

EPOCHS = 30

BATCH_SIZE = 8

LEARNING_RATE = 1e-3

VALIDATION_SIZE = 0.15

EARLY_STOPPING_PATIENCE = 5

# -------------------------------------------------------------------------
# Frozen publication reference
# -------------------------------------------------------------------------

FROZEN_PRIMARY_BASELINE_F1 = 0.6159

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# =============================================================================
# LABEL FUNCTION
# =============================================================================

def get_label(bearing):

    if bearing in HEALTHY_BEARINGS:

        return 0

    if bearing in DAMAGED_BEARINGS:

        return 1

    raise ValueError(
        f"Unknown bearing: {bearing}"
    )


# =============================================================================
# DISCOVER MATLAB FILES
# =============================================================================

def discover_files():

    records = []

    print("=" * 80)

    print(
        "DISCOVERING MATLAB FILES"
    )

    print("=" * 80)

    for bearing in BEARINGS:

        bearing_dir = os.path.join(
            ROOT,
            bearing
        )

        if not os.path.isdir(
            bearing_dir
        ):

            raise FileNotFoundError(
                f"Bearing directory not found: "
                f"{bearing_dir}"
            )

        files = sorted(
            os.path.join(
                bearing_dir,
                f
            )
            for f in os.listdir(
                bearing_dir
            )
            if f.lower().endswith(".mat")
        )

        print(
            f"{bearing:<6} "
            f"{len(files):>3} files"
        )

        if len(files) != (
            EXPECTED_RECORDINGS_PER_BEARING
        ):

            raise RuntimeError(
                f"{bearing} contains "
                f"{len(files)} files; "
                f"expected "
                f"{EXPECTED_RECORDINGS_PER_BEARING}"
            )

        for path in files:

            records.append(
                {
                    "Bearing": bearing,
                    "File": path,
                    "True_Label": get_label(
                        bearing
                    ),
                }
            )

    if len(records) != (
        EXPECTED_RECORDINGS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_RECORDINGS} recordings, "
            f"found {len(records)}"
        )

    print()

    print(
        "Total MATLAB files discovered:",
        len(records)
    )

    return records


# =============================================================================
# CORRECT PADERBORN MATLAB LOADER
#
# REQUIRED CHANNEL:
#     root.Y
#     Y[6]
#     Name == vibration_1
#     Data
# =============================================================================

def extract_vibration_signal(path):

    mat = sio.loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False
    )

    keys = [
        key
        for key in mat.keys()
        if not key.startswith("__")
    ]

    if not keys:

        raise RuntimeError(
            f"No MATLAB root structure found: "
            f"{path}"
        )

    root = mat[keys[0]]

    if not hasattr(
        root,
        "Y"
    ):

        raise RuntimeError(
            f"MATLAB structure has no Y field: "
            f"{path}"
        )

    y = np.asarray(
        root.Y,
        dtype=object
    ).ravel()

    if len(y) < 7:

        raise RuntimeError(
            f"Y contains only "
            f"{len(y)} channels: "
            f"{path}"
        )

    channel = y[6]

    channel_name = getattr(
        channel,
        "Name",
        None
    )

    if isinstance(
        channel_name,
        np.ndarray
    ):

        if channel_name.size > 0:

            try:

                channel_name = (
                    channel_name.item()
                )

            except Exception:

                channel_name = str(
                    channel_name.flatten()[0]
                )

    channel_name = str(
        channel_name
    )

    if channel_name != "vibration_1":

        raise RuntimeError(
            f"Unexpected Y[6] channel: "
            f"{channel_name}"
        )

    if not hasattr(
        channel,
        "Data"
    ):

        raise RuntimeError(
            f"Y[6] has no Data field: "
            f"{path}"
        )

    signal = np.asarray(
        channel.Data,
        dtype=np.float32
    ).squeeze().ravel()

    # -------------------------------------------------------------------------
    # Original handling of 1-10 samples short
    # -------------------------------------------------------------------------

    if len(signal) < REQUIRED_SAMPLES:

        missing = (
            REQUIRED_SAMPLES
            - len(signal)
        )

        if missing <= 10:

            signal = np.pad(
                signal,
                (
                    0,
                    missing
                ),
                mode="constant",
                constant_values=0
            )

        else:

            raise RuntimeError(
                f"Signal too short: "
                f"{path}; "
                f"available={len(signal)}, "
                f"required={REQUIRED_SAMPLES}"
            )

    if not np.all(
        np.isfinite(signal)
    ):

        raise RuntimeError(
            f"Non-finite samples found: "
            f"{path}"
        )

    return signal


# =============================================================================
# EXTRACT FOUR NON-OVERLAPPING WINDOWS
# =============================================================================

def extract_windows(signal):

    windows = []

    for i in range(
        N_WINDOWS_PER_RECORDING
    ):

        start = (
            i * WINDOW_SIZE
        )

        end = (
            start + WINDOW_SIZE
        )

        window = signal[
            start:end
        ].copy()

        if len(window) != WINDOW_SIZE:

            raise RuntimeError(
                f"Unexpected window length: "
                f"{len(window)}"
            )

        windows.append(
            window
        )

    return np.asarray(
        windows,
        dtype=np.float32
    )


# =============================================================================
# PER-WINDOW STANDARDIZATION
# =============================================================================

def standardize_windows(
    windows
):

    mean = np.mean(
        windows,
        axis=1,
        keepdims=True
    )

    std = np.std(
        windows,
        axis=1,
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-8
    )

    normalized = (
        windows - mean
    ) / std

    return normalized.astype(
        np.float32
    )


# =============================================================================
# CONVERT TO 128 x 500 TEMPORAL CHUNKS
# =============================================================================

def create_temporal_chunks(
    windows
):

    if (
        N_CHUNKS * CHUNK_SIZE
        != WINDOW_SIZE
    ):

        raise RuntimeError(
            "Chunk configuration does "
            "not cover the full window."
        )

    chunks = windows.reshape(
        windows.shape[0],
        N_CHUNKS,
        CHUNK_SIZE
    )

    return chunks.astype(
        np.float32
    )


# =============================================================================
# LOAD COMPLETE DATASET
# =============================================================================

def load_dataset(
    records
):

    print()

    print("=" * 80)

    print(
        "LOADING AND PREPROCESSING DATA"
    )

    print("=" * 80)

    X = []

    y = []

    metadata = []

    failures = []

    total = len(records)

    for idx, record in enumerate(
        records,
        start=1
    ):

        path = record[
            "File"
        ]

        bearing = record[
            "Bearing"
        ]

        label = record[
            "True_Label"
        ]

        try:

            signal = (
                extract_vibration_signal(
                    path
                )
            )

            windows = (
                extract_windows(
                    signal
                )
            )

            windows = (
                standardize_windows(
                    windows
                )
            )

            chunks = (
                create_temporal_chunks(
                    windows
                )
            )

            # -----------------------------------------------------------------
            # Four examples per recording
            # -----------------------------------------------------------------

            for window_idx in range(
                N_WINDOWS_PER_RECORDING
            ):

                X.append(
                    chunks[
                        window_idx
                    ]
                )

                y.append(
                    label
                )

                metadata.append(
                    {
                        "Recording_ID": (
                            idx - 1
                        ),
                        "Bearing": bearing,
                        "File": path,
                        "Window": window_idx,
                        "True_Label": label,
                    }
                )

        except Exception as exc:

            failures.append(
                {
                    "Bearing": bearing,
                    "File": path,
                    "Error": str(exc),
                }
            )

        if (
            idx % 50 == 0
            or idx == total
        ):

            print(
                f"Processed "
                f"{idx}/{total}"
            )

    failures_df = pd.DataFrame(
        failures
    )

    failures_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "step155_loading_failures.csv"
        ),
        index=False
    )

    if failures:

        print()

        print(
            failures_df.head(
                10
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{len(failures)} recordings "
            f"failed to load."
        )

    expected_windows = (
        EXPECTED_RECORDINGS
        * N_WINDOWS_PER_RECORDING
    )

    if len(X) != expected_windows:

        raise RuntimeError(
            f"Expected "
            f"{expected_windows} windows, "
            f"got {len(X)}"
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    print()

    print(
        "CNN-GRU input shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    print(
        "Metadata rows:",
        len(metadata_df)
    )

    return (
        X,
        y,
        metadata_df
    )


# =============================================================================
# BUILD ORIGINAL CNN-GRU
# =============================================================================

def build_model():

    model = Sequential(
        [

            Input(
                shape=(
                    N_CHUNKS,
                    CHUNK_SIZE
                )
            ),

            # -------------------------------------------------------------
            # First convolution
            # -------------------------------------------------------------

            Conv1D(
                filters=16,
                kernel_size=11,
                padding="same"
            ),

            BatchNormalization(),

            Activation(
                "relu"
            ),

            MaxPooling1D(
                pool_size=4
            ),

            # -------------------------------------------------------------
            # Second convolution
            # -------------------------------------------------------------

            Conv1D(
                filters=32,
                kernel_size=7,
                padding="same"
            ),

            BatchNormalization(),

            Activation(
                "relu"
            ),

            # -------------------------------------------------------------
            # GRU temporal modeling
            # -------------------------------------------------------------

            GRU(
                64,
                return_sequences=False
            ),

            # -------------------------------------------------------------
            # Classifier
            # -------------------------------------------------------------

            Dense(
                64,
                activation="relu"
            ),

            Dropout(
                0.30
            ),

            Dense(
                1,
                activation="sigmoid"
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# =============================================================================
# RECORDING-LEVEL METADATA
# =============================================================================

def create_recording_table(
    metadata
):

    recording_table = (
        metadata[
            [
                "Recording_ID",
                "Bearing",
                "File",
                "True_Label"
            ]
        ]
        .drop_duplicates(
            "Recording_ID"
        )
        .reset_index(
            drop=True
        )
    )

    return recording_table


# =============================================================================
# CONVERT RECORDING IDS TO WINDOW INDICES
# =============================================================================

def get_window_indices(
    metadata,
    recording_ids
):

    recording_ids = set(
        recording_ids
    )

    mask = metadata[
        "Recording_ID"
    ].isin(
        recording_ids
    )

    return np.where(
        mask.values
    )[0]


# =============================================================================
# RECORDING-LEVEL PREDICTION
# =============================================================================

def predict_recordings(
    model,
    X_test,
    test_metadata
):

    probabilities = (
        model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .reshape(-1)
    )

    result = test_metadata.copy()

    result[
        "Window_Probability"
    ] = probabilities

    rows = []

    grouped = result.groupby(
        [
            "Bearing",
            "File",
            "True_Label",
            "Recording_ID"
        ],
        sort=False
    )

    for (
        bearing,
        file_path,
        true_label,
        recording_id
    ), group in grouped:

        probability = (
            group[
                "Window_Probability"
            ].mean()
        )

        predicted_label = int(
            probability >= 0.5
        )

        rows.append(
            {
                "Recording_ID": int(
                    recording_id
                ),
                "Bearing": bearing,
                "File": file_path,
                "True_Label": int(
                    true_label
                ),
                "Predicted_Label": (
                    predicted_label
                ),
                "Probability": float(
                    probability
                ),
                "Number_of_Windows": (
                    len(group)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "TN": int(
            cm[0, 0]
        ),

        "FP": int(
            cm[0, 1]
        ),

        "FN": int(
            cm[1, 0]
        ),

        "TP": int(
            cm[1, 1]
        ),
    }


# =============================================================================
# BEARING-LEVEL METRICS
# =============================================================================

def calculate_bearing_metrics(
    predictions_df
):

    rows = []

    for bearing, group in (
        predictions_df.groupby(
            "Bearing",
            sort=True
        )
    ):

        y_true = group[
            "True_Label"
        ].values

        y_pred = group[
            "Predicted_Label"
        ].values

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        tn = metrics[
            "TN"
        ]

        fp = metrics[
            "FP"
        ]

        if (
            tn + fp
        ) > 0:

            healthy_fpr = (
                fp / (
                    tn + fp
                )
            )

            specificity = (
                tn / (
                    tn + fp
                )
            )

        else:

            healthy_fpr = 0.0

            specificity = 0.0

        rows.append(
            {
                "Bearing": bearing,

                "Accuracy": metrics[
                    "Accuracy"
                ],

                "Precision": metrics[
                    "Precision"
                ],

                "Recall": metrics[
                    "Recall"
                ],

                "F1": metrics[
                    "F1"
                ],

                "Healthy_FPR": (
                    healthy_fpr
                ),

                "Specificity": (
                    specificity
                ),

                "Failures": (
                    metrics["FP"]
                    + metrics["FN"]
                ),

                "TN": metrics[
                    "TN"
                ],

                "FP": metrics[
                    "FP"
                ],

                "FN": metrics[
                    "FN"
                ],

                "TP": metrics[
                    "TP"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# STRICT LOBO
# =============================================================================

def run_lobo(
    X,
    y,
    metadata
):

    all_predictions = []

    fold_results = []

    validation_audit = []

    recording_meta = (
        create_recording_table(
            metadata
        )
    )

    print()

    print("=" * 80)

    print(
        "STRICT 10-BEARING LOBO"
    )

    print(
        "CORRECTED RECORDING-LEVEL "
        "STRATIFIED VALIDATION"
    )

    print("=" * 80)

    for fold_idx, test_bearing in enumerate(
        BEARINGS,
        start=1
    ):

        print()

        print("-" * 80)

        print(
            f"LOBO FOLD {fold_idx}/10: "
            f"TEST BEARING = "
            f"{test_bearing}"
        )

        print("-" * 80)

        # =====================================================================
        # OUTER LOBO SPLIT
        # =====================================================================

        test_recordings = (
            recording_meta[
                recording_meta[
                    "Bearing"
                ] == test_bearing
            ].copy()
        )

        train_pool_recordings = (
            recording_meta[
                recording_meta[
                    "Bearing"
                ] != test_bearing
            ].copy()
        )

        if len(
            test_recordings
        ) != 80:

            raise RuntimeError(
                f"{test_bearing}: "
                f"expected 80 test recordings, "
                f"got {len(test_recordings)}"
            )

        if len(
            train_pool_recordings
        ) != 720:

            raise RuntimeError(
                f"{test_bearing}: "
                f"expected 720 training-pool "
                f"recordings, "
                f"got {len(train_pool_recordings)}"
            )

        # =====================================================================
        # CORRECTED INTERNAL VALIDATION SPLIT
        #
        # IMPORTANT:
        # Split recordings FIRST.
        # Then convert recordings to their four window indices.
        #
        # Therefore no recording can be split across train/validation.
        # =====================================================================

        (
            train_recordings,
            val_recordings
        ) = train_test_split(
            train_pool_recordings,
            test_size=VALIDATION_SIZE,
            stratify=(
                train_pool_recordings[
                    "True_Label"
                ]
            ),
            random_state=SEED,
            shuffle=True
        )

        train_recordings = (
            train_recordings
            .reset_index(
                drop=True
            )
        )

        val_recordings = (
            val_recordings
            .reset_index(
                drop=True
            )
        )

        # =====================================================================
        # RECORDING OVERLAP AUDIT
        # =====================================================================

        train_ids = set(
            train_recordings[
                "Recording_ID"
            ]
        )

        val_ids = set(
            val_recordings[
                "Recording_ID"
            ]
        )

        test_ids = set(
            test_recordings[
                "Recording_ID"
            ]
        )

        if train_ids & val_ids:

            raise RuntimeError(
                "TRAIN/VALIDATION "
                "RECORDING OVERLAP"
            )

        if train_ids & test_ids:

            raise RuntimeError(
                "TRAIN/TEST "
                "RECORDING OVERLAP"
            )

        if val_ids & test_ids:

            raise RuntimeError(
                "VALIDATION/TEST "
                "RECORDING OVERLAP"
            )

        # =====================================================================
        # EXPECTED RECORDING COUNTS
        # =====================================================================

        if len(
            train_recordings
        ) != 612:

            raise RuntimeError(
                f"Expected 612 training "
                f"recordings, "
                f"got {len(train_recordings)}"
            )

        if len(
            val_recordings
        ) != 108:

            raise RuntimeError(
                f"Expected 108 validation "
                f"recordings, "
                f"got {len(val_recordings)}"
            )

        # =====================================================================
        # BOTH CLASSES MUST BE PRESENT
        # =====================================================================

        if (
            train_recordings[
                "True_Label"
            ].nunique()
            != 2
        ):

            raise RuntimeError(
                "Training split does not "
                "contain both classes."
            )

        if (
            val_recordings[
                "True_Label"
            ].nunique()
            != 2
        ):

            raise RuntimeError(
                "Validation split does not "
                "contain both classes."
            )

        # =====================================================================
        # RECORDINGS -> WINDOW INDICES
        # =====================================================================

        train_indices = (
            get_window_indices(
                metadata,
                train_ids
            )
        )

        val_indices = (
            get_window_indices(
                metadata,
                val_ids
            )
        )

        test_indices = (
            get_window_indices(
                metadata,
                test_ids
            )
        )

        # =====================================================================
        # EXPECTED WINDOW COUNTS
        # =====================================================================

        if len(
            train_indices
        ) != 2448:

            raise RuntimeError(
                f"Expected 2448 training "
                f"windows, "
                f"got {len(train_indices)}"
            )

        if len(
            val_indices
        ) != 432:

            raise RuntimeError(
                f"Expected 432 validation "
                f"windows, "
                f"got {len(val_indices)}"
            )

        if len(
            test_indices
        ) != 320:

            raise RuntimeError(
                f"Expected 320 test "
                f"windows, "
                f"got {len(test_indices)}"
            )

        # =====================================================================
        # CLASS DISTRIBUTION AUDIT
        # =====================================================================

        train_counts = (
            train_recordings[
                "True_Label"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        val_counts = (
            val_recordings[
                "True_Label"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print()

        print(
            "Recording-level split:"
        )

        print(
            "  Outer training pool:",
            len(
                train_pool_recordings
            )
        )

        print(
            "  Training recordings:",
            len(
                train_recordings
            )
        )

        print(
            "  Validation recordings:",
            len(
                val_recordings
            )
        )

        print(
            "  Test recordings:",
            len(
                test_recordings
            )
        )

        print()

        print(
            "Training class distribution:",
            train_counts
        )

        print(
            "Validation class distribution:",
            val_counts
        )

        print()

        print(
            "Window counts:"
        )

        print(
            "  Training:",
            len(
                train_indices
            )
        )

        print(
            "  Validation:",
            len(
                val_indices
            )
        )

        print(
            "  Test:",
            len(
                test_indices
            )
        )

        # =====================================================================
        # VALIDATION AUDIT
        # =====================================================================

        validation_audit.append(
            {
                "Fold": fold_idx,

                "Test_Bearing": (
                    test_bearing
                ),

                "Outer_Training_Recordings": (
                    len(
                        train_pool_recordings
                    )
                ),

                "Training_Recordings": (
                    len(
                        train_recordings
                    )
                ),

                "Validation_Recordings": (
                    len(
                        val_recordings
                    )
                ),

                "Test_Recordings": (
                    len(
                        test_recordings
                    )
                ),

                "Training_Windows": (
                    len(
                        train_indices
                    )
                ),

                "Validation_Windows": (
                    len(
                        val_indices
                    )
                ),

                "Test_Windows": (
                    len(
                        test_indices
                    )
                ),

                "Train_Healthy_Recordings": (
                    int(
                        (
                            train_recordings[
                                "True_Label"
                            ] == 0
                        ).sum()
                    )
                ),

                "Train_Damaged_Recordings": (
                    int(
                        (
                            train_recordings[
                                "True_Label"
                            ] == 1
                        ).sum()
                    )
                ),

                "Val_Healthy_Recordings": (
                    int(
                        (
                            val_recordings[
                                "True_Label"
                            ] == 0
                        ).sum()
                    )
                ),

                "Val_Damaged_Recordings": (
                    int(
                        (
                            val_recordings[
                                "True_Label"
                            ] == 1
                        ).sum()
                    )
                ),
            }
        )

        # =====================================================================
        # FRESH MODEL FOR EACH LOBO FOLD
        # =====================================================================

        tf.keras.backend.clear_session()

        random.seed(
            SEED
        )

        np.random.seed(
            SEED
        )

        tf.random.set_seed(
            SEED
        )

        model = build_model()

        if fold_idx == 1:

            print()

            print(
                "CNN-GRU architecture:"
            )

            model.summary()

        # =====================================================================
        # EARLY STOPPING
        # =====================================================================

        early_stopping = (
            EarlyStopping(
                monitor="val_loss",
                patience=(
                    EARLY_STOPPING_PATIENCE
                ),
                restore_best_weights=True,
                verbose=1
            )
        )

        # =====================================================================
        # CORRECTED MODEL TRAINING
        #
        # DO NOT USE:
        #     validation_split=0.15
        #
        # USE EXPLICIT RECORDING-LEVEL VALIDATION DATA.
        # =====================================================================

        history = model.fit(
            X[
                train_indices
            ],

            y[
                train_indices
            ],

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            validation_data=(
                X[
                    val_indices
                ],

                y[
                    val_indices
                ]
            ),

            shuffle=True,

            callbacks=[
                early_stopping
            ],

            verbose=1
        )

        # =====================================================================
        # SAVE HISTORY
        # =====================================================================

        history_df = pd.DataFrame(
            history.history
        )

        history_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                (
                    f"fold_{fold_idx:02d}_"
                    f"{test_bearing}_history.csv"
                )
            ),
            index=False
        )

        best_epoch = (
            int(
                np.argmin(
                    history.history[
                        "val_loss"
                    ]
                )
            )
            + 1
        )

        best_val_loss = float(
            np.min(
                history.history[
                    "val_loss"
                ]
            )
        )

        validation_audit[-1][
            "Best_Epoch"
        ] = best_epoch

        validation_audit[-1][
            "Best_Val_Loss"
        ] = best_val_loss

        # =====================================================================
        # TEST PREDICTION
        # =====================================================================

        test_metadata = (
            metadata
            .iloc[
                test_indices
            ]
            .reset_index(
                drop=True
            )
        )

        recording_df = (
            predict_recordings(
                model,
                X[
                    test_indices
                ],
                test_metadata
            )
        )

        if len(
            recording_df
        ) != 80:

            raise RuntimeError(
                f"{test_bearing}: "
                f"expected 80 recording "
                f"predictions, "
                f"got {len(recording_df)}"
            )

        y_true_fold = (
            recording_df[
                "True_Label"
            ].values
        )

        y_pred_fold = (
            recording_df[
                "Predicted_Label"
            ].values
        )

        fold_metrics = (
            calculate_metrics(
                y_true_fold,
                y_pred_fold
            )
        )

        recording_df[
            "Fold"
        ] = fold_idx

        all_predictions.append(
            recording_df
        )

        # =====================================================================
        # STORE FOLD RESULT
        # =====================================================================

        fold_results.append(
            {
                "Fold": fold_idx,

                "Test_Bearing": (
                    test_bearing
                ),

                "True_Class": (
                    "Healthy"
                    if get_label(
                        test_bearing
                    ) == 0
                    else "Damaged"
                ),

                "Accuracy": (
                    fold_metrics[
                        "Accuracy"
                    ]
                ),

                "Precision": (
                    fold_metrics[
                        "Precision"
                    ]
                ),

                "Recall": (
                    fold_metrics[
                        "Recall"
                    ]
                ),

                "F1": (
                    fold_metrics[
                        "F1"
                    ]
                ),

                "TN": (
                    fold_metrics[
                        "TN"
                    ]
                ),

                "FP": (
                    fold_metrics[
                        "FP"
                    ]
                ),

                "FN": (
                    fold_metrics[
                        "FN"
                    ]
                ),

                "TP": (
                    fold_metrics[
                        "TP"
                    ]
                ),
            }
        )

        # =====================================================================
        # PRINT FOLD RESULTS
        # =====================================================================

        print()

        print(
            f"{test_bearing} RESULTS"
        )

        print(
            f"Accuracy : "
            f"{fold_metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{fold_metrics['Precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{fold_metrics['Recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{fold_metrics['F1']:.4f}"
        )

        print(
            f"TN={fold_metrics['TN']} "
            f"FP={fold_metrics['FP']} "
            f"FN={fold_metrics['FN']} "
            f"TP={fold_metrics['TP']}"
        )

        # =====================================================================
        # SAVE FOLD PREDICTIONS
        # =====================================================================

        recording_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                (
                    f"fold_{fold_idx:02d}_"
                    f"{test_bearing}_predictions.csv"
                )
            ),
            index=False
        )

        del model

        tf.keras.backend.clear_session()

    # =========================================================================
    # COMBINE ALL FOLDS
    # =========================================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    fold_results_df = (
        pd.DataFrame(
            fold_results
        )
    )

    validation_audit_df = (
        pd.DataFrame(
            validation_audit
        )
    )

    # =========================================================================
    # POOLED RECORDING-LEVEL METRICS
    # =========================================================================

    y_true = (
        predictions_df[
            "True_Label"
        ].values
    )

    y_pred = (
        predictions_df[
            "Predicted_Label"
        ].values
    )

    final_metrics = (
        calculate_metrics(
            y_true,
            y_pred
        )
    )

    # =========================================================================
    # BEARING-LEVEL RESULTS
    # =========================================================================

    bearing_results_df = (
        calculate_bearing_metrics(
            predictions_df
        )
    )

    # =========================================================================
    # HEALTHY-BEARING FPR
    # =========================================================================

    healthy_results = (
        bearing_results_df[
            bearing_results_df[
                "Bearing"
            ].isin(
                HEALTHY_BEARINGS
            )
        ]
    )

    mean_healthy_fpr = float(
        healthy_results[
            "Healthy_FPR"
        ].mean()
    )

    mean_healthy_specificity = (
        1.0
        - mean_healthy_fpr
    )

    # =========================================================================
    # BEARING-LEVEL F1 SUMMARY
    # =========================================================================

    bearing_f1 = (
        bearing_results_df[
            "F1"
        ]
        .astype(float)
        .values
    )

    bearing_f1_mean = float(
        np.mean(
            bearing_f1
        )
    )

    bearing_f1_sd = float(
        np.std(
            bearing_f1,
            ddof=1
        )
    )

    bearing_f1_median = float(
        np.median(
            bearing_f1
        )
    )

    bearing_f1_q1 = float(
        np.percentile(
            bearing_f1,
            25
        )
    )

    bearing_f1_q3 = float(
        np.percentile(
            bearing_f1,
            75
        )
    )

    # =========================================================================
    # COMPARISON
    # =========================================================================

    difference_vs_classical = (
        final_metrics[
            "F1"
        ]
        - FROZEN_PRIMARY_BASELINE_F1
    )

    difference_vs_original_cnn = (
        final_metrics[
            "F1"
        ]
        - ORIGINAL_STEP150_CNN_F1
    )

    difference_vs_original_gru = (
        final_metrics[
            "F1"
        ]
        - ORIGINAL_STEP151_CNN_GRU_F1
    )

    # =========================================================================
    # SAVE CSV FILES
    # =========================================================================

    predictions_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "step155_cnn_gru_recording_predictions.csv"
        ),
        index=False
    )

    bearing_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "step155_cnn_gru_bearing_results.csv"
        ),
        index=False
    )

    fold_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "step155_cnn_gru_fold_results.csv"
        ),
        index=False
    )

    validation_audit_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "step155_validation_audit.csv"
        ),
        index=False
    )

    # =========================================================================
    # JSON SUMMARY
    # =========================================================================

    summary = {

        "Step":
            "Step 155 - Corrected CNN-GRU",

        "TensorFlow_Version":
            tf.__version__,

        "Sampling_Frequency_Hz":
            FS,

        "Validated_Channel":
            "Y[6]/vibration_1",

        "Recordings":
            EXPECTED_RECORDINGS,

        "Bearings":
            len(BEARINGS),

        "Recordings_Per_Bearing":
            EXPECTED_RECORDINGS_PER_BEARING,

        "Window_Size":
            WINDOW_SIZE,

        "Windows_Per_Recording":
            N_WINDOWS_PER_RECORDING,

        "Chunk_Size":
            CHUNK_SIZE,

        "Chunks_Per_Window":
            N_CHUNKS,

        "Strict_LOBO":
            True,

        "Validation_Method":
            "Recording-level stratified 15% split",

        "Validation_Size":
            VALIDATION_SIZE,

        "Training_Recordings_Per_Fold":
            612,

        "Validation_Recordings_Per_Fold":
            108,

        "Test_Recordings_Per_Fold":
            80,

        "Training_Windows_Per_Fold":
            2448,

        "Validation_Windows_Per_Fold":
            432,

        "Test_Windows_Per_Fold":
            320,

        "Maximum_Epochs":
            EPOCHS,

        "Batch_Size":
            BATCH_SIZE,

        "Learning_Rate":
            LEARNING_RATE,

        "Early_Stopping_Patience":
            EARLY_STOPPING_PATIENCE,

        "Classical_Baseline_F1":
            FROZEN_PRIMARY_BASELINE_F1,

        "Original_Step150_CNN_F1":
            ORIGINAL_STEP150_CNN_F1,

        "Original_Step151_CNN_GRU_F1":
            ORIGINAL_STEP151_CNN_GRU_F1,

        "Corrected_CNN_GRU_Accuracy":
            final_metrics[
                "Accuracy"
            ],

        "Corrected_CNN_GRU_Precision":
            final_metrics[
                "Precision"
            ],

        "Corrected_CNN_GRU_Recall":
            final_metrics[
                "Recall"
            ],

        "Corrected_CNN_GRU_F1":
            final_metrics[
                "F1"
            ],

        "CNN_GRU_minus_Classical":
            difference_vs_classical,

        "CNN_GRU_minus_Original_CNN":
            difference_vs_original_cnn,

        "CNN_GRU_minus_Original_CNN_GRU":
            difference_vs_original_gru,

        "Mean_Healthy_Bearing_FPR":
            mean_healthy_fpr,

        "Mean_Healthy_Bearing_Specificity":
            mean_healthy_specificity,

        "Bearing_Level_F1_Mean":
            bearing_f1_mean,

        "Bearing_Level_F1_Sample_SD":
            bearing_f1_sd,

        "Bearing_Level_F1_Median":
            bearing_f1_median,

        "Bearing_Level_F1_Q1":
            bearing_f1_q1,

        "Bearing_Level_F1_Q3":
            bearing_f1_q3,

        "TN":
            final_metrics[
                "TN"
            ],

        "FP":
            final_metrics[
                "FP"
            ],

        "FN":
            final_metrics[
                "FN"
            ],

        "TP":
            final_metrics[
                "TP"
            ],
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "step155_cnn_gru_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================

    with open(
        os.path.join(
            OUTPUT_DIR,
            "step155_cnn_gru_summary.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "STEP 155 - CORRECTED CNN-GRU\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Validated channel: "
            "Y[6]/vibration_1\n"
        )

        f.write(
            "Sampling frequency: "
            "64000 Hz\n"
        )

        f.write(
            "Strict 10-bearing LOBO: "
            "YES\n"
        )

        f.write(
            "Validation: "
            "recording-level stratified 15%\n\n"
        )

        f.write(
            "FINAL POOLED RESULTS\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Accuracy : "
            f"{final_metrics['Accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{final_metrics['Precision']:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{final_metrics['Recall']:.4f}\n"
        )

        f.write(
            f"F1       : "
            f"{final_metrics['F1']:.4f}\n"
        )

        f.write(
            f"Healthy FPR: "
            f"{mean_healthy_fpr:.4f}\n"
        )

        f.write(
            f"Specificity: "
            f"{mean_healthy_specificity:.4f}\n"
        )

        f.write(
            f"TN={final_metrics['TN']} "
            f"FP={final_metrics['FP']} "
            f"FN={final_metrics['FN']} "
            f"TP={final_metrics['TP']}\n\n"
        )

        f.write(
            "COMPARISON\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Classical baseline: "
            f"{FROZEN_PRIMARY_BASELINE_F1:.4f}\n"
        )

        f.write(
            f"Original CNN: "
            f"{ORIGINAL_STEP150_CNN_F1:.4f}\n"
        )

        f.write(
            f"Original CNN-GRU: "
            f"{ORIGINAL_STEP151_CNN_GRU_F1:.4f}\n"
        )

        f.write(
            f"Corrected CNN-GRU: "
            f"{final_metrics['F1']:.4f}\n"
        )

        f.write(
            f"Corrected CNN-GRU minus classical: "
            f"{difference_vs_classical:+.4f}\n"
        )

        f.write(
            f"Corrected CNN-GRU minus original CNN-GRU: "
            f"{difference_vs_original_gru:+.4f}\n"
        )

    # =========================================================================
    # FINAL TERMINAL OUTPUT
    # =========================================================================

    print()

    print("=" * 80)

    print(
        "STEP 155 - FINAL CORRECTED CNN-GRU RESULTS"
    )

    print("=" * 80)

    print(
        f"Accuracy : "
        f"{final_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{final_metrics['Recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{final_metrics['F1']:.4f}"
    )

    print(
        f"Healthy FPR: "
        f"{mean_healthy_fpr:.4f}"
    )

    print(
        f"Specificity: "
        f"{mean_healthy_specificity:.4f}"
    )

    print()

    print(
        f"TN={final_metrics['TN']} "
        f"FP={final_metrics['FP']} "
        f"FN={final_metrics['FN']} "
        f"TP={final_metrics['TP']}"
    )

    print()

    print(
        f"Classical baseline F1: "
        f"{FROZEN_PRIMARY_BASELINE_F1:.4f}"
    )

    print(
        f"Original CNN-GRU F1: "
        f"{ORIGINAL_STEP151_CNN_GRU_F1:.4f}"
    )

    print(
        f"Corrected CNN-GRU F1: "
        f"{final_metrics['F1']:.4f}"
    )

    print(
        f"Difference vs classical: "
        f"{difference_vs_classical:+.4f}"
    )

    print(
        f"Difference vs original CNN-GRU: "
        f"{difference_vs_original_gru:+.4f}"
    )

    print()

    print("=" * 80)

    print(
        "BEARING-WISE RESULTS"
    )

    print("=" * 80)

    print(
        bearing_results_df[
            [
                "Bearing",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Healthy_FPR",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print("=" * 80)

    print(
        "BEARING-LEVEL F1 SUMMARY"
    )

    print("=" * 80)

    print(
        f"Mean:   "
        f"{bearing_f1_mean:.4f}"
    )

    print(
        f"SD:     "
        f"{bearing_f1_sd:.4f}"
    )

    print(
        f"Median: "
        f"{bearing_f1_median:.4f}"
    )

    print(
        f"IQR:    "
        f"{bearing_f1_q1:.4f} - "
        f"{bearing_f1_q3:.4f}"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("=" * 80)

    print(
        "STEP 155 COMPLETE"
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)

    print(
        "STEP 155 - CORRECTED CNN-GRU"
    )

    print(
        "STRICT 10-BEARING LOBO + "
        "RECORDING-LEVEL STRATIFIED VALIDATION"
    )

    print("=" * 80)

    print()

    print(
        "TensorFlow:",
        tf.__version__
    )

    print(
        "Nominal sampling rate:",
        FS,
        "Hz"
    )

    print(
        "Recording window:",
        WINDOW_SIZE,
        "samples"
    )

    print(
        "Windows per recording:",
        N_WINDOWS_PER_RECORDING
    )

    print(
        "Temporal chunk size:",
        CHUNK_SIZE,
        "samples"
    )

    print(
        "Temporal chunks per window:",
        N_CHUNKS
    )

    print(
        "Expected recordings:",
        EXPECTED_RECORDINGS
    )

    print(
        "Frozen classical baseline F1:",
        FROZEN_PRIMARY_BASELINE_F1
    )

    print(
        "Original Step 150 CNN F1:",
        ORIGINAL_STEP150_CNN_F1
    )

    print(
        "Original Step 151 CNN-GRU F1:",
        ORIGINAL_STEP151_CNN_GRU_F1
    )

    print()

    print(
        "Validated channel:"
    )

    print(
        "  Y[6] / vibration_1"
    )

    print()

    # -------------------------------------------------------------------------
    # LOAD FILE LIST
    # -------------------------------------------------------------------------

    records = discover_files()

    # -------------------------------------------------------------------------
    # LOAD AND PREPROCESS DATA
    # -------------------------------------------------------------------------

    (
        X,
        y,
        metadata
    ) = load_dataset(
        records
    )

    # -------------------------------------------------------------------------
    # RUN STRICT LOBO
    # -------------------------------------------------------------------------

    run_lobo(
        X,
        y,
        metadata
    )


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    main()