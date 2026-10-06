# =============================================================================
# STEP 156 - CORRECTED 1D-CNN WITH RECORDING-LEVEL STRATIFIED VALIDATION
# =============================================================================
#
# FINAL PUBLICATION-AUDIT VERSION
#
# Outer evaluation:
#   Strict 10-bearing Leave-One-Bearing-Out (LOBO)
#
# Inner validation:
#   15% recording-level stratified split
#   All 4 windows from each recording remain in the same partition
#
# Dataset:
#   800 recordings
#   10 physical bearings
#   80 recordings per bearing
#   480 healthy recordings
#   320 damaged recordings
#
# Signal:
#   Y[6] / vibration_1
#   Nominal sampling rate = 64 kHz
#
# Critical correction:
#   The original Keras validation_split=0.15 is NOT used.
#   Validation is explicitly split at the RECORDING level and stratified
#   by class. Therefore, all four windows from one recording remain together.
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    ReLU,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam


# =============================================================================
# 1. REPRODUCIBILITY
# =============================================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# =============================================================================
# 2. DATASET PATH
# =============================================================================
#
# YOUR ACTUAL PADERBORN DATASET LOCATION
#
# The dataset contains:
#
#   K001
#   K002
#   K003
#   K004
#   K005
#   K006
#   KA01
#   KA03
#   KA04
#   KI05
#
# =============================================================================

ROOT = os.environ.get("BEARING_DATA_ROOT", str(Path(__file__).resolve().parent))


# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "step156_1d_cnn_lobo_stratified_validation_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# 4. DATASET CONFIGURATION
# =============================================================================

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


EXPECTED_RECORDINGS = 800

EXPECTED_RECORDINGS_PER_BEARING = 80


# =============================================================================
# 5. SIGNAL CONFIGURATION
# =============================================================================

FS = 64000

WINDOW_SIZE = 64000

N_WINDOWS_PER_RECORDING = 4

REQUIRED_SAMPLES = (
    WINDOW_SIZE *
    N_WINDOWS_PER_RECORDING
)


# =============================================================================
# 6. CNN CONFIGURATION
# =============================================================================

EPOCHS = 30

BATCH_SIZE = 8

LEARNING_RATE = 1e-3

VALIDATION_FRACTION = 0.15

EARLY_STOPPING_PATIENCE = 5


# =============================================================================
# 7. FROZEN CLASSICAL BASELINE
# =============================================================================

FROZEN_BASELINE_F1 = 0.6159


# =============================================================================
# 8. WARNINGS
# =============================================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# =============================================================================
# 9. HEADER
# =============================================================================

print("=" * 80)

print(
    "STEP 156 - CORRECTED 1D-CNN"
)

print(
    "RECORDING-LEVEL STRATIFIED VALIDATION + STRICT 10-BEARING LOBO"
)

print("=" * 80)

print()

print(
    "TensorFlow:",
    tf.__version__
)

print(
    "Seed:",
    SEED
)

print(
    "Dataset root:",
    ROOT
)

print(
    "Validated channel: Y[6] / vibration_1"
)

print(
    "Nominal sampling rate:",
    FS,
    "Hz"
)

print(
    "Window size:",
    WINDOW_SIZE
)

print(
    "Windows per recording:",
    N_WINDOWS_PER_RECORDING
)

print(
    "Required samples:",
    REQUIRED_SAMPLES
)

print(
    "Expected recordings:",
    EXPECTED_RECORDINGS
)

print(
    "Maximum epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Internal validation:",
    VALIDATION_FRACTION
)

print(
    "Early stopping patience:",
    EARLY_STOPPING_PATIENCE
)

print(
    "Frozen classical F1:",
    FROZEN_BASELINE_F1
)

print()


# =============================================================================
# 10. LABEL FUNCTION
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
# 11. DISCOVER PADERBORN FILES
# =============================================================================
#
# IMPORTANT:
#
# We intentionally search ONLY inside the ten bearing directories.
#
# This prevents unrelated .mat files inside:
#
#   signal\dl_env\
#
# from being included.
#
# =============================================================================

def discover_files():

    print("=" * 80)

    print(
        "DISCOVERING PADERBORN MATLAB FILES"
    )

    print("=" * 80)

    records = []

    for bearing in BEARINGS:

        bearing_root = os.path.join(
            ROOT,
            bearing
        )

        if not os.path.isdir(
            bearing_root
        ):

            raise RuntimeError(
                "Missing bearing directory:\n"
                f"{bearing_root}"
            )

        bearing_files = []

        for current_root, dirs, files in os.walk(
            bearing_root
        ):

            for filename in files:

                if not filename.lower().endswith(
                    ".mat"
                ):

                    continue

                path = os.path.join(
                    current_root,
                    filename
                )

                bearing_files.append(
                    path
                )

        bearing_files = sorted(
            bearing_files
        )

        print(
            f"{bearing}: "
            f"{len(bearing_files)} MAT files"
        )

        if len(bearing_files) != (
            EXPECTED_RECORDINGS_PER_BEARING
        ):

            raise RuntimeError(
                f"{bearing}: expected "
                f"{EXPECTED_RECORDINGS_PER_BEARING} recordings, "
                f"but found {len(bearing_files)}."
            )

        for path in bearing_files:

            records.append(
                {
                    "path": path,

                    "bearing": bearing,

                    "label": get_label(
                        bearing
                    ),

                    "filename":
                        os.path.basename(
                            path
                        ),
                }
            )

    records = sorted(
        records,
        key=lambda x: (
            BEARINGS.index(
                x["bearing"]
            ),
            x["filename"]
        )
    )

    print()

    print(
        "TOTAL PADERBORN RECORDINGS:",
        len(records)
    )

    print()

    if len(records) != EXPECTED_RECORDINGS:

        raise RuntimeError(
            f"Expected {EXPECTED_RECORDINGS} total recordings, "
            f"but found {len(records)}."
        )

    return records


# =============================================================================
# 12. CORRECT PADERBORN MAT LOADER
# =============================================================================
#
# VERIFIED STRUCTURE:
#
#   root
#      |
#      +-- Y
#           |
#           +-- Y[6]
#                 |
#                 +-- Name = vibration_1
#                 |
#                 +-- Data
#
# =============================================================================

def load_vibration_signal(path):

    mat = sio.loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False
    )

    root = None

    # -------------------------------------------------------------------------
    # Locate MATLAB root object containing Y
    # -------------------------------------------------------------------------

    for key, value in mat.items():

        if key.startswith(
            "__"
        ):

            continue

        if hasattr(
            value,
            "Y"
        ):

            root = value

            break

    if root is None:

        raise RuntimeError(
            "Could not locate root structure "
            "containing Y:\n"
            f"{path}"
        )

    # -------------------------------------------------------------------------
    # Get Y
    # -------------------------------------------------------------------------

    Y = np.atleast_1d(
        root.Y
    )

    if len(Y) <= 6:

        raise RuntimeError(
            "Y[6] unavailable in:\n"
            f"{path}"
        )

    # -------------------------------------------------------------------------
    # Correct vibration channel
    # -------------------------------------------------------------------------

    channel = Y[6]

    channel_name = getattr(
        channel,
        "Name",
        None
    )

    if isinstance(
        channel_name,
        np.ndarray
    ):

        if channel_name.size == 1:

            channel_name = (
                channel_name.item()
            )

    if isinstance(
        channel_name,
        bytes
    ):

        channel_name = (
            channel_name.decode(
                errors="ignore"
            )
        )

    channel_name = str(
        channel_name
    ).strip()

    if channel_name != "vibration_1":

        raise RuntimeError(
            f"Y[6] channel mismatch in {path}: "
            f"found '{channel_name}', "
            f"expected 'vibration_1'."
        )

    # -------------------------------------------------------------------------
    # Extract Data
    # -------------------------------------------------------------------------

    data = getattr(
        channel,
        "Data",
        None
    )

    if data is None:

        raise RuntimeError(
            "Data field missing from Y[6]:\n"
            f"{path}"
        )

    signal = np.asarray(
        data,
        dtype=np.float32
    ).squeeze()

    signal = signal.reshape(
        -1
    )

    # -------------------------------------------------------------------------
    # Finite-value check
    # -------------------------------------------------------------------------

    if not np.all(
        np.isfinite(
            signal
        )
    ):

        raise RuntimeError(
            "Non-finite signal values found:\n"
            f"{path}"
        )

    # -------------------------------------------------------------------------
    # Small length correction
    # -------------------------------------------------------------------------

    if len(signal) < REQUIRED_SAMPLES:

        missing = (
            REQUIRED_SAMPLES -
            len(signal)
        )

        if missing > 10:

            raise RuntimeError(
                f"Signal too short in {path}: "
                f"{len(signal)} samples; "
                f"missing {missing}."
            )

        signal = np.pad(
            signal,
            (
                0,
                missing
            ),
            mode="constant"
        )

    elif len(signal) > REQUIRED_SAMPLES:

        signal = signal[
            :REQUIRED_SAMPLES
        ]

    return signal


# =============================================================================
# 13. FOUR NON-OVERLAPPING WINDOWS
# =============================================================================

def extract_windows(signal):

    windows = []

    for window_id in range(
        N_WINDOWS_PER_RECORDING
    ):

        start = (
            window_id *
            WINDOW_SIZE
        )

        end = (
            start +
            WINDOW_SIZE
        )

        window = signal[
            start:end
        ]

        if len(window) != WINDOW_SIZE:

            raise RuntimeError(
                "Window length mismatch."
            )

        windows.append(
            window.astype(
                np.float32
            )
        )

    return windows


# =============================================================================
# 14. PER-WINDOW STANDARDIZATION
# =============================================================================

def standardize_window(window):

    mean = np.mean(
        window,
        dtype=np.float64
    )

    std = np.std(
        window,
        dtype=np.float64
    )

    if std < 1e-12:

        return (
            window -
            mean
        ).astype(
            np.float32
        )

    standardized = (
        window -
        mean
    ) / std

    return standardized.astype(
        np.float32
    )


# =============================================================================
# 15. LOAD COMPLETE DATASET
# =============================================================================

def load_dataset(records):

    print("=" * 80)

    print(
        "LOADING DATASET"
    )

    print("=" * 80)

    X = []

    y = []

    metadata = []

    for recording_index, record in enumerate(
        records
    ):

        signal = load_vibration_signal(
            record["path"]
        )

        windows = extract_windows(
            signal
        )

        for window_id, window in enumerate(
            windows
        ):

            standardized_window = (
                standardize_window(
                    window
                )
            )

            X.append(
                standardized_window
            )

            y.append(
                record["label"]
            )

            metadata.append(
                {
                    "recording_index":
                        recording_index,

                    "bearing":
                        record["bearing"],

                    "filename":
                        record["filename"],

                    "path":
                        record["path"],

                    "window_id":
                        window_id,

                    "True_Label":
                        record["label"],
                }
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

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Healthy windows:",
        np.sum(
            y == 0
        )
    )

    print(
        "Damaged windows:",
        np.sum(
            y == 1
        )
    )

    print()

    expected_windows = (
        EXPECTED_RECORDINGS *
        N_WINDOWS_PER_RECORDING
    )

    if X.shape[0] != expected_windows:

        raise RuntimeError(
            f"Expected {expected_windows} windows, "
            f"got {X.shape[0]}."
        )

    return (
        X,
        y,
        metadata_df
    )


# =============================================================================
# 16. ORIGINAL STEP-150 1D-CNN ARCHITECTURE
# =============================================================================

def build_model():

    model = Sequential(
        [
            Input(
                shape=(
                    WINDOW_SIZE,
                    1
                )
            ),

            # -----------------------------------------------------------------
            # Block 1
            # -----------------------------------------------------------------

            Conv1D(
                filters=16,
                kernel_size=11,
                padding="same"
            ),

            BatchNormalization(),

            ReLU(),

            MaxPooling1D(
                pool_size=8
            ),

            # -----------------------------------------------------------------
            # Block 2
            # -----------------------------------------------------------------

            Conv1D(
                filters=32,
                kernel_size=11,
                padding="same"
            ),

            BatchNormalization(),

            ReLU(),

            MaxPooling1D(
                pool_size=8
            ),

            # -----------------------------------------------------------------
            # Block 3
            # -----------------------------------------------------------------

            Conv1D(
                filters=64,
                kernel_size=11,
                padding="same"
            ),

            BatchNormalization(),

            ReLU(),

            # -----------------------------------------------------------------
            # Global pooling
            # -----------------------------------------------------------------

            GlobalAveragePooling1D(),

            # -----------------------------------------------------------------
            # Classifier
            # -----------------------------------------------------------------

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
# 17. RECORDING-LEVEL STRATIFIED VALIDATION SPLIT
# =============================================================================

def make_internal_split(
    outer_train_recordings,
    metadata_df
):

    outer_train_recordings = np.asarray(
        outer_train_recordings,
        dtype=int
    )

    recording_labels = []

    # -------------------------------------------------------------------------
    # Get ONE label per recording
    # -------------------------------------------------------------------------

    for recording_index in (
        outer_train_recordings
    ):

        rows = metadata_df[
            metadata_df[
                "recording_index"
            ]
            == recording_index
        ]

        labels = rows[
            "True_Label"
        ].unique()

        if len(labels) != 1:

            raise RuntimeError(
                f"Inconsistent label for recording "
                f"{recording_index}."
            )

        recording_labels.append(
            int(
                labels[0]
            )
        )

    recording_labels = np.asarray(
        recording_labels,
        dtype=np.int32
    )

    # -------------------------------------------------------------------------
    # RECORDING-LEVEL stratified split
    # -------------------------------------------------------------------------

    train_recordings, val_recordings = (
        train_test_split(
            outer_train_recordings,

            test_size=VALIDATION_FRACTION,

            random_state=SEED,

            shuffle=True,

            stratify=recording_labels
        )
    )

    train_recordings = np.asarray(
        train_recordings,
        dtype=int
    )

    val_recordings = np.asarray(
        val_recordings,
        dtype=int
    )

    # -------------------------------------------------------------------------
    # Convert recording IDs to window indices
    # -------------------------------------------------------------------------

    train_window_indices = (
        metadata_df[
            metadata_df[
                "recording_index"
            ].isin(
                train_recordings
            )
        ]
        .index
        .to_numpy()
    )

    val_window_indices = (
        metadata_df[
            metadata_df[
                "recording_index"
            ].isin(
                val_recordings
            )
        ]
        .index
        .to_numpy()
    )

    # -------------------------------------------------------------------------
    # Exact expected sizes
    # -------------------------------------------------------------------------

    if len(train_recordings) != 612:

        raise RuntimeError(
            f"Expected 612 training recordings; "
            f"got {len(train_recordings)}."
        )

    if len(val_recordings) != 108:

        raise RuntimeError(
            f"Expected 108 validation recordings; "
            f"got {len(val_recordings)}."
        )

    if len(train_window_indices) != 2448:

        raise RuntimeError(
            f"Expected 2448 training windows; "
            f"got {len(train_window_indices)}."
        )

    if len(val_window_indices) != 432:

        raise RuntimeError(
            f"Expected 432 validation windows; "
            f"got {len(val_window_indices)}."
        )

    # -------------------------------------------------------------------------
    # No recording overlap
    # -------------------------------------------------------------------------

    overlap = set(
        train_recordings
    ).intersection(
        set(val_recordings)
    )

    if overlap:

        raise RuntimeError(
            f"Recording overlap detected: {overlap}"
        )

    # -------------------------------------------------------------------------
    # Both classes must be present
    # -------------------------------------------------------------------------

    train_labels = metadata_df.loc[
        train_window_indices,
        "True_Label"
    ].to_numpy()

    val_labels = metadata_df.loc[
        val_window_indices,
        "True_Label"
    ].to_numpy()

    if len(
        np.unique(
            train_labels
        )
    ) != 2:

        raise RuntimeError(
            "Internal training set does not "
            "contain both classes."
        )

    if len(
        np.unique(
            val_labels
        )
    ) != 2:

        raise RuntimeError(
            "Internal validation set does not "
            "contain both classes."
        )

    return (
        train_window_indices,
        val_window_indices,
        train_recordings,
        val_recordings
    )


# =============================================================================
# 18. RECORDING-LEVEL PREDICTION
# =============================================================================

def aggregate_recording_predictions(
    model,
    X_test,
    metadata_test
):

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    ).reshape(
        -1
    )

    temp = metadata_test.copy()

    temp[
        "Probability"
    ] = probabilities

    rows = []

    grouped = temp.groupby(
        [
            "bearing",
            "recording_index",
            "filename",
            "True_Label"
        ],
        sort=False
    )

    for (
        bearing,
        recording_index,
        filename,
        true_label
    ), group in grouped:

        probability = group[
            "Probability"
        ].mean()

        prediction = int(
            probability >= 0.5
        )

        rows.append(
            {
                "Bearing":
                    bearing,

                "recording_index":
                    recording_index,

                "filename":
                    filename,

                "True_Label":
                    int(
                        true_label
                    ),

                "Probability":
                    float(
                        probability
                    ),

                "Prediction":
                    prediction,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 19. METRIC CALCULATION
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        )
        .ravel()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    healthy_fpr = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "Accuracy":
            float(
                accuracy
            ),

        "Precision":
            float(
                precision
            ),

        "Recall":
            float(
                recall
            ),

        "F1":
            float(
                f1
            ),

        "Healthy_FPR":
            float(
                healthy_fpr
            ),

        "Specificity":
            float(
                specificity
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            ),
    }


# =============================================================================
# 20. MAIN EXPERIMENT
# =============================================================================

def main():

    # =========================================================================
    # DISCOVER DATA
    # =========================================================================

    records = discover_files()

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    X, y, metadata_df = load_dataset(
        records
    )

    # Add channel dimension:
    #
    # (3200, 64000)
    #
    # ->
    #
    # (3200, 64000, 1)

    X = X[
        ...,
        np.newaxis
    ]

    all_predictions = []

    fold_results = []

    # =========================================================================
    # STRICT 10-BEARING LOBO
    # =========================================================================

    for test_bearing in BEARINGS:

        print()

        print("=" * 80)

        print(
            f"LOBO TEST BEARING: {test_bearing}"
        )

        print("=" * 80)

        # ---------------------------------------------------------------------
        # OUTER TEST MASK
        # ---------------------------------------------------------------------

        test_mask = (
            metadata_df[
                "bearing"
            ]
            == test_bearing
        ).to_numpy()

        train_mask = ~test_mask

        outer_train_window_indices = (
            np.where(
                train_mask
            )[0]
        )

        test_window_indices = (
            np.where(
                test_mask
            )[0]
        )

        # ---------------------------------------------------------------------
        # Recording IDs
        # ---------------------------------------------------------------------

        outer_train_recordings = (
            metadata_df.loc[
                outer_train_window_indices,
                "recording_index"
            ]
            .unique()
        )

        test_recordings = (
            metadata_df.loc[
                test_window_indices,
                "recording_index"
            ]
            .unique()
        )

        print(
            "Outer training recordings:",
            len(
                outer_train_recordings
            )
        )

        print(
            "Test recordings:",
            len(
                test_recordings
            )
        )

        print(
            "Outer training windows:",
            len(
                outer_train_window_indices
            )
        )

        print(
            "Test windows:",
            len(
                test_window_indices
            )
        )

        # ---------------------------------------------------------------------
        # Exact LOBO checks
        # ---------------------------------------------------------------------

        if len(
            outer_train_recordings
        ) != 720:

            raise RuntimeError(
                "Outer training pool must contain "
                "720 recordings."
            )

        if len(
            test_recordings
        ) != 80:

            raise RuntimeError(
                "Held-out bearing must contain "
                "80 recordings."
            )

        # ---------------------------------------------------------------------
        # CORRECTED INTERNAL VALIDATION
        # ---------------------------------------------------------------------

        (
            train_indices,
            val_indices,
            train_recordings,
            val_recordings
        ) = make_internal_split(
            outer_train_recordings,
            metadata_df
        )

        print()

        print(
            "Internal training recordings:",
            len(
                train_recordings
            )
        )

        print(
            "Internal validation recordings:",
            len(
                val_recordings
            )
        )

        print(
            "Internal training windows:",
            len(
                train_indices
            )
        )

        print(
            "Internal validation windows:",
            len(
                val_indices
            )
        )

        # ---------------------------------------------------------------------
        # CLASS COUNTS
        # ---------------------------------------------------------------------

        train_y = y[
            train_indices
        ]

        val_y = y[
            val_indices
        ]

        print()

        print(
            "Internal training healthy:",
            np.sum(
                train_y == 0
            )
        )

        print(
            "Internal training damaged:",
            np.sum(
                train_y == 1
            )
        )

        print(
            "Internal validation healthy:",
            np.sum(
                val_y == 0
            )
        )

        print(
            "Internal validation damaged:",
            np.sum(
                val_y == 1
            )
        )

        # ---------------------------------------------------------------------
        # BEARING LEAKAGE CHECK
        # ---------------------------------------------------------------------

        train_bearing_set = set(
            metadata_df.loc[
                train_indices,
                "bearing"
            ]
        )

        val_bearing_set = set(
            metadata_df.loc[
                val_indices,
                "bearing"
            ]
        )

        if test_bearing in (
            train_bearing_set
        ):

            raise RuntimeError(
                "TEST BEARING LEAKAGE INTO TRAINING!"
            )

        if test_bearing in (
            val_bearing_set
        ):

            raise RuntimeError(
                "TEST BEARING LEAKAGE INTO VALIDATION!"
            )

        # ---------------------------------------------------------------------
        # RECORDING OVERLAP CHECK
        # ---------------------------------------------------------------------

        recording_overlap = set(
            train_recordings
        ).intersection(
            set(val_recordings)
        )

        if recording_overlap:

            raise RuntimeError(
                "TRAINING/VALIDATION RECORDING OVERLAP!"
            )

        # ---------------------------------------------------------------------
        # FRESH MODEL
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------------

        early_stopping = EarlyStopping(
            monitor="val_loss",

            patience=(
                EARLY_STOPPING_PATIENCE
            ),

            restore_best_weights=True,

            verbose=1
        )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # validation_split is NOT used.
        #
        # Explicit validation_data is used instead.
        #
        # ---------------------------------------------------------------------

        history = model.fit(

            X[
                train_indices
            ],

            y[
                train_indices
            ],

            validation_data=(

                X[
                    val_indices
                ],

                y[
                    val_indices
                ]

            ),

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            shuffle=True,

            callbacks=[
                early_stopping
            ],

            verbose=1
        )

        # ---------------------------------------------------------------------
        # HELD-OUT TEST BEARING
        # ---------------------------------------------------------------------

        test_metadata = (
            metadata_df.loc[
                test_window_indices
            ].copy()
        )

        predictions = (
            aggregate_recording_predictions(
                model,
                X[
                    test_window_indices
                ],
                test_metadata
            )
        )

        # ---------------------------------------------------------------------
        # FOLD METRICS
        # ---------------------------------------------------------------------

        metrics = calculate_metrics(

            predictions[
                "True_Label"
            ].to_numpy(),

            predictions[
                "Prediction"
            ].to_numpy()

        )

        metrics[
            "Bearing"
        ] = test_bearing

        fold_results.append(
            metrics
        )

        predictions[
            "Test_Bearing"
        ] = test_bearing

        all_predictions.append(
            predictions
        )

        # ---------------------------------------------------------------------
        # PRINT FOLD RESULTS
        # ---------------------------------------------------------------------

        print()

        print(
            f"{test_bearing} RESULTS"
        )

        print(
            "Accuracy:",
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            "Precision:",
            f"{metrics['Precision']:.4f}"
        )

        print(
            "Recall:",
            f"{metrics['Recall']:.4f}"
        )

        print(
            "F1:",
            f"{metrics['F1']:.4f}"
        )

        print(
            "Healthy FPR:",
            f"{metrics['Healthy_FPR']:.4f}"
        )

        print(
            "Specificity:",
            f"{metrics['Specificity']:.4f}"
        )

        print(
            "TN:",
            metrics["TN"],

            "FP:",
            metrics["FP"],

            "FN:",
            metrics["FN"],

            "TP:",
            metrics["TP"]
        )

        # ---------------------------------------------------------------------
        # SAVE FOLD PREDICTIONS
        # ---------------------------------------------------------------------

        predictions.to_csv(

            os.path.join(
                OUTPUT_DIR,
                f"predictions_{test_bearing}.csv"
            ),

            index=False
        )

        # ---------------------------------------------------------------------
        # SAVE HISTORY
        # ---------------------------------------------------------------------

        pd.DataFrame(
            history.history
        ).to_csv(

            os.path.join(
                OUTPUT_DIR,
                f"history_{test_bearing}.csv"
            ),

            index=False
        )

    # =========================================================================
    # POOLED RESULTS
    # =========================================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    y_true = (
        predictions_df[
            "True_Label"
        ]
        .to_numpy()
    )

    y_pred = (
        predictions_df[
            "Prediction"
        ]
        .to_numpy()
    )

    final_metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print()

    print("=" * 80)

    print(
        "STEP 156 FINAL POOLED RESULTS"
    )

    print("=" * 80)

    print(
        "Accuracy:",
        f"{final_metrics['Accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{final_metrics['Precision']:.4f}"
    )

    print(
        "Recall:",
        f"{final_metrics['Recall']:.4f}"
    )

    print(
        "F1:",
        f"{final_metrics['F1']:.4f}"
    )

    print(
        "Healthy FPR:",
        f"{final_metrics['Healthy_FPR']:.4f}"
    )

    print(
        "Specificity:",
        f"{final_metrics['Specificity']:.4f}"
    )

    print()

    print(
        "TN:",
        final_metrics["TN"]
    )

    print(
        "FP:",
        final_metrics["FP"]
    )

    print(
        "FN:",
        final_metrics["FN"]
    )

    print(
        "TP:",
        final_metrics["TP"]
    )

    # =========================================================================
    # BEARING-LEVEL RESULTS
    # =========================================================================

    bearing_results = pd.DataFrame(
        fold_results
    )

    bearing_results = bearing_results[
        [
            "Bearing",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Healthy_FPR",
            "Specificity",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]

    print()

    print("=" * 80)

    print(
        "BEARING-LEVEL RESULTS"
    )

    print("=" * 80)

    print(
        bearing_results.to_string(
            index=False
        )
    )

    # =========================================================================
    # BEARING-LEVEL F1 STATISTICS
    # =========================================================================

    f1_values = (
        bearing_results[
            "F1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    bearing_mean = np.mean(
        f1_values
    )

    bearing_sd = np.std(
        f1_values,
        ddof=1
    )

    bearing_median = np.median(
        f1_values
    )

    q1 = np.percentile(
        f1_values,
        25
    )

    q3 = np.percentile(
        f1_values,
        75
    )

    bearing_iqr = (
        q3 -
        q1
    )

    print()

    print("=" * 80)

    print(
        "BEARING-LEVEL F1 SUMMARY"
    )

    print("=" * 80)

    print(
        "Mean:",
        f"{bearing_mean:.4f}"
    )

    print(
        "Sample SD:",
        f"{bearing_sd:.4f}"
    )

    print(
        "Median:",
        f"{bearing_median:.4f}"
    )

    print(
        "IQR:",
        f"{q1:.4f} - {q3:.4f}"
    )

    print(
        "Minimum:",
        f"{np.min(f1_values):.4f}"
    )

    print(
        "Maximum:",
        f"{np.max(f1_values):.4f}"
    )

    # =========================================================================
    # CLASSICAL COMPARISON
    # =========================================================================

    cnn_minus_classical = (
        final_metrics[
            "F1"
        ]
        -
        FROZEN_BASELINE_F1
    )

    print()

    print("=" * 80)

    print(
        "COMPARISON WITH FROZEN CLASSICAL BASELINE"
    )

    print("=" * 80)

    print(
        "Classical F1:",
        f"{FROZEN_BASELINE_F1:.4f}"
    )

    print(
        "Corrected CNN F1:",
        f"{final_metrics['F1']:.4f}"
    )

    print(
        "CNN - Classical:",
        f"{cnn_minus_classical:+.4f}"
    )

    # =========================================================================
    # SAVE POOLED PREDICTIONS
    # =========================================================================

    predictions_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "step156_cnn_recording_predictions.csv"
        ),

        index=False
    )

    # =========================================================================
    # SAVE BEARING RESULTS
    # =========================================================================

    bearing_results.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "step156_cnn_bearing_results.csv"
        ),

        index=False
    )

    # =========================================================================
    # SUMMARY JSON
    # =========================================================================

    summary = {

        "Step":
            "Step 156 - Corrected 1D-CNN",

        "Dataset_Root":
            ROOT,

        "Validated_Channel":
            "Y[6]/vibration_1",

        "Sampling_Frequency_Hz":
            FS,

        "Number_of_Bearings":
            10,

        "Number_of_Recordings":
            800,

        "Recordings_Per_Bearing":
            80,

        "Window_Size_Samples":
            WINDOW_SIZE,

        "Windows_Per_Recording":
            N_WINDOWS_PER_RECORDING,

        "Strict_LOBO":
            True,

        "Internal_Validation":
            "Recording-level stratified",

        "Validation_Fraction":
            VALIDATION_FRACTION,

        "Outer_Training_Recordings":
            720,

        "Internal_Training_Recordings":
            612,

        "Internal_Validation_Recordings":
            108,

        "Internal_Training_Windows":
            2448,

        "Internal_Validation_Windows":
            432,

        "Test_Recordings_Per_Fold":
            80,

        "Test_Windows_Per_Fold":
            320,

        "Epochs_Max":
            EPOCHS,

        "Batch_Size":
            BATCH_SIZE,

        "Learning_Rate":
            LEARNING_RATE,

        "Early_Stopping_Patience":
            EARLY_STOPPING_PATIENCE,

        "Frozen_Classical_F1":
            FROZEN_BASELINE_F1,

        "CNN_Accuracy":
            final_metrics[
                "Accuracy"
            ],

        "CNN_Precision":
            final_metrics[
                "Precision"
            ],

        "CNN_Recall":
            final_metrics[
                "Recall"
            ],

        "CNN_F1":
            final_metrics[
                "F1"
            ],

        "CNN_Healthy_FPR":
            final_metrics[
                "Healthy_FPR"
            ],

        "CNN_Specificity":
            final_metrics[
                "Specificity"
            ],

        "CNN_minus_Classical_F1":
            cnn_minus_classical,

        "Bearing_Level_F1_Mean":
            bearing_mean,

        "Bearing_Level_F1_Sample_SD":
            bearing_sd,

        "Bearing_Level_F1_Median":
            bearing_median,

        "Bearing_Level_F1_Q1":
            q1,

        "Bearing_Level_F1_Q3":
            q3,

        "Bearing_Level_F1_IQR":
            bearing_iqr,

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
            "step156_summary.json"
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
    # FINAL MESSAGE
    # =========================================================================

    print()

    print("=" * 80)

    print(
        "STEP 156 COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        "Protocol:"
    )

    print(
        "  Strict 10-bearing LOBO"
    )

    print(
        "  Recording-level stratified validation"
    )

    print(
        "  15% internal validation"
    )

    print(
        "  612 training recordings"
    )

    print(
        "  108 validation recordings"
    )

    print(
        "  80 held-out test recordings/fold"
    )

    print(
        "  Four windows/recording"
    )

    print(
        "  Correct channel: Y[6]/vibration_1"
    )

    print(
        "  Fresh CNN for every LOBO fold"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# =============================================================================
# 21. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()