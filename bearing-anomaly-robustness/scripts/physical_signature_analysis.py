# ============================================================
# STEP 155 - PHYSICAL SIGNATURE DIAGNOSTIC ANALYSIS
# ============================================================
#
# Purpose:
#   Compare K002 (healthy), KA03 (damaged), and KI05 (damaged)
#   using time-domain, spectral, envelope, order-band, and
#   noise-floor features.
#
# IMPORTANT:
#   - Diagnostic experiment only.
#   - Does NOT modify frozen primary results.
#   - Uses Paderborn Y[6] / vibration_1.
#   - Nominal sampling rate = 64 kHz.
#   - Strict LOBO is NOT used as the primary objective here;
#     this is a physical-signature diagnostic.
#
# ============================================================

import os
from pathlib import Path
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy import signal
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(os.environ.get("BEARING_DATA_ROOT", Path(__file__).resolve().parent))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "step155_physical_signature_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 64000.0

TARGET_BEARINGS = [
    "K002",
    "KA03",
    "KI05"
]

HEALTHY_BEARINGS = [
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006"
]

DAMAGED_BEARINGS = [
    "KA01",
    "KA03",
    "KA04",
    "KI05"
]

ALL_BEARINGS = HEALTHY_BEARINGS + DAMAGED_BEARINGS


CONDITIONS = {
    "N09_M07_F10": 900,
    "N15_M01_F10": 1500,
    "N15_M07_F04": 1500,
    "N15_M07_F10": 1500,
}


# ============================================================
# FEATURE SETTINGS
# ============================================================

SPECTRAL_BANDS = [
    ("0_1k", 0, 1000),
    ("1_3k", 1000, 3000),
    ("3_5k", 3000, 5000),
    ("5_10k", 5000, 10000),
    ("10_20k", 10000, 20000),
    ("20_32k", 20000, 32000),
]

ORDER_BANDS = [
    ("0.5_2", 0.5, 2),
    ("2_5", 2, 5),
    ("5_10", 5, 10),
    ("10_20", 10, 20),
    ("20_40", 20, 40),
]

ENVELOPE_LOW = 500.0
ENVELOPE_HIGH = 10000.0


# ============================================================
# MATLAB STRUCTURE LOADER
# ============================================================

def get_struct(mat):
    """
    Extract the actual MATLAB recording structure.

    Paderborn .mat files contain the recording as a 1x1
    NumPy object array. The actual MATLAB struct is stored
    inside that array.
    """

    keys = [
        k for k in mat.keys()
        if not k.startswith("__")
    ]

    if not keys:
        raise RuntimeError(
            "No MATLAB data structure found."
        )

    root = mat[keys[0]]

    # IMPORTANT FIX:
    # MATLAB structure is stored inside a 1x1 ndarray.
    if isinstance(root, np.ndarray):

        if root.size != 1:
            raise RuntimeError(
                f"Unexpected MATLAB structure shape: "
                f"{root.shape}"
            )

        root = root.flat[0]

    return root


# ============================================================
# VIBRATION EXTRACTION
# ============================================================

def extract_vibration(path):

    mat = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False
    )

    main = get_struct(mat)

    if not hasattr(main, "Y"):
        raise RuntimeError(
            "MATLAB recording does not contain Y field."
        )

    Y = main.Y

    if Y.size < 7:
        raise RuntimeError(
            f"Y contains only {Y.size} channels; "
            "Y[6] is unavailable."
        )

    # Required vibration channel:
    # Y[6] / vibration_1
    channel = Y.flat[6]

    if not hasattr(channel, "Data"):
        raise RuntimeError(
            "Y[6] does not contain Data field."
        )

    data = np.asarray(
        channel.Data,
        dtype=np.float64
    ).squeeze()

    if data.ndim != 1:
        data = data.ravel()

    if data.size == 0:
        raise RuntimeError(
            "Extracted vibration signal is empty."
        )

    # Remove NaN / Inf values
    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return data


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_bearing_files(bearing):

    bearing_dir = os.path.join(
        BASE_DIR,
        bearing
    )

    if not os.path.isdir(bearing_dir):
        raise RuntimeError(
            f"Bearing directory not found: "
            f"{bearing_dir}"
        )

    files = sorted(
        glob.glob(
            os.path.join(
                bearing_dir,
                "*.mat"
            )
        )
    )

    return files


# ============================================================
# BASIC STATISTICS
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(
            np.square(x)
        )
    )


def safe_kurtosis(x):

    mean = np.mean(x)

    centered = x - mean

    variance = np.mean(
        centered ** 2
    )

    if variance <= 0:
        return 0.0

    fourth = np.mean(
        centered ** 4
    )

    return fourth / (
        variance ** 2
    )


def crest_factor(x):

    r = rms(x)

    if r <= 0:
        return 0.0

    return np.max(
        np.abs(x)
    ) / r


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def extract_time_features(x):

    r = rms(x)

    std = np.std(x)

    peak = np.max(
        np.abs(x)
    )

    p2p = (
        np.max(x)
        - np.min(x)
    )

    kurt = safe_kurtosis(x)

    cf = crest_factor(x)

    return {
        "RMS": r,
        "STD": std,
        "Peak": peak,
        "Peak_to_Peak": p2p,
        "Kurtosis": kurt,
        "Crest_Factor": cf,
    }


# ============================================================
# WELCH SPECTRUM
# ============================================================

def compute_welch(x):

    nperseg = min(
        8192,
        len(x)
    )

    if nperseg < 256:
        nperseg = len(x)

    frequencies, power = signal.welch(
        x,
        fs=FS,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density"
    )

    return frequencies, power


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def extract_spectral_features(x):

    frequencies, power = compute_welch(x)

    features = {}

    total_power = np.trapezoid(
        power,
        frequencies
    )

    features[
        "Spectral_Total_Power"
    ] = total_power

    # Spectral bands
    band_values = {}

    for name, low, high in SPECTRAL_BANDS:

        mask = (
            (frequencies >= low)
            &
            (frequencies < high)
        )

        if np.any(mask):

            value = np.trapezoid(
                power[mask],
                frequencies[mask]
            )

        else:

            value = 0.0

        band_values[name] = value

        features[
            f"Spectral_Energy_{name}"
        ] = value

        if total_power > 0:

            features[
                f"Spectral_Ratio_{name}"
            ] = value / total_power

        else:

            features[
                f"Spectral_Ratio_{name}"
            ] = 0.0

    # Dominant frequency
    if len(power) > 0:

        idx = np.argmax(power)

        features[
            "Dominant_Frequency_Hz"
        ] = frequencies[idx]

        features[
            "Dominant_Power"
        ] = power[idx]

    else:

        features[
            "Dominant_Frequency_Hz"
        ] = 0.0

        features[
            "Dominant_Power"
        ] = 0.0

    return features


# ============================================================
# ENVELOPE EXTRACTION
# ============================================================

def extract_envelope(x):

    nyquist = FS / 2.0

    low = ENVELOPE_LOW / nyquist
    high = ENVELOPE_HIGH / nyquist

    # Ensure valid normalized cutoff
    low = max(
        low,
        0.001
    )

    high = min(
        high,
        0.999
    )

    if low >= high:

        raise RuntimeError(
            "Invalid envelope filter limits."
        )

    b, a = signal.butter(
        4,
        [low, high],
        btype="bandpass"
    )

    filtered = signal.filtfilt(
        b,
        a,
        x
    )

    analytic = signal.hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
    )

    return envelope


# ============================================================
# ENVELOPE FEATURES
# ============================================================

def extract_envelope_features(x):

    envelope = extract_envelope(x)

    features = {}

    features[
        "Envelope_RMS"
    ] = rms(envelope)

    features[
        "Envelope_STD"
    ] = np.std(envelope)

    features[
        "Envelope_Peak"
    ] = np.max(envelope)

    features[
        "Envelope_Kurtosis"
    ] = safe_kurtosis(
        envelope
    )

    features[
        "Envelope_Crest_Factor"
    ] = crest_factor(
        envelope
    )

    frequencies, power = compute_welch(
        envelope
    )

    total_power = np.trapezoid(
        power,
        frequencies
    )

    features[
        "Envelope_Total_Power"
    ] = total_power

    # Envelope spectrum in order bands
    for name, low_order, high_order in ORDER_BANDS:

        # Convert order to Hz
        #
        # Actual shaft speed is supplied later by the
        # condition-specific calculation. Here the
        # envelope spectrum is stored independently.
        #
        # For this function, use frequency bands that
        # correspond to a normalized diagnostic range.
        #
        # The actual order-band features are computed
        # separately in extract_order_features().

        pass

    return features, envelope


# ============================================================
# ORDER-BAND FEATURES
# ============================================================

def extract_order_features(
    envelope,
    rpm
):

    frequencies, power = signal.welch(
        envelope,
        fs=FS,
        window="hann",
        nperseg=min(
            8192,
            len(envelope)
        ),
        noverlap=min(
            8192,
            len(envelope)
        ) // 2,
        scaling="density"
    )

    shaft_frequency = rpm / 60.0

    features = {}

    total_power = np.trapezoid(
        power,
        frequencies
    )

    features[
        "Envelope_Total_Power"
    ] = total_power

    for name, low_order, high_order in ORDER_BANDS:

        low_hz = (
            low_order
            * shaft_frequency
        )

        high_hz = (
            high_order
            * shaft_frequency
        )

        mask = (
            (frequencies >= low_hz)
            &
            (frequencies < high_hz)
        )

        if np.any(mask):

            energy = np.trapezoid(
                power[mask],
                frequencies[mask]
            )

        else:

            energy = 0.0

        features[
            f"Order_{name}_Energy"
        ] = energy

        if total_power > 0:

            features[
                f"Order_{name}_Ratio"
            ] = (
                energy
                / total_power
            )

        else:

            features[
                f"Order_{name}_Ratio"
            ] = 0.0

    return features


# ============================================================
# NOISE-FLOOR FEATURES
# ============================================================

def extract_noise_features(x):

    frequencies, power = compute_welch(x)

    total_power = np.trapezoid(
        power,
        frequencies
    )

    if len(power) > 0:

        median_power = np.median(
            power
        )

        peak_power = np.max(
            power
        )

    else:

        median_power = 0.0
        peak_power = 0.0

    if median_power > 0:

        peak_to_median = (
            peak_power
            / median_power
        )

    else:

        peak_to_median = 0.0

    return {
        "Noise_Total_Spectral_Power":
            total_power,

        "Noise_Median_Spectral_Power":
            median_power,

        "Noise_Peak_to_Median":
            peak_to_median,
    }


# ============================================================
# RECORDING METADATA
# ============================================================

def parse_condition(filename):

    basename = os.path.basename(
        filename
    )

    for condition, rpm in CONDITIONS.items():

        if condition in basename:

            return condition, rpm

    return "UNKNOWN", np.nan


def get_label(bearing):

    if bearing in HEALTHY_BEARINGS:

        return 0

    if bearing in DAMAGED_BEARINGS:

        return 1

    return np.nan


# ============================================================
# SINGLE RECORDING ANALYSIS
# ============================================================

def analyze_recording(
    path,
    bearing
):

    condition, rpm = parse_condition(
        path
    )

    x = extract_vibration(
        path
    )

    features = {}

    features["Bearing"] = bearing
    features["Condition"] = condition
    features["RPM"] = rpm
    features["Label"] = get_label(
        bearing
    )

    features["Filename"] = os.path.basename(
        path
    )

    features["Samples"] = len(x)

    # --------------------------------------------------------
    # Time domain
    # --------------------------------------------------------

    time_features = extract_time_features(
        x
    )

    features.update(
        time_features
    )

    # --------------------------------------------------------
    # Direct spectrum
    # --------------------------------------------------------

    spectral_features = extract_spectral_features(
        x
    )

    features.update(
        spectral_features
    )

    # --------------------------------------------------------
    # Envelope
    # --------------------------------------------------------

    envelope_features, envelope = (
        extract_envelope_features(x)
    )

    features.update(
        envelope_features
    )

    # --------------------------------------------------------
    # Order-normalized envelope bands
    # --------------------------------------------------------

    if np.isfinite(rpm) and rpm > 0:

        order_features = extract_order_features(
            envelope,
            rpm
        )

        features.update(
            order_features
        )

    else:

        for name, _, _ in ORDER_BANDS:

            features[
                f"Order_{name}_Energy"
            ] = np.nan

            features[
                f"Order_{name}_Ratio"
            ] = np.nan

    # --------------------------------------------------------
    # Noise proxies
    # --------------------------------------------------------

    noise_features = extract_noise_features(
        x
    )

    features.update(
        noise_features
    )

    return features


# ============================================================
# ANALYZE TARGET BEARINGS
# ============================================================

def analyze_targets():

    all_rows = []

    print("=" * 80)
    print("STEP 155 - PHYSICAL SIGNATURE DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    print(
        f"\nSampling rate: {FS:.0f} Hz"
    )

    print(
        "\nRequired vibration channel: "
        "Y[6] / vibration_1"
    )

    print(
        "\nTarget bearings:"
    )

    for bearing in TARGET_BEARINGS:

        print(
            f"  {bearing}"
        )

    print(
        "\nCondition mapping:"
    )

    for condition, rpm in CONDITIONS.items():

        print(
            f"  {condition}: {rpm} RPM"
        )

    print("\n" + "=" * 80)

    for bearing in TARGET_BEARINGS:

        print(
            f"\nPROCESSING {bearing}"
        )

        print(
            "-" * 60
        )

        files = find_bearing_files(
            bearing
        )

        print(
            f"Found {len(files)} MAT files"
        )

        success = 0
        errors = 0

        for i, path in enumerate(
            files,
            start=1
        ):

            try:

                row = analyze_recording(
                    path,
                    bearing
                )

                all_rows.append(
                    row
                )

                success += 1

                if (
                    i % 10 == 0
                    or i == len(files)
                ):

                    print(
                        f"  Processed "
                        f"{i}/{len(files)}"
                    )

            except Exception as e:

                errors += 1

                print(
                    f"  ERROR: "
                    f"{os.path.basename(path)}"
                )

                print(
                    f"         {e}"
                )

        print(
            f"Completed {bearing}: "
            f"{success} successful, "
            f"{errors} errors"
        )

    df = pd.DataFrame(
        all_rows
    )

    return df


# ============================================================
# SUMMARY BY BEARING
# ============================================================

def summarize_by_bearing(df):

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if c not in [
            "Label",
            "RPM",
            "Samples"
        ]
    ]

    summary = (
        df.groupby("Bearing")[numeric_cols]
        .agg(
            [
                "mean",
                "std",
                "median"
            ]
        )
    )

    summary.columns = [
        "_".join(
            [str(x) for x in col]
        )
        for col in summary.columns
    ]

    summary = summary.reset_index()

    return summary


# ============================================================
# SUMMARY BY CONDITION
# ============================================================

def summarize_by_condition(df):

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if c not in [
            "Label",
            "RPM",
            "Samples"
        ]
    ]

    summary = (
        df.groupby(
            ["Bearing", "Condition"]
        )[numeric_cols]
        .mean()
        .reset_index()
    )

    return summary


# ============================================================
# MANN-WHITNEY EFFECT SIZE
# ============================================================

def rank_biserial_from_u(
    u,
    n1,
    n2
):

    if n1 <= 0 or n2 <= 0:

        return np.nan

    return (
        (2.0 * u)
        / (n1 * n2)
        - 1.0
    )


def compare_features(
    df,
    bearing_a,
    bearing_b
):

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    excluded = {
        "Label",
        "RPM",
        "Samples"
    }

    numeric_cols = [
        c for c in numeric_cols
        if c not in excluded
    ]

    rows = []

    a = df[
        df["Bearing"] == bearing_a
    ]

    b = df[
        df["Bearing"] == bearing_b
    ]

    for feature in numeric_cols:

        x = pd.to_numeric(
            a[feature],
            errors="coerce"
        ).dropna().values

        y = pd.to_numeric(
            b[feature],
            errors="coerce"
        ).dropna().values

        if len(x) == 0 or len(y) == 0:

            continue

        try:

            u, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            effect = rank_biserial_from_u(
                u,
                len(x),
                len(y)
            )

        except Exception:

            u = np.nan
            p = np.nan
            effect = np.nan

        rows.append(
            {
                "Bearing_A": bearing_a,
                "Bearing_B": bearing_b,
                "Feature": feature,
                "N_A": len(x),
                "N_B": len(y),
                "Mann_Whitney_U": u,
                "P_Value": p,
                "Rank_Biserial_Effect":
                    effect,
                "A_Median":
                    np.median(x),
                "B_Median":
                    np.median(y),
                "A_Mean":
                    np.mean(x),
                "B_Mean":
                    np.mean(y),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HEALTHY REFERENCE COMPARISON
# ============================================================

def healthy_reference_analysis():

    print(
        "\n" + "=" * 80
    )

    print(
        "HEALTHY REFERENCE ANALYSIS"
    )

    print(
        "=" * 80
    )

    rows = []

    # Load all healthy bearings.
    for bearing in HEALTHY_BEARINGS:

        print(
            f"\nLoading healthy reference: "
            f"{bearing}"
        )

        files = find_bearing_files(
            bearing
        )

        print(
            f"  {len(files)} recordings"
        )

        for path in files:

            try:

                row = analyze_recording(
                    path,
                    bearing
                )

                rows.append(
                    row
                )

            except Exception as e:

                print(
                    f"  ERROR: "
                    f"{os.path.basename(path)}"
                )

                print(
                    f"         {e}"
                )

    healthy_df = pd.DataFrame(
        rows
    )

    # Target data are loaded separately
    # so that the target-vs-healthy comparison
    # is explicit.

    target_rows = []

    for bearing in [
        "KA03",
        "KI05"
    ]:

        files = find_bearing_files(
            bearing
        )

        for path in files:

            try:

                target_rows.append(
                    analyze_recording(
                        path,
                        bearing
                    )
                )

            except Exception:

                pass

    target_df = pd.DataFrame(
        target_rows
    )

    numeric_cols = healthy_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_cols = [
        c for c in numeric_cols
        if c not in [
            "Label",
            "RPM",
            "Samples"
        ]
    ]

    output_rows = []

    for target_bearing in [
        "KA03",
        "KI05"
    ]:

        target = target_df[
            target_df["Bearing"]
            == target_bearing
        ]

        for feature in numeric_cols:

            x = pd.to_numeric(
                target[feature],
                errors="coerce"
            ).dropna().values

            y = pd.to_numeric(
                healthy_df[feature],
                errors="coerce"
            ).dropna().values

            if len(x) == 0 or len(y) == 0:

                continue

            try:

                u, p = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided"
                )

                effect = (
                    rank_biserial_from_u(
                        u,
                        len(x),
                        len(y)
                    )
                )

            except Exception:

                u = np.nan
                p = np.nan
                effect = np.nan

            output_rows.append(
                {
                    "Target_Bearing":
                        target_bearing,

                    "Feature":
                        feature,

                    "Target_N":
                        len(x),

                    "Healthy_N":
                        len(y),

                    "Target_Mean":
                        np.mean(x),

                    "Healthy_Mean":
                        np.mean(y),

                    "Target_Median":
                        np.median(x),

                    "Healthy_Median":
                        np.median(y),

                    "Mann_Whitney_U":
                        u,

                    "P_Value":
                        p,

                    "Rank_Biserial_Effect":
                        effect,
                }
            )

    return pd.DataFrame(
        output_rows
    )


# ============================================================
# PLOT FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(df):

    selected_features = [
        "RMS",
        "Kurtosis",
        "Crest_Factor",
        "Spectral_Total_Power",
        "Dominant_Frequency_Hz",
        "Envelope_RMS",
        "Envelope_Kurtosis",
        "Order_0.5_2_Ratio",
        "Order_2_5_Ratio",
        "Noise_Peak_to_Median",
    ]

    available = [
        f for f in selected_features
        if f in df.columns
    ]

    if not available:

        print(
            "No selected plotting features available."
        )

        return

    n = len(available)

    rows = int(
        np.ceil(n / 2)
    )

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 4 * rows)
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for idx, feature in enumerate(
        available
    ):

        ax = axes[idx]

        for bearing in TARGET_BEARINGS:

            values = pd.to_numeric(
                df[
                    df["Bearing"] == bearing
                ][feature],
                errors="coerce"
            ).dropna()

            ax.hist(
                values,
                bins=25,
                alpha=0.45,
                label=bearing
            )

        ax.set_title(
            feature
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Count"
        )

        ax.legend()

        ax.grid(
            alpha=0.2
        )

    for idx in range(
        len(available),
        len(axes)
    ):

        axes[idx].axis(
            "off"
        )

    fig.suptitle(
        "Physical Signature Feature Distributions",
        fontsize=16
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "step155_feature_distributions.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# CONDITION PROFILES
# ============================================================

def plot_condition_profiles(
    df
):

    selected_features = [
        "RMS",
        "Kurtosis",
        "Crest_Factor",
        "Spectral_Total_Power",
        "Envelope_RMS",
        "Envelope_Kurtosis",
        "Noise_Peak_to_Median",
    ]

    available = [
        f for f in selected_features
        if f in df.columns
    ]

    if not available:

        return

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for bearing in TARGET_BEARINGS:

        means = []

        conditions = []

        for condition in CONDITIONS.keys():

            subset = df[
                (df["Bearing"] == bearing)
                &
                (df["Condition"] == condition)
            ]

            if len(subset) == 0:

                continue

            value = subset[
                "RMS"
            ].mean()

            means.append(
                value
            )

            conditions.append(
                condition
            )

        if means:

            ax.plot(
                range(len(means)),
                means,
                marker="o",
                label=bearing
            )

    ax.set_xticks(
        range(len(CONDITIONS))
    )

    ax.set_xticklabels(
        list(CONDITIONS.keys()),
        rotation=20
    )

    ax.set_ylabel(
        "Mean RMS"
    )

    ax.set_title(
        "RMS Profile Across Operating Conditions"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "step155_condition_profiles.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# TEXT REPORT
# ============================================================

def create_report(
    df,
    summary,
    condition_summary,
    effect_df,
    healthy_comparison
):

    output_path = os.path.join(
        OUTPUT_DIR,
        "step155_results.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "STEP 155 - PHYSICAL SIGNATURE "
            "DIAGNOSTIC ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Purpose:\n"
        )

        f.write(
            "Targeted diagnostic comparison of K002, "
            "KA03, and KI05 using time-domain, "
            "direct spectral, envelope, order-band, "
            "and noise-floor features.\n\n"
        )

        f.write(
            "Signal channel: Y[6] / vibration_1\n"
        )

        f.write(
            "Nominal sampling rate: 64 kHz\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This experiment is diagnostic only and "
            "does not modify the frozen primary "
            "LOBO results.\n\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "RECORDING COUNTS\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        for bearing in TARGET_BEARINGS:

            count = (
                df["Bearing"]
                .eq(bearing)
                .sum()
            )

            f.write(
                f"{bearing}: {count} recordings\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "TARGET BEARING MEANS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "CONDITION SUMMARY\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            condition_summary.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "PAIRWISE MANN-WHITNEY ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "Pairs:\n"
        )

        f.write(
            "  KI05 vs K002\n"
        )

        f.write(
            "  KI05 vs KA03\n"
        )

        f.write(
            "  KA03 vs K002\n\n"
        )

        f.write(
            effect_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "HEALTHY REFERENCE COMPARISON\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            healthy_comparison.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "INTERPRETATION GUIDANCE\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "1. Differences in feature distributions "
            "identify measurable signal signatures.\n"
        )

        f.write(
            "2. Large rank-biserial effects indicate "
            "strong distributional separation.\n"
        )

        f.write(
            "3. Mann-Whitney p-values are diagnostic "
            "and should not be interpreted as causal "
            "proof.\n"
        )

        f.write(
            "4. No multiple-testing correction is "
            "applied in this diagnostic script.\n"
        )

        f.write(
            "5. The experiment does not establish "
            "specific bearing damage geometry unless "
            "independent geometry metadata support "
            "such a conclusion.\n"
        )

        f.write(
            "6. The analysis should therefore be used "
            "to characterize signal-space differences "
            "rather than claim a definitive physical "
            "failure mechanism.\n"
        )

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nStarting Step 155...\n"
    )

    # --------------------------------------------------------
    # Target bearings
    # --------------------------------------------------------

    df = analyze_targets()

    if df.empty:

        raise RuntimeError(
            "No target recordings were successfully processed."
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "TARGET DATASET COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Save per-recording data
    # --------------------------------------------------------

    per_recording_path = os.path.join(
        OUTPUT_DIR,
        "step155_physical_signature_per_recording.csv"
    )

    df.to_csv(
        per_recording_path,
        index=False
    )

    print(
        f"\nSaved: {per_recording_path}"
    )

    # --------------------------------------------------------
    # Bearing summary
    # --------------------------------------------------------

    summary = summarize_by_bearing(
        df
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "step155_physical_signature_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Saved: {summary_path}"
    )

    # --------------------------------------------------------
    # Condition summary
    # --------------------------------------------------------

    condition_summary = summarize_by_condition(
        df
    )

    condition_path = os.path.join(
        OUTPUT_DIR,
        "step155_condition_summary.csv"
    )

    condition_summary.to_csv(
        condition_path,
        index=False
    )

    print(
        f"Saved: {condition_path}"
    )

    # --------------------------------------------------------
    # Pairwise effect sizes
    # --------------------------------------------------------

    pairs = [
        ("KI05", "K002"),
        ("KI05", "KA03"),
        ("KA03", "K002"),
    ]

    effect_tables = []

    for a, b in pairs:

        print(
            f"\nComparing {a} vs {b}"
        )

        result = compare_features(
            df,
            a,
            b
        )

        effect_tables.append(
            result
        )

    effect_df = pd.concat(
        effect_tables,
        ignore_index=True
    )

    effect_path = os.path.join(
        OUTPUT_DIR,
        "step155_effect_sizes.csv"
    )

    effect_df.to_csv(
        effect_path,
        index=False
    )

    print(
        f"\nSaved: {effect_path}"
    )

    # --------------------------------------------------------
    # Healthy reference
    # --------------------------------------------------------

    healthy_comparison = (
        healthy_reference_analysis()
    )

    healthy_path = os.path.join(
        OUTPUT_DIR,
        "step155_healthy_reference_comparison.csv"
    )

    healthy_comparison.to_csv(
        healthy_path,
        index=False
    )

    print(
        f"\nSaved: {healthy_path}"
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_feature_distributions(
        df
    )

    plot_condition_profiles(
        df
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    create_report(
        df,
        summary,
        condition_summary,
        effect_df,
        healthy_comparison
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 155 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nGenerated files:"
    )

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f"  {filename}"
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These results are diagnostic and "
        "must not replace the frozen primary "
        "LOBO results."
    )


if __name__ == "__main__":

    main()