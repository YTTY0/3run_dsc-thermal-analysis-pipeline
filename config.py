from pathlib import Path

# Data and output directories
DATA_DIR = Path("./Exampledata")
OUTPUT_DIR = DATA_DIR / "output"

# Sample selection
# Leave TARGET_SAMPLES empty ([]) to process all non-standard, non-empty samples in DATA_DIR.
TARGET_SAMPLES = []
STANDARD_SAMPLE = "Sapphire"
EMPTY_SAMPLE = None  # Set to a sample name or leave None to auto-detect blank pan runs

# Baseline symmetry window (°C) and derivative cutoff (mW/K) for cycle selection
TMIN = -40.0
TMAX = 40.0
DERIV_THRESHOLD = 0.005

# Temperature intervals (K) for linear Cp(T) baseline fits before and after transition
Cp_ranges_HEATING = {
    "low_min": 245.0,
    "low_max": 265.0,
    "high_min": 290.0,
    "high_max": 305.0,
}

Cp_ranges_COOLING = {
    "high_min": 290.0,
    "high_max": 305.0,
    "low_min": 245.0,
    "low_max": 265.0,
}