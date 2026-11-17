from io import StringIO
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

COLUMN_ALIASES = {
    "Temperature (°C)": "Temperature",
    "Temperature (K)": "Temperature",
    "Time (min)": "Time",
    "Heat Flow (mW)": "HeatFlow",
    "Heat Flow (Normalized)": "HeatFlow(Normalized)",
    "Heat Flow": "HeatFlow",
    "HeatFlow(Normalized)": "HeatFlow(Normalized)",
    "Heat Capacity (Normalized)": "HeatCapacity(Normalized)",
    "Heat Capacity": "HeatCapacity",
}

RAMP_PATTERN = re.compile(
    r"Ramp\s+([+-]?\d+(?:\.\d+)?)\s*"
    r"(?:°?\s*[CK])?\s*/\s*(min(?:ute)?|s(?:ec(?:ond)?)?)\s+to\s+"
    r"([+-]?\d+(?:\.\d+)?)\s*(°?\s*[CK])",
    re.IGNORECASE,
)

EMPTY_PAN_KEYWORDS = ("empty", "emptypan", "blank", "baseline")
SAPPHIRE_KEYWORDS = ("sapphire", "al2o3", "srm720", "standard")

# Reference specific heat capacity of synthetic sapphire (alpha-Al2O3, 90 - 1650 K)
# D. A. Ditmars et al., J. Res. Nat. Bur. Stand., 87(2), 159-163 (1982).
SAPPHIRE_REFERENCE = (
    "D. A. Ditmars et al., J. Res. Nat. Bur. Stand., 87(2), 159-163 (1982) "
    "[NBS synthetic sapphire, 99.98-99.99% purity, MW = 101.9613 g/mol]"
)

_SAPPHIRE_T_K = np.array([
    90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
    190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 273.15,
    280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0,
    380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0,
    480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0,
    580.0, 590.0, 600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0,
    680.0, 690.0, 700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0,
    860.0, 880.0, 900.0, 920.0, 940.0, 960.0, 980.0, 1000.0, 1020.0, 1040.0,
    1060.0, 1080.0, 1100.0, 1120.0, 1140.0, 1160.0, 1180.0, 1200.0, 1250.0,
    1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0,
])

_SAPPHIRE_CP_J_G_K = np.array([
    0.0949, 0.1261, 0.1603, 0.1968, 0.2349, 0.2739, 0.3134, 0.3526, 0.3913, 0.4291,
    0.4659, 0.5014, 0.5356, 0.5684, 0.5996, 0.6294, 0.6579, 0.6848, 0.7103, 0.7180,
    0.7343, 0.7572, 0.7788, 0.7994, 0.8188, 0.8373, 0.8548, 0.8713, 0.8871, 0.9020,
    0.9161, 0.9296, 0.9423, 0.9545, 0.9660, 0.9770, 0.9875, 0.9975, 1.0070, 1.0161,
    1.0247, 1.0330, 1.0409, 1.0484, 1.0557, 1.0627, 1.0692, 1.0756, 1.0817, 1.0876,
    1.0932, 1.0987, 1.1038, 1.1089, 1.1137, 1.1183, 1.1228, 1.1271, 1.1313, 1.1353,
    1.1393, 1.1431, 1.1467, 1.1538, 1.1604, 1.1667, 1.1726, 1.1783, 1.1837, 1.1888,
    1.1937, 1.1985, 1.2030, 1.2074, 1.2117, 1.2159, 1.2198, 1.2237, 1.2275, 1.2312,
    1.2348, 1.2383, 1.2417, 1.2451, 1.2484, 1.2516, 1.2548, 1.2578, 1.2653, 1.2724,
    1.2792, 1.2856, 1.2917, 1.2975, 1.3028, 1.3079, 1.3128,
])

_SAPPHIRE_SPLINE = CubicSpline(_SAPPHIRE_T_K, _SAPPHIRE_CP_J_G_K, extrapolate=False)


def sapphire_cp_literature(temp_k: np.ndarray) -> np.ndarray:
    """Interpolate reference Cp (J g^-1 K^-1) of synthetic sapphire at temp_k (K)."""
    arr = np.asarray(temp_k, dtype=float)
    if np.any((arr < _SAPPHIRE_T_K[0]) | (arr > _SAPPHIRE_T_K[-1])):
        raise ValueError(
            f"Temperature out of sapphire reference range ({_SAPPHIRE_T_K[0]}-{_SAPPHIRE_T_K[-1]} K)."
        )
    return _SAPPHIRE_SPLINE(arr)


def sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters in sample names with underscores."""
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "unnamed_sample"


def natural_file_key(path: Path | str) -> list:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", Path(path).name)]


def standardize_columns(cols) -> list[str]:
    cleaned = []
    for col in cols:
        name = " ".join(str(col).replace("Â", "").strip().split())
        cleaned.append(COLUMN_ALIASES.get(name, name))
    return cleaned


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in df.columns:
            return cand
    for cand in standardize_columns(candidates):
        if cand in df.columns:
            return cand
    for col in df.columns:
        col_norm = col.lower().replace(" ", "")
        for cand in candidates:
            cand_norm = cand.lower().replace(" ", "")
            if cand_norm in col_norm or col_norm in cand_norm:
                return col
    return None


def _detect_temp_unit(raw_header: str, raw_unit_cell: str, ramp_unit: str | None) -> str:
    """Infer whether a TRIOS segment records temperature in Kelvin ('K') or Celsius ('C')."""
    unit_clean = raw_unit_cell.replace("Â", "").strip()
    if unit_clean.upper() in ("K", "KELVIN"):
        return "K"
    if unit_clean.upper() in ("°C", "C", "DEGC", "CELSIUS"):
        return "C"

    header_clean = raw_header.replace("Â", "").strip()
    if re.search(r"(?:\(|\[|\b)K(?:\)|\]|$)", header_clean):
        return "K"
    if "°C" in header_clean or re.search(r"(?:\(|\[|\b)C(?:\)|\]|$)", header_clean):
        return "C"

    if ramp_unit:
        ramp_clean = ramp_unit.replace("Â", "").strip().upper()
        if "K" in ramp_clean:
            return "K"
        if "C" in ramp_clean:
            return "C"

    return "C"


def _matches_keywords(labels: tuple, keywords: tuple[str, ...]) -> bool:
    for val in labels:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        norm = re.sub(r"[^a-z0-9]+", "", str(val).lower())
        if any(key in norm for key in keywords):
            return True
    return False


def is_empty_pan(*labels) -> bool:
    return _matches_keywords(labels, EMPTY_PAN_KEYWORDS)


def is_sapphire(*labels) -> bool:
    return _matches_keywords(labels, SAPPHIRE_KEYWORDS)


def matches_sample_query(sample_name: str, query: str) -> bool:
    s_norm = re.sub(r"[^a-z0-9]+", "", sample_name.lower())
    q_norm = re.sub(r"[^a-z0-9]+", "", query.lower())
    return bool(q_norm) and (q_norm == s_norm or q_norm in s_norm)


def read_header_info(filepath: Path | str, max_lines: int = 80) -> tuple[str | None, float | None]:
    """Read Sample Name and Sample Mass (mg) from the metadata block of a TRIOS export."""
    sample_name = None
    sample_mass_mg = None

    with Path(filepath).open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 2:
                continue
            key = parts[0]
            if key.startswith("Sample Name"):
                sample_name = parts[1]
            elif key.startswith("Sample Mass") and "Reference" not in key:
                match = re.search(r"[+-]?\d+(?:\.\d+)?", parts[1])
                if match:
                    val = float(match.group(0))
                    unit = parts[2].lower() if len(parts) > 2 else ""
                    if not unit:
                        unit_match = re.search(r"(mg|g)\b", parts[1].lower())
                        unit = unit_match.group(1) if unit_match else "mg"
                    sample_mass_mg = val * 1000.0 if unit == "g" else val

    return sample_name, sample_mass_mg


def _to_kelvin_per_sec(rate: float, time_unit: str) -> float:
    return rate / 60.0 if time_unit.lower().startswith("min") else rate


def parse_ramp_segments(
    filepath: Path | str,
    median_window: int = 10,
    eps: float = 1e-6,
    kelvin_offset: float = 273.15,
) -> list[dict]:
    """Extract all Ramp segments, heating/cooling rates (K/s), and standardized temperatures."""
    filepath = Path(filepath)
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    segments = []
    i = 0
    ramp_idx = 0

    while i < len(lines):
        line = lines[i].strip().replace("Â", "")
        match = RAMP_PATTERN.search(line)
        if not match:
            i += 1
            continue

        ramp_idx += 1
        rate_k_per_s = _to_kelvin_per_sec(float(match.group(1)), match.group(2))
        target_temp = float(match.group(3))
        ramp_unit = match.group(4)
        i += 1

        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i + 2 >= len(lines):
            break

        raw_headers = [x.replace("Â", "").strip() for x in lines[i].split("\t")]
        raw_units = [x.replace("Â", "").strip() for x in lines[i + 1].split("\t")]
        col_names = standardize_columns(raw_headers)
        i += 2

        data_lines = []
        while i < len(lines):
            raw = lines[i].replace("Â", "")
            stripped = raw.strip()
            if stripped == "" or stripped.lower().startswith("[step]") or RAMP_PATTERN.search(stripped):
                break
            data_lines.append(raw)
            i += 1

        if len(data_lines) < 2:
            continue

        df = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t", engine="python", names=col_names)
        temp_col = find_column(df, ["Temperature", "Temperature (°C)", "Temperature (K)"])
        if temp_col is None or df.empty:
            continue

        col_idx = list(df.columns).index(temp_col)
        raw_hdr = raw_headers[col_idx] if col_idx < len(raw_headers) else temp_col
        raw_unit = raw_units[col_idx] if col_idx < len(raw_units) else ""
        temp_unit = _detect_temp_unit(raw_hdr, raw_unit, ramp_unit)

        numeric_temps = pd.to_numeric(df[temp_col], errors="coerce")
        valid_temps = numeric_temps.dropna()
        if valid_temps.empty:
            continue

        df["Temperature(K)"] = numeric_temps if temp_unit == "K" else (numeric_temps + kelvin_offset)
        df["Temperature(C)"] = (numeric_temps - kelvin_offset) if temp_unit == "K" else numeric_temps

        n = min(median_window, len(valid_temps))
        t_start = float(valid_temps.iloc[:n].median())
        t_end = float(valid_temps.iloc[-n:].median())

        if t_end > t_start + eps:
            phase = "heating"
        elif t_end < t_start - eps:
            phase = "cooling"
        else:
            phase = "unknown"

        segments.append({
            "phase": phase,
            "df": df,
            "rate_K_per_s": rate_k_per_s,
            "target_T": target_temp,
            "temp_unit": temp_unit,
            "source_file": filepath.name,
            "ramp_index": ramp_idx,
            "start_T": t_start,
            "end_T": t_end,
        })

    return segments


def pair_segments_into_cycles(
    segments: list[dict],
    allow_reverse: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Pair sequential heating and cooling segments into thermal cycles."""
    cycles = []
    pending = {"heating": None, "cooling": None}

    for seg in segments:
        phase = seg.get("phase")
        if phase not in ("heating", "cooling"):
            continue

        opposite = "cooling" if phase == "heating" else "heating"
        can_pair = pending[opposite] is not None and (allow_reverse or opposite == "heating")

        if can_pair:
            heat_seg = pending["heating"] if pending["heating"] is not None else seg
            cool_seg = pending["cooling"] if pending["cooling"] is not None else seg

            cycles.append({
                "cycle_number": len(cycles) + 1,
                "heating": heat_seg["df"],
                "cooling": cool_seg["df"],
                "heating_rate_K_per_s": heat_seg["rate_K_per_s"],
                "cooling_rate_K_per_s": cool_seg["rate_K_per_s"],
                "source_file": f"{heat_seg['source_file']} + {cool_seg['source_file']}",
                "heating_source_file": heat_seg["source_file"],
                "cooling_source_file": cool_seg["source_file"],
            })
            pending = {"heating": None, "cooling": None}
        else:
            pending[phase] = seg

    if verbose:
        unpaired = [p for p, seg in pending.items() if seg is not None]
        if unpaired:
            print(f"Warning: unpaired segment(s) remaining: {unpaired}")

    return cycles


def build_cycles_from_files(
    filepaths: list[Path],
    median_window: int = 10,
    eps: float = 1e-6,
    kelvin_offset: float = 273.15,
    allow_reverse: bool = True,
    verbose: bool = True,
) -> list[dict]:
    segments = []
    for fp in sorted(filepaths, key=natural_file_key):
        segments.extend(
            parse_ramp_segments(fp, median_window=median_window, eps=eps, kelvin_offset=kelvin_offset)
        )
    return pair_segments_into_cycles(segments, allow_reverse=allow_reverse, verbose=verbose)


def _extract_temp_hf_kelvin(
    df: pd.DataFrame,
    temp_cands: list[str],
    hf_cands: list[str],
    kelvin_offset: float = 273.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, deduplicated Temperature (K) and Heat Flow (mW) arrays."""
    hcol = find_column(df, hf_cands)
    if hcol is None:
        return np.array([]), np.array([])

    if "Temperature(K)" in df.columns:
        t_k = pd.to_numeric(df["Temperature(K)"], errors="coerce").to_numpy(dtype=float)
    else:
        tcol = find_column(df, temp_cands)
        if tcol is None:
            return np.array([]), np.array([])
        is_kelvin = bool(re.search(r"(?:\(|\[|\b)K(?:\)|\]|$)", tcol))
        offset = 0.0 if is_kelvin else kelvin_offset
        t_k = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float) + offset

    hf = pd.to_numeric(df[hcol], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t_k) & np.isfinite(hf)
    t_k, hf = t_k[valid], hf[valid]

    if t_k.size < 2:
        return np.array([]), np.array([])

    order = np.argsort(t_k)
    t_k, hf = t_k[order], hf[order]

    t_unique, inv = np.unique(t_k, return_inverse=True)
    if t_unique.size == t_k.size:
        return t_k, hf

    hf_mean = np.bincount(inv, weights=hf) / np.bincount(inv)
    return t_unique, hf_mean


def select_best_cycle(
    cycles: list[dict],
    temp_cands: list[str],
    hf_cands: list[str],
    tmin_c: float = -40.0,
    tmax_c: float = 100.0,
    deriv_threshold: float = 0.005,
    kelvin_offset: float = 273.15,
) -> tuple[dict, pd.DataFrame]:
    """Select the cycle pair minimizing rate-normalized |heat + cool| asymmetry outside transitions."""
    if not cycles:
        raise ValueError("Cannot select best cycle from an empty cycle list.")

    score_rows = []
    best_cycle = cycles[0]
    best_score = np.inf

    for cyc in cycles:
        t_h_k, hf_h = _extract_temp_hf_kelvin(cyc["heating"], temp_cands, hf_cands, kelvin_offset)
        t_c_k, hf_c = _extract_temp_hf_kelvin(cyc["cooling"], temp_cands, hf_cands, kelvin_offset)

        if t_h_k.size < 2 or t_c_k.size < 2:
            score_rows.append({
                "Cycle": cyc["cycle_number"],
                "Source": cyc["source_file"],
                "Baseline_Points": 0,
                "Mean_Abs_Sum_mW": np.nan,
                "Selected": False,
            })
            continue

        t_low = max(t_h_k.min(), t_c_k.min())
        t_high = min(t_h_k.max(), t_c_k.max())
        t_union_k = np.union1d(t_h_k, t_c_k)
        t_union_k = t_union_k[(t_union_k >= t_low) & (t_union_k <= t_high)]

        if t_union_k.size < 5:
            score_rows.append({
                "Cycle": cyc["cycle_number"],
                "Source": cyc["source_file"],
                "Baseline_Points": 0,
                "Mean_Abs_Sum_mW": np.nan,
                "Selected": False,
            })
            continue

        hf_h_grid = np.interp(t_union_k, t_h_k, hf_h, left=np.nan, right=np.nan)
        hf_c_grid = np.interp(t_union_k, t_c_k, hf_c, left=np.nan, right=np.nan)

        rate_h = cyc.get("heating_rate_K_per_s") or 1.0
        rate_c = cyc.get("cooling_rate_K_per_s") or 1.0
        sum_mw = hf_h_grid + hf_c_grid * (rate_h / rate_c)

        t_union_c = t_union_k - kelvin_offset
        d_sum = np.abs(np.gradient(sum_mw, t_union_c))

        in_window = (t_union_c >= tmin_c) & (t_union_c <= tmax_c)
        baseline_mask = in_window & (d_sum <= deriv_threshold)
        if baseline_mask.sum() < 5:
            baseline_mask = in_window if in_window.sum() >= 5 else np.ones_like(t_union_c, dtype=bool)

        mean_abs_sum = float(np.mean(np.abs(sum_mw[baseline_mask])))

        score_rows.append({
            "Cycle": cyc["cycle_number"],
            "Source": cyc["source_file"],
            "Baseline_Points": int(baseline_mask.sum()),
            "Mean_Abs_Sum_mW": mean_abs_sum,
            "Selected": False,
        })

        if mean_abs_sum < best_score:
            best_score = mean_abs_sum
            best_cycle = cyc

    for row in score_rows:
        if row["Cycle"] == best_cycle["cycle_number"]:
            row["Selected"] = True

    return best_cycle, pd.DataFrame(score_rows)


def compute_calibration_factors(
    sapphire_cycle: dict,
    sapphire_mass_mg: float,
    blank_cycle: dict | None,
    temp_cands: list[str],
    hf_cands: list[str],
    flip_cooling_sign: bool = True,
    kelvin_offset: float = 273.15,
) -> dict[str, dict]:
    """Compute dimensionless calibration factor E(T) = Cp_ref(T) / Cp_apparent(T) from sapphire."""
    calib = {}

    for mode in ("heating", "cooling"):
        df_sap = sapphire_cycle.get(mode)
        rate_sap = sapphire_cycle.get(f"{mode}_rate_K_per_s")
        if df_sap is None or not rate_sap:
            continue

        t_sap, hf_sap = _extract_temp_hf_kelvin(df_sap, temp_cands, hf_cands, kelvin_offset)
        if t_sap.size < 2:
            continue

        if blank_cycle is not None and blank_cycle.get(mode) is not None:
            t_blk, hf_blk = _extract_temp_hf_kelvin(blank_cycle[mode], temp_cands, hf_cands, kelvin_offset)
            hf_blk_on_sap = np.interp(t_sap, t_blk, hf_blk, left=np.nan, right=np.nan)
            blank_curve = (t_blk, hf_blk)
        else:
            hf_blk_on_sap = np.zeros_like(hf_sap)
            blank_curve = None

        net_hf_sap = hf_sap - hf_blk_on_sap
        cp_sap_apparent = net_hf_sap / (sapphire_mass_mg * rate_sap)
        cp_sap_ref = sapphire_cp_literature(t_sap)

        with np.errstate(divide="ignore", invalid="ignore"):
            e_factor = cp_sap_ref / cp_sap_apparent

        if mode == "cooling" and flip_cooling_sign:
            e_factor = -e_factor

        valid = np.isfinite(e_factor)
        calib[mode] = {
            "T_K": t_sap[valid],
            "E_factor": e_factor[valid],
            "blank_curve": blank_curve,
            "flip_cooling_sign": flip_cooling_sign,
            "cp_sap_ref": cp_sap_ref[valid],
            "cp_sap_apparent": cp_sap_apparent[valid],
        }

    return calib


def calibrate_sample_cycle(
    sample_cycle: dict,
    sample_mass_mg: float,
    calib: dict[str, dict],
    temp_cands: list[str],
    hf_cands: list[str],
    kelvin_offset: float = 273.15,
) -> dict[str, pd.DataFrame]:
    """Apply optional blank subtraction and sapphire calibration factor E(T) to a sample cycle."""
    calibrated = {}

    for mode in ("heating", "cooling"):
        if mode not in calib:
            continue

        df_smp = sample_cycle.get(mode)
        rate_smp = sample_cycle.get(f"{mode}_rate_K_per_s")
        if df_smp is None or not rate_smp:
            continue

        t_smp, hf_smp = _extract_temp_hf_kelvin(df_smp, temp_cands, hf_cands, kelvin_offset)
        if t_smp.size < 2:
            continue

        blank_curve = calib[mode]["blank_curve"]
        hf_blk = (
            np.interp(t_smp, blank_curve[0], blank_curve[1], left=np.nan, right=np.nan)
            if blank_curve is not None
            else np.zeros_like(hf_smp)
        )

        net_hf = hf_smp - hf_blk
        raw_dqdt = net_hf / (sample_mass_mg * rate_smp)
        e_vals = np.interp(t_smp, calib[mode]["T_K"], calib[mode]["E_factor"], left=np.nan, right=np.nan)
        cal_cp = raw_dqdt * e_vals

        if mode == "cooling" and calib[mode].get("flip_cooling_sign", False):
            cal_cp = -cal_cp

        calibrated[mode] = pd.DataFrame({
            "Temperature(K)": t_smp,
            "Temperature(C)": t_smp - kelvin_offset,
            "NetHeatFlow(mW)": net_hf,
            "Raw_dQdT": raw_dqdt,
            "CalibrationFactor_E": e_vals,
            "CalibratedCp": cal_cp,
        })

    return calibrated