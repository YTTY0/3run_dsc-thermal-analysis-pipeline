import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from TAdsc_io import (
    SAPPHIRE_REFERENCE,
    build_cycles_from_files,
    calibrate_sample_cycle,
    compute_calibration_factors,
    is_empty_pan,
    is_sapphire,
    matches_sample_query,
    natural_file_key,
    parse_ramp_segments,
    read_header_info,
    sanitize_filename,
    select_best_cycle,
)

N_MEDIAN = 10
KELVIN_OFFSET = 273.15
FLIP_COOLING_CALIB_SIGN = True
TEMP_CANDS = ["Temperature °C", "Temperature (K)", "Temperature"]
HF_CANDS = ["HeatFlow mW", "HeatFlow"]

mpl.rcParams.update({
    "legend.fontsize": "large",
    "figure.figsize": (6, 4),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "axes.linewidth": 1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 12,
})

LINEAR_BASELINE_RANGES = {
    "heating": {
        "before": (cfg.Cp_ranges_HEATING["low_min"], cfg.Cp_ranges_HEATING["low_max"]),
        "after": (cfg.Cp_ranges_HEATING["high_min"], cfg.Cp_ranges_HEATING["high_max"]),
    },
    "cooling": {
        "before": (cfg.Cp_ranges_COOLING["high_min"], cfg.Cp_ranges_COOLING["high_max"]),
        "after": (cfg.Cp_ranges_COOLING["low_min"], cfg.Cp_ranges_COOLING["low_max"]),
    },
}


def format_box_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as an aligned ASCII grid table for plain-text reports."""
    if df.empty:
        return "(No data)"

    formatted_cols: dict[str, list[str]] = {}
    is_numeric: dict[str, bool] = {}

    for col in df.columns:
        series = df[col]
        is_numeric[col] = pd.api.types.is_numeric_dtype(series)
        vals = []
        for v in series:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, (int, np.integer)):
                vals.append(f"{int(v)}")
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.6g}")
            else:
                vals.append(str(v))
        formatted_cols[col] = vals

    col_widths = {
        col: max(len(str(col)), max((len(v) for v in formatted_cols[col]), default=0))
        for col in df.columns
    }

    border = "+" + "+".join("-" * (col_widths[c] + 2) for c in df.columns) + "+"
    header_sep = "+" + "+".join("=" * (col_widths[c] + 2) for c in df.columns) + "+"

    def _row_line(values: list[str]) -> str:
        cells = []
        for col, val in zip(df.columns, values):
            w = col_widths[col]
            padded = val.rjust(w) if is_numeric[col] else val.ljust(w)
            cells.append(f" {padded} ")
        return "|" + "|".join(cells) + "|"

    lines = [border, _row_line([str(c) for c in df.columns]), header_sep]
    for row_idx in range(len(df)):
        row_vals = [formatted_cols[c][row_idx] for c in df.columns]
        lines.append(_row_line(row_vals))
        lines.append(border)

    return "\n".join(lines)


def summarize_dataset(data_dir: Path) -> pd.DataFrame:
    """Scan TRIOS .txt files in data_dir and collect metadata, units, and ramp phases."""
    rows = []
    for fp in sorted(data_dir.glob("*.txt"), key=natural_file_key):
        sample_name, mass_mg = read_header_info(fp)
        segments = parse_ramp_segments(fp, median_window=N_MEDIAN, kelvin_offset=KELVIN_OFFSET)
        detected_units = ",".join(sorted({s["temp_unit"] for s in segments})) if segments else ""
        rows.append({
            "Filename": fp.name,
            "Samplename": sample_name,
            "Samplemass_mg": mass_mg,
            "Temp_unit": detected_units,
            "Rampsegments": len(segments),
            "Phases_in_file": ",".join(s["phase"] for s in segments),
        })
    return pd.DataFrame(rows)


def plot_calibration_factors(calib: dict[str, dict], out_dir: Path, show: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    mode_colors = {"heating": "red", "cooling": "blue"}
    for mode in ("heating", "cooling"):
        if mode not in calib:
            continue
        ax.plot(
            calib[mode]["T_K"],
            calib[mode]["E_factor"],
            color=mode_colors[mode],
            lw=1.8,
            label=f"{mode.capitalize()} calibration factor",
        )

    ax.axhline(1.0, color="gray", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_title("Sapphire Calibration Factor $E(T)$")
    ax.set_xlabel(r"$\mathit{T}$ ($\mathrm{K}$)")
    ax.set_ylabel(r"Calibration Factor $\mathit{E}$")
    ax.set_ylim(0.8, 1.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    save_path = out_dir / "Sapphire_Calibration_Factor_E.svg"
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved calibration plot: {save_path}")


def run_pipeline(data_dir: Path, output_dir: Path, show_plots: bool = False) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    summary_df = summarize_dataset(data_dir)
    if summary_df.empty:
        raise RuntimeError(f"No .txt files found in {data_dir}")
    print(summary_df.to_string(index=False))

    cycles_by_sample = {}
    mass_by_sample = {}
    for sample_name, group in summary_df.groupby("Samplename", dropna=True):
        if not isinstance(sample_name, str) or not sample_name.strip():
            continue
        files = [data_dir / fn for fn in group["Filename"]]
        cycles_by_sample[sample_name] = build_cycles_from_files(
            files, median_window=N_MEDIAN, kelvin_offset=KELVIN_OFFSET, verbose=True
        )
        masses = group["Samplemass_mg"].dropna()
        mass_by_sample[sample_name] = float(masses.iloc[0]) if not masses.empty else np.nan

    # Locate sapphire standard and optional empty-pan baseline
    sapphire_name = next(
        (n for n in cycles_by_sample if matches_sample_query(n, cfg.STANDARD_SAMPLE) or is_sapphire(n)),
        None,
    )
    if sapphire_name is None or not cycles_by_sample[sapphire_name]:
        raise RuntimeError(f"No valid Sapphire calibration run matching '{cfg.STANDARD_SAMPLE}' found in {data_dir}.")

    if cfg.EMPTY_SAMPLE:
        blank_name = next((n for n in cycles_by_sample if matches_sample_query(n, cfg.EMPTY_SAMPLE)), None)
    else:
        blank_name = next((n for n in cycles_by_sample if is_empty_pan(n)), None)

    sapphire_cycle, _ = select_best_cycle(
        cycles_by_sample[sapphire_name],
        TEMP_CANDS,
        HF_CANDS,
        tmin_c=cfg.TMIN,
        tmax_c=cfg.TMAX,
        deriv_threshold=cfg.DERIV_THRESHOLD,
        kelvin_offset=KELVIN_OFFSET,
    )
    sapphire_mass = mass_by_sample[sapphire_name]

    if blank_name and cycles_by_sample[blank_name]:
        blank_cycle, _ = select_best_cycle(
            cycles_by_sample[blank_name],
            TEMP_CANDS,
            HF_CANDS,
            tmin_c=cfg.TMIN,
            tmax_c=cfg.TMAX,
            deriv_threshold=cfg.DERIV_THRESHOLD,
            kelvin_offset=KELVIN_OFFSET,
        )
        print(f"\nStandard: {sapphire_name} ({sapphire_mass:.3f} mg) | Empty pan: {blank_name}")
    else:
        blank_cycle = None
        print(f"\nStandard: {sapphire_name} ({sapphire_mass:.3f} mg) | Empty pan: none (no blank subtraction)")

    calib = compute_calibration_factors(
        sapphire_cycle=sapphire_cycle,
        sapphire_mass_mg=sapphire_mass,
        blank_cycle=blank_cycle,
        temp_cands=TEMP_CANDS,
        hf_cands=HF_CANDS,
        flip_cooling_sign=FLIP_COOLING_CALIB_SIGN,
        kelvin_offset=KELVIN_OFFSET,
    )
    plot_dir = output_dir / "plots"
    plot_calibration_factors(calib, plot_dir, show=show_plots)

    # Calibrate best thermal cycle for each target sample
    cal_dir = output_dir / "calibrated_cp"
    cal_dir.mkdir(parents=True, exist_ok=True)
    processed_items = []

    for sample_name, cycles in cycles_by_sample.items():
        if sample_name in (sapphire_name, blank_name):
            continue
        if is_sapphire(sample_name) or is_empty_pan(sample_name):
            continue
        if cfg.TARGET_SAMPLES and not any(matches_sample_query(sample_name, q) for q in cfg.TARGET_SAMPLES):
            continue

        mass_mg = mass_by_sample.get(sample_name, np.nan)
        if not cycles or not np.isfinite(mass_mg) or mass_mg <= 0:
            print(f"Skipping {sample_name}: missing cycles or invalid sample mass.")
            continue

        best_cycle, score_df = select_best_cycle(
            cycles,
            TEMP_CANDS,
            HF_CANDS,
            tmin_c=cfg.TMIN,
            tmax_c=cfg.TMAX,
            deriv_threshold=cfg.DERIV_THRESHOLD,
            kelvin_offset=KELVIN_OFFSET,
        )
        print(f"\nCycle selection for {sample_name}:")
        print(score_df.to_string(index=False))

        cal_modes = calibrate_sample_cycle(
            sample_cycle=best_cycle,
            sample_mass_mg=mass_mg,
            calib=calib,
            temp_cands=TEMP_CANDS,
            hf_cands=HF_CANDS,
            kelvin_offset=KELVIN_OFFSET,
        )

        safe_name = sanitize_filename(sample_name)
        for mode, df_cal in cal_modes.items():
            n_oob = int(df_cal["CalibratedCp"].isna().sum())
            if n_oob:
                print(
                    f"Warning: {n_oob} point(s) in {sample_name} ({mode}) lie outside the sapphire "
                    "calibration temperature range and were set to NaN."
                )

            safe_stem = f"{safe_name}_{mode}"
            csv_path = cal_dir / f"{safe_stem}.csv"
            txt_path = cal_dir / f"{safe_stem}.txt"

            df_cal.to_csv(csv_path, index=False)
            df_cal.to_csv(txt_path, sep="\t", index=False, float_format="%.6g", lineterminator="\n")
            processed_items.append((sample_name, mass_mg, best_cycle["cycle_number"], mode, df_cal))

    # Fit linear baselines before and after phase transition
    linear_baseline_functions = {}
    linear_baseline_rows = []

    for name, mass, cyc_num, mode, dfp in processed_items:
        T = dfp["Temperature(K)"].to_numpy(dtype=float)
        Cp = dfp["CalibratedCp"].to_numpy(dtype=float)

        for side, (Tmin, Tmax) in LINEAR_BASELINE_RANGES[mode].items():
            mask = np.isfinite(T) & np.isfinite(Cp) & (T >= Tmin) & (T <= Tmax)
            if mask.sum() < 2:
                print(f"Not enough points in {Tmin}-{Tmax} K for: {name}, {mode}, {side}")
                continue

            slope, intercept = np.polyfit(T[mask], Cp[mask], 1)
            baseline_function = np.poly1d([slope, intercept])

            Cp_fit = baseline_function(T[mask])
            ss_res = np.sum((Cp[mask] - Cp_fit) ** 2)
            ss_tot = np.sum((Cp[mask] - np.mean(Cp[mask])) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            linear_baseline_functions[(name, mode, side)] = baseline_function
            linear_baseline_rows.append({
                "Sample": name,
                "Selected_Cycle": cyc_num,
                "Sample_Mass_mg": mass,
                "Mode": mode.capitalize(),
                "Transition_Side": side.capitalize(),
                "Fit_Range_K": f"{Tmin:g}-{Tmax:g}",
                "Linear_Baseline": f"Cp(T) = {slope:.6g} * T {intercept:+.6g}",
                "Slope_J_g_K2": slope,
                "Intercept_J_g_K": intercept,
                "R2": r_squared,
            })

    linear_baseline_summary = pd.DataFrame(linear_baseline_rows)
    boxed_summary = format_box_table(linear_baseline_summary)

    print("\nFitted Linear Baseline Functions:")
    print(boxed_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    eq_txt_path = output_dir / "linear_cp_equations.txt"
    summary_csv_path = output_dir / "linear_cp_summary.csv"

    with eq_txt_path.open("w", encoding="utf-8") as f:
        f.write("# Calibrated Specific Heat Capacity Linear Baseline Functions Cp(T) [J g^-1 K^-1]\n")
        f.write(f"# Sapphire calibration reference: {SAPPHIRE_REFERENCE}\n")
        f.write("# Heating and cooling Cp curves are reported on a unified positive Cp scale.\n\n")
        for row in linear_baseline_rows:
            f.write(
                f"Sample: {row['Sample']} (Cycle {row['Selected_Cycle']}, {row['Sample_Mass_mg']:.3f} mg) | "
                f"{row['Mode']} ({row['Transition_Side']} transition, {row['Fit_Range_K']} K)\n"
                f"  {row['Linear_Baseline']}   [R^2 = {row['R2']:.5f}]\n\n"
            )
        f.write("# Summary Table:\n")
        f.write(boxed_summary + "\n")

    linear_baseline_summary.to_csv(summary_csv_path, index=False, float_format="%.6g")
    print(f"\nExported boxed equations report to: {eq_txt_path}")
    print(f"Exported CSV summary table to:     {summary_csv_path}")

    # Plot calibrated Cp(T) and fitted baselines for the selected cycle of each sample
    processed_by_sample = defaultdict(dict)
    for name, mass, cyc_num, mode, dfp in processed_items:
        processed_by_sample[(name, mass, cyc_num)][mode] = dfp

    all_range_bounds = [
        cfg.Cp_ranges_HEATING["low_min"], cfg.Cp_ranges_HEATING["high_max"],
        cfg.Cp_ranges_COOLING["low_min"], cfg.Cp_ranges_COOLING["high_max"],
    ]
    xlim_min = min(all_range_bounds) - 5.0
    xlim_max = max(all_range_bounds) + 5.0

    for (name, mass, cyc_num), mode_dfs in processed_by_sample.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        mode_colors = {"heating": "red", "cooling": "blue"}
        side_styles = {"before": "--", "after": ":"}
        visible_cp = []

        for mode in ["heating", "cooling"]:
            if mode not in mode_dfs:
                continue

            dfp = mode_dfs[mode]
            T = dfp["Temperature(K)"].to_numpy(dtype=float)
            Cp = dfp["CalibratedCp"].to_numpy(dtype=float)

            in_xlim = (T >= xlim_min) & (T <= xlim_max) & np.isfinite(Cp)
            if in_xlim.any():
                visible_cp.append(Cp[in_xlim])

            color = mode_colors[mode]
            ax.plot(T, Cp, color=color, lw=1.5, alpha=0.75, label=f"{mode.capitalize()} (Cycle {cyc_num})")

            for side in ["before", "after"]:
                key = (name, mode, side)
                if key not in linear_baseline_functions:
                    continue
                Tmin, Tmax = LINEAR_BASELINE_RANGES[mode][side]
                T_fit = np.linspace(Tmin, Tmax, 200)
                ax.plot(
                    T_fit,
                    linear_baseline_functions[key](T_fit),
                    color=color,
                    linestyle=side_styles[side],
                    lw=2.5,
                    label=f"{mode.capitalize()} {side} transition",
                )

        ax.set_title(name)
        ax.set_xlabel(r"$\mathit{T}$ ($\mathrm{K}$)")
        ax.set_ylabel(r"$\mathit{C}_p$ ($\mathrm{J\,g^{-1}\,K^{-1}}$)")
        ax.set_xlim(xlim_min, xlim_max)

        if visible_cp:
            cp_concat = np.concatenate(visible_cp)
            ymin, ymax = float(np.min(cp_concat)), float(np.max(cp_concat))
            pad = max(0.1 * (ymax - ymin), 0.1)
            ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

        ax.legend(fontsize="small", ncol=2, frameon=False)
        fig.tight_layout()

        save_path = plot_dir / f"{sanitize_filename(name)}_CalibratedCp.svg"
        fig.savefig(save_path, dpi=300)
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Saved calibrated Cp plot: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate specific heat capacity Cp(T) from TA Instruments DSC 2500 TRIOS exports."
    )
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR, help="Directory containing TRIOS .txt files")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (defaults to config.OUTPUT_DIR)")
    parser.add_argument("--show", action="store_true", help="Display matplotlib windows interactively")
    args = parser.parse_args()

    if args.out is not None:
        out_dir = args.out
    elif args.data != cfg.DATA_DIR:
        out_dir = args.data / "output"
    else:
        out_dir = cfg.OUTPUT_DIR

    run_pipeline(data_dir=args.data, output_dir=out_dir, show_plots=args.show)


if __name__ == "__main__":
    main()