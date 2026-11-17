# TA Instruments DSC 2500 Specific Heat Capacity ($c_p$) Calibration & Analysis away (before and after) a first-order transition

An automated Python pipeline built with **NumPy**, **Pandas**, **SciPy**, and **Matplotlib** for **TA Instruments DSC 2500 TRIOS `.txt` exports**. Designed to replace manual spreadsheet and Origin plotting workflows, this tool batch-processes raw thermal scans, selects optimal heating–cooling cycles, calibrates specific heat capacity ($c_p$) against synthetic sapphire ($\alpha\text{-Al}_2\mathrm{O}_3$), fits linear pre- and post-transition baselines, and exports publication-ready vector figures.

---

## Key Features & Workflow Automation

* **End-to-End Batch Processing (`pandas` & `numpy`):** Scans entire experimental directories, parses specimen names and masses from TRIOS headers (converting `g` to `mg` when needed), normalizes mixed Kelvin/Celsius exports, and processes multiple samples in a single run.
* **Objective Cycle Selection (`numpy`):** Evaluates scan-rate-normalized heating–cooling baseline symmetry with numerical derivative masking (`np.gradient`) to automatically isolate and select the cleanest thermal cycle per specimen.
* **Spline-Based Sapphire Calibration (`scipy`):** Uses a 97-point natural cubic spline (`scipy.interpolate.CubicSpline`, $90\text{–}1650\text{ K}$) of NIST/NBS synthetic sapphire reference data and vectorized linear interpolation (`np.interp`) with strict out-of-bounds `NaN` guarding.
* **Automated Baseline Regression & Vector Figures (`matplotlib`):** Fits ordinary least-squares baselines ($c_p(T) = aT + b$) before and after phase transitions and exports standardized `.csv`/`.txt` datasets, fixed-width ASCII summary tables, and `.svg` plots without blocking batch execution.

---

## Methodology & Analysis Pipeline

### 1. TRIOS File Parsing & Automatic Unit Detection (`TAdsc_io.py`)
* **Ramp Segmentation:** Scans TRIOS `.txt` exports for `Ramp` command headers, extracts the scan rate $\beta$ (converted to $\mathrm{K\,s^{-1}}$), and classifies sequential heating ($\mathrm{d}T/\mathrm{d}t > 0$) and cooling ($\mathrm{d}T/\mathrm{d}t < 0$) segments from median start and end temperatures.
* **Automatic Kelvin / Celsius Handling:** Inspects the TRIOS unit row, column headers, and `Ramp` command strings to distinguish Kelvin (`K`) from Celsius (`°C`). All temperature arrays are standardized internally to Kelvin ($`T[\mathrm{K}] = T[^\circ\mathrm{C}] + 273.15`$ when exported in Celsius, or preserved directly when already in Kelvin).

### 2. Optimal Heating–Cooling Cycle Selection
Multi-cycle DSC experiments frequently exhibit first-cycle thermal history or pan-settling drift. Sequential heating and cooling segments are paired into complete cycles and scored across `[TMIN, TMAX]`:

1. Heating and cooling heat-flow curves ($`\dot{Q}_{\text{heat}}`$ and $`\dot{Q}_{\text{cool}}`$, in $\mathrm{mW}$) are interpolated (`np.interp`) onto a common overlapping temperature grid.
2. Cooling heat flow is scaled by the scan-rate ratio $`\beta_{\text{heat}} / \beta_{\text{cool}}`$ so sensible heat contributions cancel even when heating and cooling rates differ:

$$\dot{Q}_{\text{asym}}(T) = \dot{Q}_{\text{heat}}(T) + \dot{Q}_{\text{cool}}(T)\,\frac{\beta_{\text{heat}}}{\beta_{\text{cool}}}$$

3. First order phase-transition peaks are masked where $`\vert{}\mathrm{d}\dot{Q}_{\text{asym}}/\mathrm{d}T\vert{} >`$ `DERIV_THRESHOLD`.
4. The cycle minimizing the mean absolute baseline residual $`\langle \vert{}\dot{Q}_{\text{asym}}(T)\vert{} \rangle`$ is selected automatically.

### 3. Blank Subtraction & Sapphire Calibration Factor $E(T)$
Specific heat capacity calibration follows the standard ratio method (ASTM E1269) independently for heating and cooling scans:

1. **Optional Empty-Pan Subtraction:** If a blank/empty-pan run is present in `DATA_DIR`, its interpolated heat flow $`\dot{Q}_{\text{blank}}(T)`$ is subtracted from both the sapphire standard and sample curves (otherwise $`\dot{Q}_{\text{blank}}(T) = 0`$):

$$\dot{Q}_{\text{net}}(T) = \dot{Q}_{\text{raw}}(T) - \dot{Q}_{\text{blank}}(T)$$

2. **Temperature-Dependent Calibration Factor $E(T)$:**

$$E(T) = \frac{C_{p,\text{sapphire, ref}}(T)}{\dot{Q}_{\text{sapphire, net}}(T) / (m_{\text{sapphire}} \cdot \beta_{\text{sapphire}})}$$

   For cooling scans, the sign of $E_{\text{cooling}}(T)$ is inverted (`FLIP_COOLING_CALIB_SIGN = True`) so $E(T) > 0$ in both scan directions.

3. **Sample $c_p(T)$ Calibration:**

$$c_{p,\text{sample}}(T) = E(T) \cdot \frac{\dot{Q}_{\text{sample, net}}(T)}{m_{\text{sample}} \cdot \beta_{\text{sample}}}$$

   Cooling $c_p(T)$ values are sign-corrected during calibration so both heating and cooling exports are reported on a unified positive $c_p$ scale. Temperatures outside the sapphire calibration window are set to `NaN` rather than extrapolated.

### 4. Built-In Synthetic Sapphire Reference Standard
The pipeline embeds a 97-point reference table of synthetic sapphire ($\alpha\text{-Al}_2\mathrm{O}_3$, corundum) specific heat capacity spanning **$90\text{–}1650\text{ K}$**, interpolated via natural cubic splines (`scipy.interpolate.CubicSpline`, `extrapolate=False`):
* **Material:** National Bureau of Standards (NBS) synthetic sapphire ($\alpha\text{-Al}_2\mathrm{O}_3$), $99.98\text{–}99.99\text{ wt\%}$ purity, molecular weight $101.9613\text{ g\,mol}^{-1}$, at $1\text{ atm}$.
* **Citation:** D. A. Ditmars, S. Ishihara, S. S. Chang, G. Bernstein, and E. D. West, *"Enthalpy and Heat-Capacity Standard Reference Material: Synthetic Sapphire ($\alpha\text{-Al}_2\mathrm{O}_3$) from 10 to 2250 K,"* *Journal of Research of the National Bureau of Standards*, **87**(2), 159–163 (1982).

### 5. Pre- and Post-Transition Linear Heat capacity function Fitting (`cp_analysis.py`)
Ordinary least-squares linear regression (`numpy.polyfit`) is performed over user-defined temperature windows before and after the phase transition (`Cp_ranges_HEATING` and `Cp_ranges_COOLING`):

$$c_p(T) = a\,T + b$$

The slope $a$ ($\mathrm{J\,g^{-1}\,K^{-2}}$), intercept $b$ ($\mathrm{J\,g^{-1}\,K^{-1}}$), and coefficient of determination ($R^2$) are exported for each branch.

---

## Repository Structure

```text
3run_dsc-thermal-analysis-pipeline/
├── config.py            # Default directories, sample filters, and Cp linear fit temperature ranges
├── TAdsc_io.py          # TRIOS .txt parser, unit detector, cycle selector, and sapphire calibration
├── cp_analysis.py       # CLI entry point: runs calibration, baseline regression, exports, and plots
├── requirements.txt     # Python dependencies (numpy, pandas, scipy, matplotlib)
├── README.md            # Methodology and usage documentation
└── .gitignore           # Excludes __pycache__, local data, and generated outputs
```

---

## Installation

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Place your TRIOS exports of `.txt` files (sapphire standard, sample runs, and optional empty-pan run) in `./Exampledata` (or configure `DATA_DIR` in `config.py`).
2. Adjust temperature range filters and baseline  intervals (`Cp_ranges_HEATING`, `Cp_ranges_COOLING`) in `config.py`.
3. Run the pipeline from the terminal:

```bash
# Run using defaults in config.py (saves SVG plots without blocking)
python cp_analysis.py

# Specify custom input/output directories or display interactive plot windows
python cp_analysis.py --data ./Exampledata --out ./Exampledata/output --show
```

---

## Outputs

All outputs are written to `OUTPUT_DIR` (`./Exampledata/output` by default):

* **`calibrated_cp/<Sample>_heating.csv` & `.txt`** (and `_cooling`): Full calibrated data tables (`Temperature(K)`, `Temperature(C)`, `NetHeatFlow(mW)`, `Raw_dQdT`, `CalibrationFactor_E`, `CalibratedCp`).
* **`linear_cp_equations.txt`**: Plain-text report listing the sapphire literature reference, fitted linear equations $C_p(T) = aT + b$, $R^2$ values, and a fixed-width ASCII summary table.
* **`linear_cp_summary.csv`**: Machine-readable CSV summary of all fitted baseline parameters.
* **`plots/Sapphire_Calibration_Factor_E.svg`**: Heating and cooling sapphire calibration factors $E(T)$, which should be close to 1.
* **`plots/<Sample>_CalibratedCp.svg`**: Calibrated heating and cooling $C_p(T)$ curves overlaid with fitted pre- and post-transition baselines.

---

## References

1. D. A. Ditmars, S. Ishihara, S. S. Chang, G. Bernstein, and E. D. West, *"Enthalpy and Heat-Capacity Standard Reference Material: Synthetic Sapphire ($\alpha\text{-Al}_2\mathrm{O}_3$) from 10 to 2250 K,"* *J. Res. Natl. Bur. Stand.*, **87**(2), 159–163 (1982).
2. ASTM E1269, *"Standard Test Method for Determining Specific Heat Capacity by Differential Scanning Calorimetry,"* ASTM International.
