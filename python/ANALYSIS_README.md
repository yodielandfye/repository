# Reaction Kinetics Analysis Scripts

## Overview

These scripts analyze LAMMPS simulation logs to extract reaction kinetics and validate Arrhenius theory.

## Files

- `analyze_cpu_data.py` - Parses CPU run logs and extracts reaction rate constants
- `plot_arrhenius.py` - Creates Arrhenius analysis plots

## Dependencies

Install required packages:
```bash
pip install scipy matplotlib pandas
# or
conda install scipy matplotlib pandas
```

## Usage

### Step 1: Extract Reaction Kinetics

```bash
cd /home/yeatcomeandgo/boolaka/python
python analyze_cpu_data.py
```

This will:
- Parse all `cpu_run_*.log` files (in current or parent directory)
- Extract thermo data (step, temp, press, pe, te, vol)
- Fit exponential decay: PE(t) = PE_final + (PE_init - PE_final)·exp(-k·t)
- Extract reaction rate constant k for each T/Ea combination
- Save results to `reaction_kinetics.csv`

**Output:**
- `reaction_kinetics.csv` - DataFrame with columns:
  - `run_id`, `T`, `Ea`, `k`, `pe_init`, `pe_final`, `n_steps`, `final_temp`
  - `vol_change_fraction`, `estimated_bonds`, `vol_init`, `vol_final`

### Step 2: Plot Arrhenius Analysis

```bash
python plot_arrhenius.py
```

This will:
- Load `reaction_kinetics.csv`
- Create 4 plots:
  1. **k vs. T** (for fixed Ea) - Shows Arrhenius temperature dependence
  2. **k vs. Ea** (for fixed T) - Shows activation energy dependence
  3. **ln(k) vs. 1/T** - Classic Arrhenius plot (should be linear)
  4. **Heatmap** - k as function of both T and Ea
- Save plots as `arrhenius_analysis.png` and `.pdf`

## Expected Results

### Reaction Rate Constants

The exponential decay fit extracts rate constants (k) that should follow Arrhenius equation:
```
k = A·exp(-Ea/RT)
```

Where:
- `A` = Pre-exponential factor (set to 1e10 in simulations)
- `Ea` = Activation energy (varied: 2.0, 4.0, 6.0, 8.0)
- `R` = Gas constant (≈1.0 in LJ units)
- `T` = Temperature (varied: 0.5, 1.0, 1.5, 2.0)

### Validation

The plots should show:
- **k increases with T** (higher temperature = faster reactions)
- **k decreases with Ea** (higher activation energy = slower reactions)
- **ln(k) vs. 1/T is linear** (classic Arrhenius behavior)

## Troubleshooting

### "No cpu_run_*.log files found"
- Make sure log files are in the current directory or parent directory
- Check that files are named `cpu_run_0.log`, `cpu_run_1.log`, etc.

### "No valid rate constant data found"
- Check that log files contain thermo output
- Verify that potential energy (PE) decreases over time (indicating bond formation)

### "Fit failed" warnings
- Some runs may have insufficient data or unusual behavior
- Check individual log files to see if they completed successfully

## Example Output

```
📊 Found 16 log files
  Processing run 0: T=0.50, Ea=2.00
  Processing run 1: T=0.50, Ea=4.00
  ...
✅ Results saved to reaction_kinetics.csv

📊 Summary Statistics
Total runs analyzed: 16
Successful fits: 16
Rate constants (k):
  Mean: 1.234e-06
  Min:  5.678e-07
  Max:  2.345e-06
```


