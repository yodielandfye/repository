"""
Parse all cpu_run_*.log files and extract reaction kinetics.

Extracts:
- Step, Temp, Press, PotEng, TotEng, Volume
- Fits exponential decay: PE(t) = PE_final + (PE_init - PE_final)·exp(-k·t)
- Extracts reaction rate constant k for each T/Ea combination
"""

import re
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple


def parse_log_file(log_path: Path) -> pd.DataFrame:
    """
    Extract thermo data from LAMMPS log file.
    
    Returns DataFrame with columns: step, temp, press, pe, te, vol
    """
    data = []
    in_thermo_section = False
    
    with open(log_path) as f:
        for line in f:
            # Detect start of thermo output
            if "Step" in line and "Temp" in line and "Press" in line:
                in_thermo_section = True
                continue
            
            if not in_thermo_section:
                continue
            
            # Match thermo output: "Step Temp Press PotEng TotEng Volume"
            # Format: spaces, then numbers
            match = re.match(
                r'^\s*(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)',
                line
            )
            if match:
                try:
                    step, temp, press, pe, te, vol = map(float, match.groups())
                    data.append({
                        'step': int(step),
                        'temp': temp,
                        'press': press,
                        'pe': pe,
                        'te': te,
                        'vol': vol
                    })
                except ValueError:
                    # Skip lines that don't parse correctly
                    continue
    
    if not data:
        print(f"⚠️  Warning: No data found in {log_path}")
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def fit_reaction_rate(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit exponential decay to potential energy.
    
    Model: PE(t) = PE_final + (PE_init - PE_final) * exp(-k * t)
    
    Returns: (k, pe_final, pe_init) or (None, None, None) if fit fails
    """
    if len(df) < 10:
        return None, None, None
    
    def exp_decay(t, k, pe_final, pe_init):
        return pe_final + (pe_init - pe_final) * np.exp(-k * t)
    
    t = df['step'].values
    pe = df['pe'].values
    
    # Initial guess
    pe_init = pe[0]
    pe_final = pe[-1]
    # Estimate k from first and last points
    if pe_final != pe_init and len(pe) > 1:
        # Solve: pe_final + (pe_init - pe_final) * exp(-k * t_final) = pe_last
        # This is approximate, use as initial guess
        k_guess = -np.log((pe[-1] - pe_final) / (pe_init - pe_final)) / t[-1] if (pe_init - pe_final) != 0 else 1e-6
        k_guess = max(1e-8, min(1e-3, k_guess))  # Clamp to reasonable range
    else:
        k_guess = 1e-6
    
    try:
        # Bounds to keep parameters reasonable
        bounds = ([1e-10, -np.inf, -np.inf], [1e-2, np.inf, np.inf])
        popt, _ = curve_fit(
            exp_decay, t, pe,
            p0=[k_guess, pe_final, pe_init],
            bounds=bounds,
            maxfev=10000
        )
        k, pe_final_fit, pe_init_fit = popt
        return k, pe_final_fit, pe_init_fit
    except Exception as e:
        print(f"⚠️  Fit failed: {e}")
        return None, None, None


def extract_bond_metrics(df: pd.DataFrame) -> dict:
    """
    Extract bond formation metrics from volume changes.
    
    Volume decreases as bonds form (atoms get closer together).
    """
    if len(df) < 2:
        return {
            'vol_change_fraction': None,
            'estimated_bonds': None,
            'vol_init': None,
            'vol_final': None
        }
    
    vol_init = df['vol'].iloc[0]
    vol_final = df['vol'].iloc[-1]
    vol_change = (vol_init - vol_final) / vol_init if vol_init > 0 else 0
    
    # Rough estimate: each bond reduces volume by ~bond_length^3
    # Bond length ~1.0 in LJ units
    # This is a heuristic, not exact
    estimated_bonds = vol_change * vol_init / (1.0**3) if vol_change > 0 else 0
    
    return {
        'vol_change_fraction': vol_change,
        'estimated_bonds': estimated_bonds,
        'vol_init': vol_init,
        'vol_final': vol_final
    }


def get_parameters_from_run_id(run_id: int) -> Tuple[float, float]:
    """
    Extract T and Ea from run_id based on parallel_cpu_sweep.py logic.
    
    The sweep uses:
    - temperatures = [0.5, 1.0, 1.5, 2.0]
    - activation_energies = [2.0, 4.0, 6.0, 8.0]
    - Outer loop: temperatures, Inner loop: activation_energies
    
    So: run_id 0-3: T=0.5, Ea=[2,4,6,8]
        run_id 4-7: T=1.0, Ea=[2,4,6,8]
        etc.
    """
    temperatures = [0.5, 1.0, 1.5, 2.0]
    activation_energies = [2.0, 4.0, 6.0, 8.0]
    
    temp_idx = run_id // len(activation_energies)
    ea_idx = run_id % len(activation_energies)
    
    if temp_idx >= len(temperatures) or ea_idx >= len(activation_energies):
        print(f"⚠️  Warning: run_id {run_id} out of range")
        return None, None
    
    T = temperatures[temp_idx]
    Ea = activation_energies[ea_idx]
    
    return T, Ea


def analyze_all_runs(log_dir: Path = Path('.')) -> pd.DataFrame:
    """
    Process all CPU runs and extract kinetics.
    
    Returns DataFrame with columns:
    - run_id, T, Ea, k, pe_init, pe_final, n_steps, final_temp, vol_change_fraction, estimated_bonds
    """
    results = []
    
    log_files = sorted(log_dir.glob('cpu_run_*.log'))
    if not log_files:
        print(f"❌ No cpu_run_*.log files found in {log_dir}")
        return pd.DataFrame()
    
    print(f"📊 Found {len(log_files)} log files")
    
    for log_path in log_files:
        # Extract run_id from filename
        match = re.search(r'cpu_run_(\d+)', str(log_path))
        if not match:
            print(f"⚠️  Warning: Could not extract run_id from {log_path}")
            continue
        
        run_id = int(match.group(1))
        T, Ea = get_parameters_from_run_id(run_id)
        
        if T is None or Ea is None:
            continue
        
        print(f"  Processing run {run_id}: T={T:.2f}, Ea={Ea:.2f}")
        
        # Parse log file
        df = parse_log_file(log_path)
        if df.empty:
            print(f"    ⚠️  No data in {log_path}")
            continue
        
        # Fit reaction rate
        k, pe_final, pe_init = fit_reaction_rate(df)
        
        # Extract bond metrics
        bond_metrics = extract_bond_metrics(df)
        
        results.append({
            'run_id': run_id,
            'T': T,
            'Ea': Ea,
            'k': k,
            'pe_init': pe_init,
            'pe_final': pe_final,
            'n_steps': len(df),
            'final_temp': df['temp'].iloc[-1] if len(df) > 0 else None,
            'vol_change_fraction': bond_metrics['vol_change_fraction'],
            'estimated_bonds': bond_metrics['estimated_bonds'],
            'vol_init': bond_metrics['vol_init'],
            'vol_final': bond_metrics['vol_final'],
            'pe_min': df['pe'].min() if len(df) > 0 else None,
            'pe_max': df['pe'].max() if len(df) > 0 else None,
        })
    
    return pd.DataFrame(results)


def main():
    """Main analysis function."""
    print("=" * 60)
    print("🔬 CPU Run Analysis: Reaction Kinetics Extraction")
    print("=" * 60)
    
    # Determine log directory (could be in parent directory)
    log_dir = Path('.')
    if not list(log_dir.glob('cpu_run_*.log')):
        # Try parent directory
        log_dir = Path('..')
        if not list(log_dir.glob('cpu_run_*.log')):
            print("❌ No cpu_run_*.log files found in current or parent directory")
            print(f"   Searched: {Path('.').absolute()} and {Path('..').absolute()}")
            return
    
    print(f"📁 Using log directory: {log_dir.absolute()}")
    
    # Analyze all runs
    df = analyze_all_runs(log_dir)
    
    if df.empty:
        print("❌ No data extracted")
        return
    
    # Save results
    output_file = 'reaction_kinetics.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Summary Statistics")
    print("=" * 60)
    print(f"Total runs analyzed: {len(df)}")
    print(f"Successful fits: {df['k'].notna().sum()}")
    print(f"\nRate constants (k):")
    print(f"  Mean: {df['k'].mean():.6e}")
    print(f"  Min:  {df['k'].min():.6e}")
    print(f"  Max:  {df['k'].max():.6e}")
    print(f"\nTemperature range: {df['T'].min():.2f} - {df['T'].max():.2f}")
    print(f"Ea range: {df['Ea'].min():.2f} - {df['Ea'].max():.2f}")
    
    # Show data preview
    print("\n" + "=" * 60)
    print("📋 Data Preview (first 5 rows)")
    print("=" * 60)
    print(df[['run_id', 'T', 'Ea', 'k', 'pe_init', 'pe_final']].head().to_string())
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

