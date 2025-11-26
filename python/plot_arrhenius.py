"""
Plot Arrhenius analysis: k vs. T and k vs. Ea

Compares extracted reaction rate constants to Arrhenius theory:
k = A·exp(-Ea/RT)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Optional, Tuple


def arrhenius_temperature(T: np.ndarray, A: float, Ea: float) -> np.ndarray:
    """
    Arrhenius equation: k = A·exp(-Ea/RT)
    
    In LJ units, R ≈ 1.0
    """
    R = 1.0  # LJ units
    return A * np.exp(-Ea / (R * T))


def arrhenius_activation(Ea: np.ndarray, A: float, T: float) -> np.ndarray:
    """
    Arrhenius equation: k = A·exp(-Ea/RT)
    """
    R = 1.0  # LJ units
    return A * np.exp(-Ea / (R * T))


def fit_arrhenius_temperature(df_subset: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Fit Arrhenius equation to k vs. T data.
    
    Returns: (A, Ea) or None if fit fails
    """
    if len(df_subset) < 2:
        return None
    
    T = df_subset['T'].values
    k = df_subset['k'].values
    
    # Filter out invalid data
    valid = (k > 0) & np.isfinite(k) & np.isfinite(T)
    T = T[valid]
    k = k[valid]
    
    if len(T) < 2:
        return None
    
    # Initial guess: A from max k, Ea from slope
    A_guess = k.max() * 10
    # Estimate Ea from two points
    if len(T) >= 2:
        T_sorted = np.sort(T)
        k_sorted = k[np.argsort(T)]
        if k_sorted[0] > 0 and k_sorted[-1] > 0:
            Ea_guess = -np.log(k_sorted[-1] / k_sorted[0]) / (1/T_sorted[-1] - 1/T_sorted[0])
            Ea_guess = max(0.1, min(20.0, Ea_guess))
        else:
            Ea_guess = 5.0
    else:
        Ea_guess = 5.0
    
    try:
        popt, _ = curve_fit(
            arrhenius_temperature,
            T, k,
            p0=[A_guess, Ea_guess],
            bounds=([1e-6, 0.1], [1e15, 50.0]),
            maxfev=10000
        )
        return tuple(popt)
    except Exception:
        return None


def plot_arrhenius_analysis(csv_file: str = 'reaction_kinetics.csv'):
    """
    Create Arrhenius analysis plots.
    """
    # Load data
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        print("   Run analyze_cpu_data.py first to generate the CSV file")
        return
    
    df = pd.read_csv(csv_file)
    
    # Filter out invalid data
    df = df[df['k'].notna() & (df['k'] > 0)]
    
    if len(df) == 0:
        print("❌ No valid rate constant data found")
        return
    
    print(f"📊 Plotting {len(df)} data points")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ===== Plot 1: k vs. T (for fixed Ea) =====
    ax = axes[0, 0]
    temperatures = sorted(df['T'].unique())
    activation_energies = sorted(df['Ea'].unique())
    
    for Ea_val in activation_energies:
        subset = df[df['Ea'] == Ea_val]
        if len(subset) > 0:
            ax.plot(subset['T'], subset['k'], 'o-', label=f'Ea={Ea_val:.1f}', linewidth=2, markersize=8)
            
            # Fit Arrhenius if we have enough points
            if len(subset) >= 2:
                fit_result = fit_arrhenius_temperature(subset)
                if fit_result:
                    A_fit, Ea_fit = fit_result
                    T_fit = np.linspace(subset['T'].min(), subset['T'].max(), 100)
                    k_fit = arrhenius_temperature(T_fit, A_fit, Ea_fit)
                    ax.plot(T_fit, k_fit, '--', alpha=0.5, linewidth=1)
                    print(f"  Ea={Ea_val:.1f}: A={A_fit:.3e}, Ea_fit={Ea_fit:.2f}")
    
    ax.set_xlabel('Temperature T', fontsize=12)
    ax.set_ylabel('Rate constant k', fontsize=12)
    ax.set_title('Arrhenius: k vs. T (for fixed Ea)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: k vs. Ea (for fixed T) =====
    ax = axes[0, 1]
    
    for T_val in temperatures:
        subset = df[df['T'] == T_val]
        if len(subset) > 0:
            ax.plot(subset['Ea'], subset['k'], 's-', label=f'T={T_val:.1f}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Activation Energy Ea', fontsize=12)
    ax.set_ylabel('Rate constant k', fontsize=12)
    ax.set_title('k vs. Ea (for fixed T)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 3: Arrhenius plot (ln(k) vs. 1/T) =====
    ax = axes[1, 0]
    
    for Ea_val in activation_energies:
        subset = df[df['Ea'] == Ea_val]
        if len(subset) > 0:
            inv_T = 1.0 / subset['T'].values
            ln_k = np.log(subset['k'].values)
            ax.plot(inv_T, ln_k, 'o-', label=f'Ea={Ea_val:.1f}', linewidth=2, markersize=8)
            
            # Fit linear: ln(k) = ln(A) - Ea/(R*T)
            if len(subset) >= 2:
                try:
                    coeffs = np.polyfit(inv_T, ln_k, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    Ea_fit = -slope  # R=1.0 in LJ units
                    A_fit = np.exp(intercept)
                    
                    inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
                    ln_k_fit = intercept + slope * inv_T_fit
                    ax.plot(inv_T_fit, ln_k_fit, '--', alpha=0.5, linewidth=1)
                except:
                    pass
    
    ax.set_xlabel('1/T (inverse temperature)', fontsize=12)
    ax.set_ylabel('ln(k)', fontsize=12)
    ax.set_title('Arrhenius Plot: ln(k) vs. 1/T', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 4: Heatmap of k vs. T and Ea =====
    ax = axes[1, 1]
    
    # Create pivot table for heatmap
    pivot = df.pivot_table(values='k', index='Ea', columns='T', aggfunc='mean')
    
    if not pivot.empty:
        im = ax.imshow(
            pivot.values,
            extent=[df['T'].min(), df['T'].max(), df['Ea'].min(), df['Ea'].max()],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax.set_xlabel('Temperature T', fontsize=12)
        ax.set_ylabel('Activation Energy Ea', fontsize=12)
        ax.set_title('Rate Constant Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='k')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'arrhenius_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved {output_file}")
    
    # Also save as PDF for publication
    output_file_pdf = 'arrhenius_analysis.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved {output_file_pdf}")
    
    # Don't show plot in headless environment
    try:
        plt.show()
    except:
        print("   (Skipping display - running in headless mode)")


def main():
    """Main plotting function."""
    print("=" * 60)
    print("📈 Arrhenius Analysis: Reaction Kinetics Visualization")
    print("=" * 60)
    
    # Check if CSV exists
    csv_file = 'reaction_kinetics.csv'
    if not Path(csv_file).exists():
        # Try parent directory
        csv_file = Path('..') / 'reaction_kinetics.csv'
        if not csv_file.exists():
            print(f"❌ File not found: reaction_kinetics.csv")
            print("   Run analyze_cpu_data.py first to generate the CSV file")
            print(f"   Searched: {Path('.').absolute() / 'reaction_kinetics.csv'}")
            print(f"   Searched: {Path('..').absolute() / 'reaction_kinetics.csv'}")
            return
    
    print(f"📁 Using CSV file: {Path(csv_file).absolute()}")
    plot_arrhenius_analysis(str(csv_file))


if __name__ == '__main__':
    main()

