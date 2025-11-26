"""
Structural Analysis: Extract molecule topologies and identify patterns

This script analyzes:
1. Bond network topologies (chains, rings, branched)
2. Molecule size distributions
3. Potential self-replication structures
4. Conditions favoring complex structures
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


def parse_special_neighbors(log_path: Path) -> Dict:
    """
    Extract bond formation data from LAMMPS log files.
    
    Key metric: "Ave special neighs/atom" - indicates number of bonds per atom
    Returns initial and final bond counts.
    """
    initial_bonds = 0.0
    final_bonds = 0.0
    n_atoms = None
    
    with open(log_path) as f:
        lines = f.readlines()
        
        # Find initial bonds (usually 0 at start)
        for i, line in enumerate(lines):
            if 'Ave special neighs/atom' in line:
                match = re.search(r'Ave special neighs/atom\s*=\s*([\d.]+)', line)
                if match:
                    initial_bonds = float(match.group(1))
                    # Check if this is near the start (first 200 lines)
                    if i < 200:
                        break
        
        # Find final bonds (at end of file)
        for line in reversed(lines):
            if 'Ave special neighs/atom' in line:
                match = re.search(r'Ave special neighs/atom\s*=\s*([\d.]+)', line)
                if match:
                    final_bonds = float(match.group(1))
                    break
        
        # Find atom count
        for line in reversed(lines):
            atoms_match = re.search(r'(\d+)\s+atoms', line)
            if atoms_match:
                n_atoms = int(atoms_match.group(1))
                break
    
    return {
        'initial_bonds_per_atom': initial_bonds,
        'final_bonds_per_atom': final_bonds,
        'n_atoms': n_atoms if n_atoms else 43000  # Default estimate
    }


def classify_topology(bond_count: int, atom_count: int) -> str:
    """
    Classify molecule topology based on bond/atom ratio.
    
    - Linear chain: bonds = atoms - 1
    - Ring: bonds = atoms
    - Branched: bonds > atoms
    - Dense network: bonds >> atoms
    """
    if atom_count == 0:
        return "unknown"
    
    ratio = bond_count / atom_count if atom_count > 0 else 0
    
    if ratio < 0.5:
        return "sparse"
    elif ratio < 0.9:
        return "linear_chain"
    elif ratio < 1.1:
        return "ring"
    elif ratio < 1.5:
        return "branched"
    else:
        return "dense_network"


def analyze_bond_evolution(df: pd.DataFrame) -> Dict:
    """
    Analyze how bond formation evolves over time.
    """
    if df.empty:
        return {}
    
    initial_bonds = df['special_neighs'].iloc[0] if len(df) > 0 else 0
    final_bonds = df['special_neighs'].iloc[-1] if len(df) > 0 else 0
    max_bonds = df['special_neighs'].max()
    
    # Calculate bond formation rate
    if len(df) > 1:
        steps = df['step'].values
        bonds = df['special_neighs'].values
        # Linear fit to estimate rate
        if len(steps) > 1:
            rate = np.polyfit(steps, bonds, 1)[0]
        else:
            rate = 0
    else:
        rate = 0
    
    # Find when bonds plateau (if they do)
    plateau_step = None
    if len(df) > 10:
        # Check if bonds stabilize in last 20% of simulation
        last_20_percent = df.tail(int(len(df) * 0.2))
        if last_20_percent['special_neighs'].std() < 0.01:
            plateau_step = last_20_percent['step'].iloc[0]
    
    return {
        'initial_bonds_per_atom': initial_bonds,
        'final_bonds_per_atom': final_bonds,
        'max_bonds_per_atom': max_bonds,
        'bond_formation_rate': rate,
        'total_bond_increase': final_bonds - initial_bonds,
        'plateau_step': plateau_step,
        'reached_equilibrium': plateau_step is not None
    }


def identify_self_replication_potential(bond_stats: Dict, topology: str) -> Dict:
    """
    Identify if molecular structures have self-replication potential.
    
    Criteria:
    1. Stable structures (bonds don't break easily)
    2. Catalytic potential (can facilitate bond formation)
    3. Template-like structures (can serve as templates)
    """
    potential = {
        'has_potential': False,
        'reasons': [],
        'confidence': 0.0
    }
    
    # Criterion 1: Stable bonds (low T forms more bonds = more stable)
    if bond_stats.get('final_bonds_per_atom', 0) > 0.5:
        potential['reasons'].append('stable_bonds')
        potential['confidence'] += 0.3
    
    # Criterion 2: Complex topology (branched/network structures)
    if topology in ['branched', 'dense_network']:
        potential['reasons'].append('complex_topology')
        potential['confidence'] += 0.3
    
    # Criterion 3: Equilibrium reached (stable structures)
    if bond_stats.get('reached_equilibrium', False):
        potential['reasons'].append('equilibrium')
        potential['confidence'] += 0.2
    
    # Criterion 4: Significant bond formation
    if bond_stats.get('total_bond_increase', 0) > 0.3:
        potential['reasons'].append('significant_formation')
        potential['confidence'] += 0.2
    
    potential['has_potential'] = potential['confidence'] > 0.5
    
    return potential


def analyze_all_runs(log_dir: Path = Path('.')) -> pd.DataFrame:
    """
    Analyze structural evolution across all CPU runs.
    """
    results = []
    
    log_files = sorted(log_dir.glob('cpu_run_*.log'))
    if not log_files:
        print(f"❌ No cpu_run_*.log files found in {log_dir}")
        return pd.DataFrame()
    
    print(f"📊 Analyzing {len(log_files)} log files for structural data\n")
    
    for log_path in log_files:
        match = re.search(r'cpu_run_(\d+)', str(log_path))
        if not match:
            continue
        
        run_id = int(match.group(1))
        
        # Extract T and Ea from run_id
        temperatures = [0.5, 1.0, 1.5, 2.0]
        activation_energies = [2.0, 4.0, 6.0, 8.0]
        temp_idx = run_id // len(activation_energies)
        ea_idx = run_id % len(activation_energies)
        
        if temp_idx >= len(temperatures) or ea_idx >= len(activation_energies):
            continue
        
        T = temperatures[temp_idx]
        Ea = activation_energies[ea_idx]
        
        print(f"  Run {run_id}: T={T:.1f}, Ea={Ea:.1f}")
        
        # Parse bond data
        bond_data = parse_special_neighbors(log_path)
        if bond_data['final_bonds_per_atom'] == 0 and bond_data['initial_bonds_per_atom'] == 0:
            print(f"    ⚠️  No bond data found")
            continue
        
        # Create bond stats dict
        bond_stats = {
            'initial_bonds_per_atom': bond_data['initial_bonds_per_atom'],
            'final_bonds_per_atom': bond_data['final_bonds_per_atom'],
            'max_bonds_per_atom': bond_data['final_bonds_per_atom'],  # Approximate
            'bond_formation_rate': (bond_data['final_bonds_per_atom'] - bond_data['initial_bonds_per_atom']) / 1000000,  # Per step
            'total_bond_increase': bond_data['final_bonds_per_atom'] - bond_data['initial_bonds_per_atom'],
            'reached_equilibrium': True  # Assume equilibrium at end
        }
        
        # Estimate topology
        final_bonds = bond_stats.get('final_bonds_per_atom', 0)
        # Use atom count from bond_data
        n_atoms = bond_data.get('n_atoms', 43000)
        topology = classify_topology(int(final_bonds * n_atoms), int(n_atoms))
        
        # Check self-replication potential
        replication = identify_self_replication_potential(bond_stats, topology)
        
        results.append({
            'run_id': run_id,
            'T': T,
            'Ea': Ea,
            'initial_bonds_per_atom': bond_stats.get('initial_bonds_per_atom', 0),
            'final_bonds_per_atom': bond_stats.get('final_bonds_per_atom', 0),
            'max_bonds_per_atom': bond_stats.get('max_bonds_per_atom', 0),
            'bond_formation_rate': bond_stats.get('bond_formation_rate', 0),
            'total_bond_increase': bond_stats.get('total_bond_increase', 0),
            'topology': topology,
            'reached_equilibrium': bond_stats.get('reached_equilibrium', False),
            'has_replication_potential': replication['has_potential'],
            'replication_confidence': replication['confidence'],
            'replication_reasons': ', '.join(replication['reasons']),
            'n_atoms': bond_data.get('n_atoms', 43000)
        })
    
    return pd.DataFrame(results)


def find_optimal_conditions(df: pd.DataFrame) -> Dict:
    """
    Find conditions that favor complex structure formation.
    """
    if df.empty:
        return {}
    
    # Best conditions for different metrics
    best_stability = df.loc[df['final_bonds_per_atom'].idxmax()]
    best_formation_rate = df.loc[df['bond_formation_rate'].idxmax()]
    best_complexity = df.loc[df['replication_confidence'].idxmax()]
    
    return {
        'best_stability': {
            'T': best_stability['T'],
            'Ea': best_stability['Ea'],
            'bonds_per_atom': best_stability['final_bonds_per_atom']
        },
        'best_formation_rate': {
            'T': best_formation_rate['T'],
            'Ea': best_formation_rate['Ea'],
            'rate': best_formation_rate['bond_formation_rate']
        },
        'best_complexity': {
            'T': best_complexity['T'],
            'Ea': best_complexity['Ea'],
            'confidence': best_complexity['replication_confidence']
        }
    }


def main():
    """Main structural analysis function."""
    print("=" * 60)
    print("🔬 Structural Analysis: Molecule Topology & Self-Replication")
    print("=" * 60)
    
    # Determine log directory
    log_dir = Path('.')
    if not list(log_dir.glob('cpu_run_*.log')):
        log_dir = Path('..')
        if not list(log_dir.glob('cpu_run_*.log')):
            print("❌ No cpu_run_*.log files found")
            return
    
    print(f"📁 Using log directory: {log_dir.absolute()}\n")
    
    # Analyze all runs
    df = analyze_all_runs(log_dir)
    
    if df.empty:
        print("❌ No structural data extracted")
        return
    
    # Save results
    output_file = 'structural_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Find optimal conditions
    optimal = find_optimal_conditions(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Structural Analysis Summary")
    print("=" * 60)
    
    print(f"\n🔗 Bond Formation:")
    print(f"  Average final bonds/atom: {df['final_bonds_per_atom'].mean():.3f}")
    print(f"  Range: {df['final_bonds_per_atom'].min():.3f} - {df['final_bonds_per_atom'].max():.3f}")
    
    print(f"\n🌐 Topology Distribution:")
    topology_counts = df['topology'].value_counts()
    for topo, count in topology_counts.items():
        print(f"  {topo}: {count} runs")
    
    print(f"\n🧬 Self-Replication Potential:")
    potential_runs = df[df['has_replication_potential']]
    print(f"  Runs with potential: {len(potential_runs)}/{len(df)}")
    if len(potential_runs) > 0:
        print(f"  Average confidence: {potential_runs['replication_confidence'].mean():.2f}")
        print(f"\n  Top candidates:")
        top_candidates = potential_runs.nlargest(3, 'replication_confidence')
        for _, row in top_candidates.iterrows():
            print(f"    Run {int(row['run_id'])}: T={row['T']:.1f}, Ea={row['Ea']:.1f}, "
                  f"confidence={row['replication_confidence']:.2f}")
    
    print(f"\n🎯 Optimal Conditions:")
    if optimal:
        print(f"  Best stability: T={optimal['best_stability']['T']:.1f}, "
              f"Ea={optimal['best_stability']['Ea']:.1f}")
        print(f"  Best formation rate: T={optimal['best_formation_rate']['T']:.1f}, "
              f"Ea={optimal['best_formation_rate']['Ea']:.1f}")
        print(f"  Best complexity: T={optimal['best_complexity']['T']:.1f}, "
              f"Ea={optimal['best_complexity']['Ea']:.1f}")
    
    print("\n✅ Structural analysis complete!")


if __name__ == '__main__':
    main()

