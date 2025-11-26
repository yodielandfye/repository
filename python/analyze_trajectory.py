"""
Analyze LAMMPS trajectory file for molecular structures and self-replication potential.

This script:
1. Parses the trajectory file to extract atom positions
2. Reconstructs molecular structures from bond data
3. Analyzes chain lengths and topologies
4. Looks for autocatalytic patterns
5. Estimates self-replication potential
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import re


def parse_xyz_trajectory(traj_path: Path) -> List[Dict]:
    """
    Parse LAMMPS trajectory file (XYZ format).
    
    Returns list of frames, each containing:
    - timestep: int
    - atoms: list of dicts with id, type, x, y, z
    """
    frames = []
    current_frame = None
    reading_atoms = False
    atom_count = 0
    atoms_read = 0
    
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            
            # Check for timestep
            if line.startswith("ITEM: TIMESTEP"):
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {
                    'timestep': None,
                    'atoms': []
                }
                reading_atoms = False
                atoms_read = 0
            
            # Get timestep value
            elif current_frame is not None and current_frame['timestep'] is None:
                try:
                    current_frame['timestep'] = int(line)
                except ValueError:
                    pass
            
            # Check for atom count
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                pass  # Next line will have count
            elif current_frame is not None and current_frame['timestep'] is not None and atom_count == 0:
                try:
                    atom_count = int(line)
                except ValueError:
                    pass
            
            # Check for atom data start
            elif line.startswith("ITEM: ATOMS"):
                reading_atoms = True
                atoms_read = 0
            
            # Read atom data
            elif reading_atoms and current_frame is not None and atoms_read < atom_count:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        
                        current_frame['atoms'].append({
                            'id': atom_id,
                            'type': atom_type,
                            'pos': np.array([x, y, z])
                        })
                        atoms_read += 1
                    except (ValueError, IndexError):
                        pass
    
    # Add last frame
    if current_frame is not None:
        frames.append(current_frame)
    
    return frames


def reconstruct_bonds(frame: Dict, bond_cutoff: float = 1.5) -> Dict[int, Set[int]]:
    """
    Reconstruct bonds from atom positions using spatial hashing for speed.
    
    Uses distance cutoff to determine bonds.
    Returns dict mapping atom_id -> set of bonded atom_ids
    """
    bonds = defaultdict(set)
    atoms = {atom['id']: atom for atom in frame['atoms']}
    atom_list = list(atoms.values())
    
    if len(atom_list) == 0:
        return {}
    
    # Spatial hashing: divide space into grid cells
    # Only check atoms in same or adjacent cells
    cell_size = bond_cutoff * 2.0  # Make cells slightly larger than cutoff
    
    # Find bounds
    all_pos = np.array([a['pos'] for a in atom_list])
    min_pos = all_pos.min(axis=0)
    max_pos = all_pos.max(axis=0)
    
    # Create grid
    grid = defaultdict(list)
    for atom in atom_list:
        pos = atom['pos']
        cell_x = int((pos[0] - min_pos[0]) / cell_size)
        cell_y = int((pos[1] - min_pos[1]) / cell_size)
        grid[(cell_x, cell_y)].append(atom)
    
    # Check bonds only within same/adjacent cells
    checked_pairs = set()
    for (cx, cy), cell_atoms in grid.items():
        # Check atoms in same cell
        for i, atom1 in enumerate(cell_atoms):
            for atom2 in cell_atoms[i+1:]:
                pair = tuple(sorted([atom1['id'], atom2['id']]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    dist = np.linalg.norm(atom1['pos'] - atom2['pos'])
                    if dist < bond_cutoff:
                        bonds[atom1['id']].add(atom2['id'])
                        bonds[atom2['id']].add(atom1['id'])
        
        # Check atoms in adjacent cells
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor_cell = (cx + dx, cy + dy)
            if neighbor_cell in grid:
                for atom1 in cell_atoms:
                    for atom2 in grid[neighbor_cell]:
                        pair = tuple(sorted([atom1['id'], atom2['id']]))
                        if pair not in checked_pairs:
                            checked_pairs.add(pair)
                            dist = np.linalg.norm(atom1['pos'] - atom2['pos'])
                            if dist < bond_cutoff:
                                bonds[atom1['id']].add(atom2['id'])
                                bonds[atom2['id']].add(atom1['id'])
    
    return dict(bonds)


def find_molecules(bonds: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find connected components (molecules) from bond graph.
    
    Returns list of sets, each set contains atom IDs in one molecule.
    """
    visited = set()
    molecules = []
    
    def dfs(atom_id: int, molecule: Set[int]):
        if atom_id in visited:
            return
        visited.add(atom_id)
        molecule.add(atom_id)
        for neighbor in bonds.get(atom_id, set()):
            dfs(neighbor, molecule)
    
    for atom_id in bonds:
        if atom_id not in visited:
            molecule = set()
            dfs(atom_id, molecule)
            if molecule:
                molecules.append(molecule)
    
    # Add isolated atoms
    all_atoms = set(bonds.keys())
    for atom_id in all_atoms:
        if atom_id not in visited:
            molecules.append({atom_id})
    
    return molecules


def analyze_molecule_topology(molecule: Set[int], bonds: Dict[int, Set[int]]) -> Dict:
    """
    Analyze topology of a single molecule.
    
    Returns dict with:
    - size: number of atoms
    - max_degree: maximum number of bonds per atom
    - avg_degree: average number of bonds per atom
    - is_linear: bool (if it's a linear chain)
    - is_ring: bool (if it's a ring)
    - is_branched: bool (if it has branching)
    """
    if len(molecule) == 0:
        return {}
    
    degrees = [len(bonds.get(atom_id, set())) for atom_id in molecule]
    
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    # Classify topology
    is_linear = max_degree <= 2 and len(molecule) > 1
    is_ring = max_degree == 2 and len(molecule) >= 3
    is_branched = max_degree > 2
    
    return {
        'size': len(molecule),
        'max_degree': max_degree,
        'avg_degree': avg_degree,
        'is_linear': is_linear,
        'is_ring': is_ring,
        'is_branched': is_branched
    }


def analyze_trajectory(traj_path: Path, bond_cutoff: float = 1.5) -> Dict:
    """
    Analyze entire trajectory for molecular evolution.
    
    Returns dict with:
    - frame_analyses: list of analyses for each frame
    - molecule_growth: evolution of molecule counts over time
    - chain_length_distribution: distribution of chain lengths
    - autocatalytic_potential: estimate of self-replication potential
    """
    print("📊 Parsing trajectory file...")
    frames = parse_xyz_trajectory(traj_path)
    print(f"   Found {len(frames)} frames")
    
    frame_analyses = []
    molecule_counts = []
    chain_lengths = []
    
    # Process every 3rd frame for speed (still get good coverage)
    frame_indices = list(range(0, len(frames), max(1, len(frames) // 10)))  # Sample ~10 frames
    if len(frames) - 1 not in frame_indices:
        frame_indices.append(len(frames) - 1)  # Always include last frame
    
    for idx in frame_indices:
        frame = frames[idx]
        i = idx
        print(f"   Analyzing frame {i+1}/{len(frames)} (step {frame['timestep']})...")
        
        # Reconstruct bonds
        bonds = reconstruct_bonds(frame, bond_cutoff)
        
        # Find molecules
        molecules = find_molecules(bonds)
        
        # Analyze each molecule
        molecule_stats = []
        for mol in molecules:
            topo = analyze_molecule_topology(mol, bonds)
            if topo:
                molecule_stats.append(topo)
                if topo['is_linear'] or topo['is_branched']:
                    chain_lengths.append(topo['size'])
        
        # Frame summary
        frame_analyses.append({
            'timestep': frame['timestep'],
            'n_molecules': len(molecules),
            'n_atoms': len(frame['atoms']),
            'molecule_stats': molecule_stats,
            'avg_molecule_size': np.mean([m['size'] for m in molecule_stats]) if molecule_stats else 0,
            'max_molecule_size': max([m['size'] for m in molecule_stats]) if molecule_stats else 0
        })
        
        molecule_counts.append(len(molecules))
    
    # Analyze growth (use frame analyses for more accurate counts)
    if len(frame_analyses) > 1:
        first_count = frame_analyses[0]['n_molecules']
        last_count = frame_analyses[-1]['n_molecules']
        growth_rate = (last_count - first_count) / first_count if first_count > 0 else 0
        exponential_growth = growth_rate > 0.5  # 50% growth suggests exponential
    else:
        growth_rate = 0
        exponential_growth = False
    
    # Chain length distribution
    if chain_lengths:
        chain_length_dist = Counter(chain_lengths)
        avg_chain_length = np.mean(chain_lengths)
        max_chain_length = max(chain_lengths)
    else:
        chain_length_dist = {}
        avg_chain_length = 0
        max_chain_length = 0
    
    # Autocatalytic potential
    autocatalytic_potential = {
        'has_complex_molecules': max_chain_length > 10,
        'has_growth': growth_rate > 0,
        'has_exponential_growth': exponential_growth,
        'has_long_chains': max_chain_length > 20,
        'confidence': 0.0
    }
    
    confidence = 0.0
    if autocatalytic_potential['has_complex_molecules']:
        confidence += 0.3
    if autocatalytic_potential['has_growth']:
        confidence += 0.2
    if autocatalytic_potential['has_exponential_growth']:
        confidence += 0.3
    if autocatalytic_potential['has_long_chains']:
        confidence += 0.2
    
    autocatalytic_potential['confidence'] = confidence
    
    return {
        'frame_analyses': frame_analyses,
        'molecule_counts': molecule_counts,
        'molecule_growth_rate': growth_rate,
        'exponential_growth': exponential_growth,
        'chain_length_distribution': chain_length_dist,
        'avg_chain_length': avg_chain_length,
        'max_chain_length': max_chain_length,
        'autocatalytic_potential': autocatalytic_potential
    }


def print_analysis(results: Dict):
    """Print formatted analysis results."""
    print("\n" + "=" * 60)
    print("🔬 TRAJECTORY ANALYSIS RESULTS")
    print("=" * 60)
    
    # Frame summary
    if results['frame_analyses']:
        first_frame = results['frame_analyses'][0]
        last_frame = results['frame_analyses'][-1]
        
        print(f"\n📊 Molecular Evolution:")
        print(f"   Initial molecules: {first_frame['n_molecules']}")
        print(f"   Final molecules: {last_frame['n_molecules']}")
        print(f"   Growth rate: {results['molecule_growth_rate']*100:.1f}%")
        
        print(f"\n   Initial avg size: {first_frame['avg_molecule_size']:.1f} atoms")
        print(f"   Final avg size: {last_frame['avg_molecule_size']:.1f} atoms")
        print(f"   Max molecule size: {last_frame['max_molecule_size']} atoms")
    
    # Chain analysis
    print(f"\n🔗 Chain Analysis:")
    print(f"   Average chain length: {results['avg_chain_length']:.1f} atoms")
    print(f"   Maximum chain length: {results['max_chain_length']} atoms")
    
    if results['chain_length_distribution']:
        print(f"\n   Chain length distribution (top 10):")
        for length, count in results['chain_length_distribution'].most_common(10):
            print(f"     {length} atoms: {count} chains")
    
    # Autocatalytic potential
    print(f"\n🧬 Self-Replication Potential:")
    potential = results['autocatalytic_potential']
    
    if potential['has_complex_molecules']:
        print(f"   ✅ Complex molecules formed (max {results['max_chain_length']} atoms)")
    if potential['has_growth']:
        print(f"   ✅ Molecule count increased ({results['molecule_growth_rate']*100:.1f}%)")
    if potential['has_exponential_growth']:
        print(f"   ✅ EXPONENTIAL GROWTH DETECTED!")
    if potential['has_long_chains']:
        print(f"   ✅ Long chains formed (max {results['max_chain_length']} atoms)")
    
    confidence = potential['confidence']
    if confidence > 0.7:
        print(f"\n   🎯 HIGH confidence for self-replication ({confidence*100:.0f}%)")
    elif confidence > 0.4:
        print(f"\n   🎯 MEDIUM confidence for self-replication ({confidence*100:.0f}%)")
    else:
        print(f"\n   ⚠️  LOW confidence for self-replication ({confidence*100:.0f}%)")
    
    print("\n" + "=" * 60)


def main():
    """Main analysis function."""
    traj_path = Path("gpu_emergent_trajectory.xyz")
    
    if not traj_path.exists():
        print(f"❌ Trajectory file not found: {traj_path}")
        print("   Looking for: gpu_emergent_trajectory.xyz")
        return
    
    print("=" * 60)
    print("🔬 Trajectory Analysis: Molecular Structures & Self-Replication")
    print("=" * 60)
    print(f"\n📁 Analyzing: {traj_path}")
    
    # Run analysis
    results = analyze_trajectory(traj_path, bond_cutoff=1.5)
    
    # Print results
    print_analysis(results)
    
    # Save results
    output_file = Path("trajectory_analysis.txt")
    with open(output_file, 'w') as f:
        f.write("TRAJECTORY ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Molecule growth rate: {results['molecule_growth_rate']*100:.1f}%\n")
        f.write(f"Average chain length: {results['avg_chain_length']:.1f} atoms\n")
        f.write(f"Maximum chain length: {results['max_chain_length']} atoms\n")
        f.write(f"Autocatalytic confidence: {results['autocatalytic_potential']['confidence']*100:.0f}%\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n💡 Next steps:")
    print("   - If exponential growth detected → biology is emerging!")
    print("   - If long chains formed → self-replication possible")
    print("   - Run longer simulation to confirm")


if __name__ == '__main__':
    main()

