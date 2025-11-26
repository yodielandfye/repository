"""
Quick trajectory analysis - processes first and last frames only.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def parse_frame(traj_path: Path, frame_number: int = 0):
    """Parse a specific frame from trajectory."""
    frames = []
    current_frame = None
    reading_atoms = False
    atom_count = 0
    atoms_read = 0
    frame_idx = -1
    
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("ITEM: TIMESTEP"):
                if current_frame is not None:
                    frames.append(current_frame)
                    if len(frames) > frame_number + 1:
                        break
                current_frame = {'timestep': None, 'atoms': []}
                reading_atoms = False
                atoms_read = 0
                frame_idx += 1
            
            elif current_frame is not None and current_frame['timestep'] is None:
                try:
                    current_frame['timestep'] = int(line)
                except ValueError:
                    pass
            
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                atom_count = 0
            elif current_frame is not None and atom_count == 0 and current_frame['timestep'] is not None:
                try:
                    atom_count = int(line)
                except ValueError:
                    pass
            
            elif line.startswith("ITEM: ATOMS"):
                reading_atoms = True
                atoms_read = 0
            
            elif reading_atoms and current_frame is not None:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        atom_id = int(parts[0])
                        atom_type = int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        current_frame['atoms'].append({
                            'id': atom_id, 'type': atom_type,
                            'pos': np.array([x, y, z])
                        })
                        atoms_read += 1
                        if atoms_read >= atom_count:
                            reading_atoms = False
                    except (ValueError, IndexError):
                        pass
    
    if current_frame is not None:
        frames.append(current_frame)
    
    return frames[frame_number] if frame_number < len(frames) else None

def find_bonds(frame, cutoff=1.5):
    """Find bonds from distances."""
    bonds = defaultdict(set)
    atoms = frame['atoms']
    for i, a1 in enumerate(atoms):
        for a2 in atoms[i+1:]:
            dist = np.linalg.norm(a1['pos'] - a2['pos'])
            if dist < cutoff:
                bonds[a1['id']].add(a2['id'])
                bonds[a2['id']].add(a1['id'])
    return dict(bonds)

def find_molecules(bonds):
    """Find connected components."""
    visited = set()
    molecules = []
    
    def dfs(atom_id, mol):
        if atom_id in visited:
            return
        visited.add(atom_id)
        mol.add(atom_id)
        for n in bonds.get(atom_id, set()):
            dfs(n, mol)
    
    for atom_id in bonds:
        if atom_id not in visited:
            mol = set()
            dfs(atom_id, mol)
            if mol:
                molecules.append(mol)
    
    return molecules

# Quick analysis
print("=" * 60)
print("🔬 QUICK TRAJECTORY ANALYSIS")
print("=" * 60)

traj_path = Path("gpu_emergent_trajectory.xyz")
print(f"\n📁 Analyzing: {traj_path}")

# Get first and last frames
print("\n📊 Parsing frames...")
first_frame = parse_frame(traj_path, 0)
last_frame = parse_frame(traj_path, -1)  # Will get last frame

if first_frame is None or last_frame is None:
    print("❌ Could not parse frames")
    exit(1)

print(f"   First frame: step {first_frame['timestep']}")
print(f"   Last frame: step {last_frame['timestep']}")

# Analyze bonds
print("\n🔗 Finding bonds...")
first_bonds = find_bonds(first_frame, cutoff=1.5)
last_bonds = find_bonds(last_frame, cutoff=1.5)

print(f"   First frame bonds: {sum(len(v) for v in first_bonds.values()) // 2}")
print(f"   Last frame bonds: {sum(len(v) for v in last_bonds.values()) // 2}")

# Find molecules
print("\n🧪 Finding molecules...")
first_mols = find_molecules(first_bonds)
last_mols = find_molecules(last_bonds)

print(f"   First frame molecules: {len(first_mols)}")
print(f"   Last frame molecules: {len(last_mols)}")

# Chain lengths
first_sizes = [len(m) for m in first_mols]
last_sizes = [len(m) for m in last_mols]

print(f"\n📏 Chain Lengths:")
print(f"   First frame:")
print(f"     Average: {np.mean(first_sizes):.1f} atoms")
print(f"     Max: {max(first_sizes) if first_sizes else 0} atoms")
print(f"   Last frame:")
print(f"     Average: {np.mean(last_sizes):.1f} atoms")
print(f"     Max: {max(last_sizes) if last_sizes else 0} atoms")

# Growth
growth = (len(last_mols) - len(first_mols)) / len(first_mols) * 100 if first_mols else 0
print(f"\n📈 Growth:")
print(f"   Molecule count change: {len(last_mols) - len(first_mols)} ({growth:+.1f}%)")

# Top chains
if last_sizes:
    top_chains = sorted(last_sizes, reverse=True)[:10]
    print(f"\n🏆 Top 10 Chain Lengths:")
    for i, size in enumerate(top_chains, 1):
        print(f"     {i}. {size} atoms")

print("\n" + "=" * 60)
print("✅ Quick analysis complete!")
print("\n💡 For full analysis, run: python analyze_trajectory.py")


