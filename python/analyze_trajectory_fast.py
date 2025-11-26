"""
FAST trajectory analysis - uses spatial hashing and samples frames.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def parse_xyz_trajectory(traj_path: Path) -> list:
    """Parse trajectory - fast version."""
    frames = []
    current_frame = None
    reading_atoms = False
    atom_count = 0
    atoms_read = 0
    
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("ITEM: TIMESTEP"):
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {'timestep': None, 'atoms': []}
                reading_atoms = False
                atoms_read = 0
            
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
    
    return frames

def find_bonds_fast(frame, cutoff=1.5):
    """Fast bond finding with spatial hashing."""
    bonds = defaultdict(set)
    atoms = frame['atoms']
    
    if len(atoms) == 0:
        return {}
    
    # Spatial hashing
    cell_size = cutoff * 2.0
    all_pos = np.array([a['pos'] for a in atoms])
    min_pos = all_pos.min(axis=0)
    
    grid = defaultdict(list)
    for atom in atoms:
        pos = atom['pos']
        cell_x = int((pos[0] - min_pos[0]) / cell_size)
        cell_y = int((pos[1] - min_pos[1]) / cell_size)
        grid[(cell_x, cell_y)].append(atom)
    
    checked = set()
    for (cx, cy), cell_atoms in grid.items():
        # Same cell
        for i, a1 in enumerate(cell_atoms):
            for a2 in cell_atoms[i+1:]:
                pair = tuple(sorted([a1['id'], a2['id']]))
                if pair not in checked:
                    checked.add(pair)
                    dist = np.linalg.norm(a1['pos'] - a2['pos'])
                    if dist < cutoff:
                        bonds[a1['id']].add(a2['id'])
                        bonds[a2['id']].add(a1['id'])
        
        # Adjacent cells
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor = (cx + dx, cy + dy)
            if neighbor in grid:
                for a1 in cell_atoms:
                    for a2 in grid[neighbor]:
                        pair = tuple(sorted([a1['id'], a2['id']]))
                        if pair not in checked:
                            checked.add(pair)
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

# Main analysis
print("=" * 60)
print("🔬 FAST TRAJECTORY ANALYSIS")
print("=" * 60)

traj_path = Path("gpu_emergent_trajectory.xyz")
print(f"\n📁 Analyzing: {traj_path}")

print("\n📊 Parsing trajectory...")
frames = parse_xyz_trajectory(traj_path)
print(f"   Found {len(frames)} frames")

# Sample frames: first, middle, last, and a few in between
if len(frames) <= 3:
    sample_indices = list(range(len(frames)))
else:
    sample_indices = [0]
    if len(frames) > 1:
        sample_indices.append(len(frames) // 4)
        sample_indices.append(len(frames) // 2)
        sample_indices.append(3 * len(frames) // 4)
    sample_indices.append(len(frames) - 1)
    sample_indices = sorted(set(sample_indices))

print(f"\n📈 Analyzing {len(sample_indices)} key frames...")

results = []
for idx in sample_indices:
    frame = frames[idx]
    print(f"   Frame {idx+1}/{len(frames)} (step {frame['timestep']})...")
    
    bonds = find_bonds_fast(frame, cutoff=1.5)
    molecules = find_molecules(bonds)
    
    sizes = [len(m) for m in molecules]
    chain_sizes = [s for s in sizes if s > 1]
    
    results.append({
        'step': frame['timestep'],
        'n_molecules': len(molecules),
        'n_bonds': sum(len(v) for v in bonds.values()) // 2,
        'avg_size': np.mean(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'chain_sizes': chain_sizes
    })

# Summary
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)

first = results[0]
last = results[-1]

print(f"\n🔗 Bond Formation:")
print(f"   First frame: {first['n_bonds']:,} bonds, {first['n_molecules']:,} molecules")
print(f"   Last frame: {last['n_bonds']:,} bonds, {last['n_molecules']:,} molecules")
print(f"   Bond growth: {last['n_bonds'] - first['n_bonds']:+,} ({((last['n_bonds']/first['n_bonds']-1)*100):+.1f}%)" if first['n_bonds'] > 0 else "   Bond growth: N/A")

print(f"\n📏 Chain Lengths:")
print(f"   First frame: avg {first['avg_size']:.1f}, max {first['max_size']}")
print(f"   Last frame: avg {last['avg_size']:.1f}, max {last['max_size']}")

if last['chain_sizes']:
    top_chains = sorted(last['chain_sizes'], reverse=True)[:10]
    print(f"\n   Top 10 chain lengths (last frame):")
    for i, size in enumerate(top_chains, 1):
        print(f"     {i}. {size} atoms")

print(f"\n📈 Molecule Growth:")
growth = (last['n_molecules'] - first['n_molecules']) / first['n_molecules'] * 100 if first['n_molecules'] > 0 else 0
print(f"   Molecule count change: {last['n_molecules'] - first['n_molecules']:+,} ({growth:+.1f}%)")

# Autocatalytic assessment
print(f"\n🧬 Self-Replication Assessment:")
confidence = 0.0
if last['max_size'] > 20:
    print(f"   ✅ Long chains formed (max {last['max_size']} atoms)")
    confidence += 0.3
if last['max_size'] > 10:
    print(f"   ✅ Complex molecules (max {last['max_size']} atoms)")
    confidence += 0.2
if growth > 10:
    print(f"   ✅ Molecule growth detected ({growth:.1f}%)")
    confidence += 0.3
if growth > 50:
    print(f"   ✅ EXPONENTIAL GROWTH!")
    confidence += 0.2

if confidence > 0.7:
    print(f"\n   🎯 HIGH confidence for self-replication ({confidence*100:.0f}%)")
elif confidence > 0.4:
    print(f"\n   🎯 MEDIUM confidence ({confidence*100:.0f}%)")
else:
    print(f"\n   ⚠️  Need more data ({confidence*100:.0f}%)")

print("\n" + "=" * 60)
print("✅ Analysis complete!")


