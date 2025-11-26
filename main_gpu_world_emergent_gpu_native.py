"""GPU-native emergent substrate runner."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np
from lammps import lammps
from scipy.spatial import cKDTree

KB = 1.380649e-23
AMU = 1.66053906660e-27
DEFAULT_MASS_AMU = 12.0


@dataclass(frozen=True)
class EmergentConfig:
    n_particles: int = 500_000
    n_steps: int = 500_000
    timestep_fs: float = 2.0
    box_width_nm: float = 200.0
    box_height_nm: float = 400.0
    gravity_m_s2: float = 9.80665
    reaction_radius_nm: float = 0.25
    activation_energy_ev: float = 0.6
    arrhenius_prefactor_hz: float = 1.0e13
    max_bonds_per_atom: int = 8


def _lj_units_time_step(fs: float) -> float:
    seconds = fs * 1.0e-15
    epsilon = 1.654e-21
    mass = 6.6335209e-26
    sigma = 3.4e-10
    tau = sigma * math.sqrt(mass / epsilon)
    return seconds / tau


def detect_reactions_gpu_native(
    lmp: lammps,
    reaction_radius: float,
    activation_energy_ea: float,
    arrhenius_prefactor_a: float,
    temperature: float,
    max_bonds_per_atom: int,
) -> int:
    natoms = int(lmp.get_natoms())
    bond_counts = np.zeros(natoms, dtype=int)
    existing_bonds: Set[Tuple[int, int]] = set()

    try:
        nspecial = lmp.extract_atom("nspecial", 1)
        special = lmp.extract_atom("special", 1)
        if nspecial is not None and special is not None:
            for i in range(natoms):
                n_neigh = nspecial[i][0] if hasattr(nspecial[i], "__len__") else 0
                bond_counts[i] = n_neigh
                for j in range(n_neigh):
                    neighbor_id = special[i][j] - 1
                    if neighbor_id >= 0:
                        existing_bonds.add(tuple(sorted((i, neighbor_id))))
    except Exception:
        pass

    try:
        lmp.command("compute pair_dist all pair/local dist")
        pair_data = lmp.extract_compute("pair_dist", 1, 2)
        if pair_data:
            pairs = [
                (int(i), int(j), dist)
                for i, j, dist, *_ in pair_data
                if dist <= reaction_radius and i < j
            ]
            lmp.command("uncompute pair_dist")
            if not pairs:
                return 0
            pairs_np = np.array(pairs)
            i_indices = pairs_np[:, 0].astype(int)
            j_indices = pairs_np[:, 1].astype(int)
            distances = pairs_np[:, 2]
        else:
            raise RuntimeError("pair/local returned no data")
    except Exception:
        x = lmp.gather_atoms("x", 1, 3)
        positions = np.array([x[i : i + 3] for i in range(0, natoms * 3, 3)])[:, :2]
        pair_array = cKDTree(positions).query_pairs(reaction_radius, output_type="ndarray")
        if len(pair_array) == 0:
            return 0
        i_indices = pair_array[:, 0]
        j_indices = pair_array[:, 1]
        distances = np.linalg.norm(positions[j_indices] - positions[i_indices], axis=1)

    valid_mask = (bond_counts[i_indices] < max_bonds_per_atom) & (
        bond_counts[j_indices] < max_bonds_per_atom
    )
    i_indices = i_indices[valid_mask]
    j_indices = j_indices[valid_mask]
    distances = distances[valid_mask]
    if len(i_indices) == 0:
        return 0

    not_bonded_mask = np.array(
        [tuple(sorted((i, j))) not in existing_bonds for i, j in zip(i_indices, j_indices, strict=True)]
    )
    i_indices = i_indices[not_bonded_mask]
    j_indices = j_indices[not_bonded_mask]
    distances = distances[not_bonded_mask]
    if len(i_indices) == 0:
        return 0

    velocities = np.array([lmp.gather_atoms("v", 1, 3)[k : k + 3] for k in range(0, natoms * 3, 3)])[:, :2]
    relative_speeds = np.linalg.norm(velocities[j_indices] - velocities[i_indices], axis=1)
    collision_energies = 0.5 * 0.5 * (relative_speeds ** 2)
    effective_T = np.maximum(temperature, collision_energies / temperature)
    k = arrhenius_prefactor_a * np.exp(-activation_energy_ea / (effective_T * temperature))
    prob = k * 0.005 * (1.0 - distances / reaction_radius)

    reaction_mask = np.random.random(len(i_indices)) < prob
    reacting_pairs = list(zip(i_indices[reaction_mask], j_indices[reaction_mask], strict=True))
    bonds_created = 0
    for i_atom, j_atom in reacting_pairs:
        try:
            lmp.command(f"create_bonds single/bond 1 {i_atom + 1} {j_atom + 1}")
            bonds_created += 1
        except Exception:
            continue
    return bonds_created


def run_lammps_simulation_gpu_native(cfg: EmergentConfig | None = None) -> None:
    cfg = cfg or EmergentConfig()
    print("🚀 Building GPU-native emergent substrate")
    print(f"   Particles............... {cfg.n_particles:,}")
    print(f"   Steps.................... {cfg.n_steps:,}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    lmp = lammps(cmdargs=["-k", "on", "g", "1", "-sf", "kk"])
    lmp.command("units lj")
    lmp.command("dimension 3")
    lmp.command("atom_style molecular")
    lmp.command("boundary p s p")
    lmp.command("neighbor 1.0 bin")
    lmp.command("neigh_modify delay 0 every 2 check yes")

    sigma_nm = 0.34
    width = cfg.box_width_nm / sigma_nm
    height = cfg.box_height_nm / sigma_nm
    lmp.command(f"region sim block 0 {width} 0 {height} 0 {width}")
    lmp.command(
        "create_box 2 sim bond/types 1 extra/bond/per/atom {maxb} extra/special/per/atom 2".format(
            maxb=cfg.max_bonds_per_atom
        )
    )
    mass_scale = DEFAULT_MASS_AMU * AMU / 6.6335209e-26
    lmp.command(f"mass 1 {mass_scale}")
    lmp.command(f"mass 2 {mass_scale}")

    half = cfg.n_particles // 2
    lmp.command("suffix off")
    lmp.command(f"create_atoms 1 random {half} 1337 sim")
    lmp.command(f"create_atoms 2 random {cfg.n_particles - half} 7331 sim")
    lmp.command("delete_atoms overlap 0.9 all all")
    lmp.command("reset_atoms id")
    print("🧘 Minimising energy (host) ...")
    lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")
    print("   Minimisation complete.")
    lmp.command("suffix kk")

    lmp.command("pair_style lj/cut/kk 2.5")
    lmp.command("pair_coeff * * 1.0 1.0")
    lmp.command("bond_style harmonic")
    lmp.command("bond_coeff 1 150.0 1.2")
    lmp.command("special_bonds lj/coul 0.0 1.0 1.0")

    timestep = _lj_units_time_step(cfg.timestep_fs)
    lmp.command(f"timestep {timestep}")
    lmp.command("velocity all create 1.0 98765 dist gaussian")
    gravity = cfg.gravity_m_s2 / 9.80665
    lmp.command("fix integrator all nve")
    lmp.command(f"fix gravity all gravity {gravity} vector 0 -1 0")
    lmp.command("fix thermostat all langevin 1.0 1.0 100.0 424242")

    lmp.command("log gpu_native.log")
    lmp.command("dump traj all custom 20000 gpu_native.xyz id type x y z")

    start = time.time()
    lmp.command(f"run {cfg.n_steps}")
    elapsed = time.time() - start
    print(f"✅ Emergent GPU run complete  |  Wall time: {elapsed/3600:.2f} h")
    print("   Log       : gpu_native.log")
    print("   Trajectory: gpu_native.xyz")


if __name__ == "__main__":
    run_lammps_simulation_gpu_native()
