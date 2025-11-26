"""
CPU "wide sweep" controller with CHAIN POLYMERIZATION.

Launches one LAMMPS job per CPU core to sweep temperature (T) and activation
energy (Ea) while the GPU runs the deep simulation. Each worker uses:
- Three reactions: initiation, propagation, linking
- Edge atoms for chain extension
- Pure Arrhenius kinetics (no shortcuts)
- Enables chains, rings, and networks to form
"""

from __future__ import annotations

import multiprocessing as mp
import os
import textwrap
import time
from pathlib import Path
from typing import List, Tuple

from lammps import lammps

try:
    from config import Config
except ImportError:  # minimal fallback for standalone use
    class Config:  # type: ignore
        WORLD = {"WIDTH": 200, "HEIGHT": 500}


def _prepare_chain_polymerization_reactions(
    prefix: str,
    reaction_radius: float,
    arrhenius_prefactor_a: float,
    activation_energy_ea: float,
    seed_offset: int,
) -> dict:
    """
    Create THREE reactions for chain polymerization using edge atoms.
    Each parallel run gets unique file names via prefix.
    
    1. Initiation: 0+0 bonds → 1+1 bonds (diatomic formation)
    2. Propagation: 0+1 bonds → 1+2 bonds (chain extension)
    3. Linking: 1+1 bonds → 2+2 bonds (chain linking)
    """
    bond_length = max(reaction_radius * 0.5, 0.5)
    base = Path.cwd()
    
    # ===== REACTION 1: INITIATION (0+0 → 1+1) =====
    init_pre = textwrap.dedent(
        f"""
        {prefix}_init_pre
        2 atoms
        0 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0

        Types

        1 1
        2 2

        Charges

        1 0.0
        2 0.0
        """
    ).strip()
    
    init_post = textwrap.dedent(
        f"""
        {prefix}_init_post
        2 atoms
        1 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0

        Types

        1 1
        2 2

        Charges

        1 0.0
        2 0.0

        Bonds

        1 1 1 2
        """
    ).strip()
    
    init_map = textwrap.dedent(
        f"""
        # Initiation: Type 1 + Type 2 (both unbonded)
        2 equivalences
        0 edgeIDs
        0 deleteIDs
        0 createIDs
        0 chiralIDs
        1 constraints

        InitiatorIDs

        1
        2

        Equivalences

        1 1
        2 2

        Constraints

        arrhenius {arrhenius_prefactor_a:.6g} 0.0 {activation_energy_ea:.6g} {97531 + seed_offset}
        """
    ).strip()
    
    # ===== REACTION 2: PROPAGATION (0+1 → 1+2) =====
    prop_pre = textwrap.dedent(
        f"""
        {prefix}_prop_pre
        3 atoms
        1 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0
        3 -{bond_length:.6f} 0.0 0.0

        Types

        1 1
        2 2
        3 1

        Charges

        1 0.0
        2 0.0
        3 0.0

        Bonds

        1 1 1 3
        """
    ).strip()
    
    prop_post = textwrap.dedent(
        f"""
        {prefix}_prop_post
        3 atoms
        2 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0
        3 -{bond_length:.6f} 0.0 0.0

        Types

        1 1
        2 2
        3 1

        Charges

        1 0.0
        2 0.0
        3 0.0

        Bonds

        1 1 1 3
        2 1 1 2
        """
    ).strip()
    
    prop_map = textwrap.dedent(
        f"""
        # Propagation: Chain end (atom 1) + monomer (atom 2)
        # Atom 3 is edge atom (represents rest of chain)
        3 equivalences
        1 edgeIDs
        0 deleteIDs
        0 createIDs
        0 chiralIDs
        1 constraints

        InitiatorIDs

        1
        2

        EdgeIDs

        3

        Equivalences

        1 1
        2 2
        3 3

        Constraints

        arrhenius {arrhenius_prefactor_a:.6g} 0.0 {activation_energy_ea:.6g} {97532 + seed_offset}
        """
    ).strip()
    
    # ===== REACTION 3: LINKING (1+1 → 2+2) =====
    link_pre = textwrap.dedent(
        f"""
        {prefix}_link_pre
        4 atoms
        2 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0
        3 -{bond_length:.6f} 0.0 0.0
        4 {bond_length * 2:.6f} 0.0 0.0

        Types

        1 1
        2 2
        3 1
        4 2

        Charges

        1 0.0
        2 0.0
        3 0.0
        4 0.0

        Bonds

        1 1 1 3
        2 1 2 4
        """
    ).strip()
    
    link_post = textwrap.dedent(
        f"""
        {prefix}_link_post
        4 atoms
        3 bonds

        Coords

        1 0.0 0.0 0.0
        2 {bond_length:.6f} 0.0 0.0
        3 -{bond_length:.6f} 0.0 0.0
        4 {bond_length * 2:.6f} 0.0 0.0

        Types

        1 1
        2 2
        3 1
        4 2

        Charges

        1 0.0
        2 0.0
        3 0.0
        4 0.0

        Bonds

        1 1 1 3
        2 1 2 4
        3 1 1 2
        """
    ).strip()
    
    link_map = textwrap.dedent(
        f"""
        # Linking: Two chain ends (atoms 1 and 2)
        # Atoms 3 and 4 are edge atoms (rest of chains)
        4 equivalences
        2 edgeIDs
        0 deleteIDs
        0 createIDs
        0 chiralIDs
        1 constraints

        InitiatorIDs

        1
        2

        EdgeIDs

        3
        4

        Equivalences

        1 1
        2 2
        3 3
        4 4

        Constraints

        arrhenius {arrhenius_prefactor_a:.6g} 0.0 {activation_energy_ea:.6g} {97533 + seed_offset}
        """
    ).strip()
    
    # Write all files with unique names per run
    files = {
        "init_pre": (base / f"{prefix}_init_pre.mol", init_pre),
        "init_post": (base / f"{prefix}_init_post.mol", init_post),
        "init_map": (base / f"{prefix}_init_map.txt", init_map),
        "prop_pre": (base / f"{prefix}_prop_pre.mol", prop_pre),
        "prop_post": (base / f"{prefix}_prop_post.mol", prop_post),
        "prop_map": (base / f"{prefix}_prop_map.txt", prop_map),
        "link_pre": (base / f"{prefix}_link_pre.mol", link_pre),
        "link_post": (base / f"{prefix}_link_post.mol", link_post),
        "link_map": (base / f"{prefix}_link_map.txt", link_map),
    }
    
    paths = {}
    for key, (path, content) in files.items():
        path = path.resolve()
        path.write_text(content + "\n")
        paths[key] = path
    
    return paths


def _run_single_sim(run_id: int, temperature: float, activation_energy_ea: float) -> str:
    """
    Execute one CPU-resident LAMMPS job with the provided thermo/chemistry knobs.
    """

    log_path = Path(f"cpu_run_{run_id}.log").resolve()
    print(
        f"🚀 [CPU#{run_id:02d}] start :: T={temperature:.2f}, "
        f"Ea={activation_energy_ea:.2f}"
    )

    os.environ["OMP_NUM_THREADS"] = "1"  # keep each worker to one thread
    lmp = lammps(cmdargs=["-log", str(log_path)])

    width = float(Config.WORLD["WIDTH"])
    height = float(Config.WORLD["HEIGHT"])
    n_particles = 50_000

    lmp.command("units lj")
    lmp.command("dimension 2")
    lmp.command("atom_style molecular")
    lmp.command("boundary p s p")
    lmp.command("neighbor 0.5 bin")
    lmp.command("neigh_modify delay 0 every 1 check yes")
    lmp.command(f"region box block 0 {width} 1.1 {height} -0.5 0.5")
    lmp.command(
        "create_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4"
    )  # Increased for complex molecules (chains)

    lmp.command("mass 1 1.0")
    lmp.command("mass 2 1.0")
    lmp.command("pair_style lj/cut 2.5")
    lmp.command("pair_coeff * * 1.0 1.0")
    lmp.command("bond_style harmonic")
    lmp.command("bond_coeff 1 100.0 1.0")

    half = n_particles // 2
    lmp.command(f"create_atoms 1 random {half} {12345 + run_id} box")
    lmp.command(f"create_atoms 2 random {n_particles - half} {67890 + run_id} box")
    lmp.command("delete_atoms overlap 1.0 all all")
    lmp.command("reset_atoms id")

    lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")
    lmp.command("region lost block INF INF -10.0 0.0 INF INF")
    lmp.command("delete_atoms region lost")
    lmp.command("reset_atoms id")

    lmp.command("fix floor all wall/lj93 ylo 0 1.0 1.0 2.5")
    lmp.command("fix gravity all gravity 0.5 vector 0 -1 0")
    lmp.command("fix integrate all nve")
    lmp.command(
        f"fix thermostat all langevin {temperature} {temperature} 1.0 {24680 + run_id}"
    )
    lmp.command("fix enforce2d all enforce2d")

    # CHAIN POLYMERIZATION: Three reactions for chains, rings, networks
    prefix = f"cpu_reaction_{run_id}"
    reaction_files = _prepare_chain_polymerization_reactions(
        prefix=prefix,
        reaction_radius=1.5,  # Increased for better matching
        arrhenius_prefactor_a=1.0e10,
        activation_energy_ea=activation_energy_ea,
        seed_offset=run_id,
    )
    
    # Register all molecule templates
    lmp.command(f"molecule {prefix}_init_pre {reaction_files['init_pre']}")
    lmp.command(f"molecule {prefix}_init_post {reaction_files['init_post']}")
    lmp.command(f"molecule {prefix}_prop_pre {reaction_files['prop_pre']}")
    lmp.command(f"molecule {prefix}_prop_post {reaction_files['prop_post']}")
    lmp.command(f"molecule {prefix}_link_pre {reaction_files['link_pre']}")
    lmp.command(f"molecule {prefix}_link_post {reaction_files['link_post']}")
    
    # Create group for stabilization
    lmp.command("group nvt_grp id < 999999999")
    
    # ONE fix command with THREE react arguments (chain polymerization)
    fix_cmd = (
        f"fix react all bond/react stabilization yes nvt_grp 0.03 "
        f"react {prefix}_init all 1 0.0 1.5 {prefix}_init_pre {prefix}_init_post {reaction_files['init_map']} "
        f"react {prefix}_prop all 1 0.0 1.5 {prefix}_prop_pre {prefix}_prop_post {reaction_files['prop_map']} "
        f"react {prefix}_link all 1 0.0 1.5 {prefix}_link_pre {prefix}_link_post {reaction_files['link_map']}"
    )
    lmp.command(fix_cmd)

    lmp.command("timestep 0.005")
    lmp.command("velocity all create {0} {1} dist gaussian".format(temperature, 3927 + run_id))

    lmp.command("thermo_style custom step temp press pe etotal vol")
    lmp.command("thermo 2000")
    lmp.command("run 1000000")

    print(f"✅ [CPU#{run_id:02d}] complete -> {log_path.name}")
    lmp.close()
    return str(log_path)


def launch_cpu_sweep() -> None:
    """
    Launch 16 CPU simulations covering a 4×4 grid of temperatures vs Ea values.
    """

    # Optimized parameter ranges for chain polymerization
    temperatures = [1.0, 1.5, 2.0, 2.5]  # Higher T for better chain growth
    activation_energies = [1.0, 1.5, 2.0, 2.5]  # Lower Ea for faster reactions

    jobs: List[Tuple[int, float, float]] = []
    run_id = 0
    for temp in temperatures:
        for ea in activation_energies:
            jobs.append((run_id, temp, ea))
            run_id += 1

    n_workers = min(len(jobs), mp.cpu_count(), 16)
    print(
        f"=== 🚀 Launching CPU sweep ({len(jobs)} runs across "
        f"{n_workers} workers) ==="
    )

    start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(_run_single_sim, jobs)

    elapsed = time.time() - start
    print(f"\n=== ✅ CPU sweep complete in {elapsed / 3600:.2f} hours ===")
    print("Logs:")
    for path in results:
        print(f"  - {path}")


if __name__ == "__main__":
    launch_cpu_sweep()








