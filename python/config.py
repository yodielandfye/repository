"""
This file holds the "Physics DNA" of our universe.
We will evolve these numbers.
"""

import numpy as np

class Config:
    # Matter Constants
    CHEMISTRY = {
        "BOND_DISTANCE": 3.0,  # Further reduced - even tighter threshold
        "DAMPING": 0.92,  # Further reduced - much more energy retained
        # Reaction parameters (Arrhenius equation)
        "ACTIVATION_ENERGY": 250.0,  # Much higher - bonds much harder to form
        "BOLTZMANN_TEMPERATURE": 1.0,  # Temperature factor (kT in energy units)
        "REACTION_PROBABILITY_SCALE": 0.1,  # Doubled - even more reactions
        "MAX_MOLECULE_SIZE": 50,  # Force break bonds if molecule gets too large
    }
    
    # Physics Constants
    PHYSICS = {
        "GRAVITY": np.array([0.0, -9.8]),  # A constant downward pull
    }
    
    # World Constants
    WORLD = {
        "GRID_CELL_SIZE": 10,  # The size of our lag-stopping "buckets"
        "WIDTH": 500,          # Size of the 2D "box"
        "HEIGHT": 1000,        # Size of the 2D "box"
    }
    
    # Sun Constants (Radiative Energy Source)
    # Real physics: Inverse Square Law (energy = constant / distance²)
    # Solar constant: ~1361 W/m² at Earth's distance (scaled for simulation)
    SUN = {
        "POSITION": np.array([250.0, 1000.0]),  # Position of energy source (top center)
        "STRENGTH": 2000.0,                      # Radiative power (scaled for simulation units)
    }
    
    # Simulation Constants
    SEED = {
        "PLANET_PARTICLES": 1000,     # Reduced for fast testing (was 10,000)
        "ATMOSPHERE_PARTICLES": 100,  # Reduced for fast testing (was 1,000)
    }







