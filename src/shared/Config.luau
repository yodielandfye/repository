--!native
--!strict
-- This file holds the "Physics DNA" of our universe.
-- We will evolve these numbers.

local Config = {
	
	-- Matter Constants
	CHEMISTRY = {
		BOND_DISTANCE = 5,  -- How close particles must be to stick
	},
	
	-- Physics Constants
	PHYSICS = {
		GRAVITY = Vector2.new(0, -9.8), -- A constant downward pull
	},
	
	-- World Constants
	WORLD = {
		GRID_CELL_SIZE = 10,  -- The size of our lag-stopping "buckets"
		WIDTH = 500,          -- Size of the 2D "box"
		HEIGHT = 1000,        -- Size of the 2D "box"
	},
	
	-- Sun Constants (Radiative Energy Source)
	-- Real physics: Inverse Square Law (energy = constant / distance²)
	-- Solar constant: ~1361 W/m² at Earth's distance (scaled for simulation)
	SUN = {
		POSITION = Vector2.new(250, 1000),  -- Position of energy source (top center)
		STRENGTH = 2000.0,                  -- Radiative power (scaled for simulation units)
	},
	
	-- Simulation Constants
	SEED = {
		PLANET_PARTICLES = 10000,    -- DOWN from 90,000 for testing
		ATMOSPHERE_PARTICLES = 1000, -- DOWN from 10,000 for testing
	},
}

return Config

