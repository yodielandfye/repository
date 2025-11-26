--!native
--!strict
local Config = require(script.Parent.Config)

-- This is the "atom" of our universe.
-- It's a pure data structure.

export type Particle = {
	-- Physics State
	position: Vector2,
	velocity: Vector2,
	mass: number,     -- Mass (for gravity and kinetic energy)
	
	-- Chemical State
	particleType: string, -- "CARBON", "HYDROGEN", "OXYGEN"
	
	-- Lennard-Jones Parameters (Real atomic properties)
	epsilon: number,  -- Bond strength (well depth)
	sigma: number,    -- Atomic radius (equilibrium distance)
	
	-- Bond Tracking (just metadata - bonds form from physics)
	bonds: {Particle}, -- List of particles this particle is bonded to
}

-- Real atomic properties (Lennard-Jones parameters + mass)
-- Based on measured values from molecular dynamics simulations
local ATOMIC_PROPERTIES = {
	CARBON = {
		epsilon = 0.105,  -- kcal/mol (bond strength)
		sigma = 3.40,     -- Angstroms (atomic radius)
		mass = 12.0,       -- Atomic mass units (Carbon-12)
	},
	HYDROGEN = {
		epsilon = 0.022,  -- Weakest bonds
		sigma = 2.65,     -- Smallest atom
		mass = 1.0,        -- Atomic mass units (Hydrogen-1)
	},
	OXYGEN = {
		epsilon = 0.120,  -- Strong bonds
		sigma = 3.12,     -- Medium size
		mass = 16.0,       -- Atomic mass units (Oxygen-16)
	},
}

-- Factory for making new particles
local function create(pType: string, pos: Vector2): Particle
	local props = ATOMIC_PROPERTIES[pType] or ATOMIC_PROPERTIES.CARBON
	
	local self: Particle = {
		position = pos,
		velocity = Vector2.new(0, 0), -- Start still
		mass = props.mass,
		
		particleType = pType,
		epsilon = props.epsilon,
		sigma = props.sigma,
		
		bonds = {}, -- Start with no bonds
	}
	
	return self
end

-- Helper function to check if particle A is bonded to particle B
local function isBonded(particle: Particle, other: Particle): boolean
	for _, bonded in particle.bonds do
		if bonded == other then
			return true
		end
	end
	return false
end

-- Helper function to add a bond (both directions)
local function addBond(particleA: Particle, particleB: Particle)
	if not isBonded(particleA, particleB) then
		table.insert(particleA.bonds, particleB)
		table.insert(particleB.bonds, particleA)
	end
end

-- Helper function to remove a bond (both directions)
local function removeBond(particleA: Particle, particleB: Particle)
	-- Remove B from A's bonds
	for i = #particleA.bonds, 1, -1 do
		if particleA.bonds[i] == particleB then
			table.remove(particleA.bonds, i)
			break
		end
	end
	-- Remove A from B's bonds
	for i = #particleB.bonds, 1, -1 do
		if particleB.bonds[i] == particleA then
			table.remove(particleB.bonds, i)
			break
		end
	end
end

return {
	create = create,
	isBonded = isBonded,
	addBond = addBond,
	removeBond = removeBond,
}

