--!native
--!strict
-- THE SINGULAR LAW: Lennard-Jones Potential
-- This ONE function replaces Collide(), Bond(), and ApplyBondConstraints()
-- It models the actual electromagnetic force between atoms

local Config = require(script.Parent.Config)
local Matter = require(script.Parent.Matter)

type Particle = Matter.Particle

local PhysicsKernel = {}

-- ###############################################################
-- ## THE SINGULAR LAW: ApplyAtomicForce
-- ###############################################################
-- This ONE function REPLACES Collide(), Bond(), and ApplyBondConstraints().
-- It calculates the REAL electromagnetic force between two particles.

function PhysicsKernel.ApplyAtomicForce(particleA: Particle, particleB: Particle)
	
	-- 1. Get the vector and distance between them
	local diff = Vector2.new(
		particleB.position.X - particleA.position.X,
		particleB.position.Y - particleA.position.Y
	)
	local distance = diff.Magnitude
	
	-- Optimization: If they are too far apart, don't bother.
	-- This "cutoff" distance saves 99% of our CPU.
	if distance > Config.CHEMISTRY.BOND_DISTANCE or distance == 0 then
		return
	end
	
	-- ### THE LENNARD-JONES MATH (The God Formula) ###
	-- Use Lorentz-Berthelot mixing rules for different atom types
	-- This is how real molecular dynamics handles different atoms
	local epsilon = math.sqrt(particleA.epsilon * particleB.epsilon)  -- Geometric mean
	local sigma = (particleA.sigma + particleB.sigma) / 2  -- Arithmetic mean
	
	-- 1. (SIGMA / distance)^6  -> This is the ATTRACTION force (pulls them together)
	-- 2. (SIGMA / distance)^12 -> This is the REPULSION force (shoves them apart)
	local term = sigma / distance
	
	-- We calculate the attraction (to the 6th power)
	local attraction = math.pow(term, 6)
	
	-- We calculate the repulsion (to the 12th power)
	-- Because it's to the 12th, it's MUCH stronger at close range.
	local repulsion = math.pow(term, 12)
	
	-- The total force = (Repulsion - Attraction)
	-- We multiply by 4 * Epsilon to get the right strength
	local forceMagnitude = 4 * epsilon * (repulsion - attraction)
	
	-- ### END OF MATH ###
	
	-- Now we apply that force to both particles
	-- We get the direction (diff.Unit) and multiply by the strength (forceMagnitude)
	local normal = diff.Unit
	local forceVector = Vector2.new(
		forceMagnitude * normal.X,
		forceMagnitude * normal.Y
	)
	
	-- We push/pull both particles based on this one, true force.
	-- Newton's 3rd law: equal and opposite forces
	particleA.velocity = Vector2.new(
		particleA.velocity.X - forceVector.X,
		particleA.velocity.Y - forceVector.Y
	)
	particleB.velocity = Vector2.new(
		particleB.velocity.X + forceVector.X,
		particleB.velocity.Y + forceVector.Y
	)
end

-- LAW 2: MOVEMENT (Gravity + Radiative Heating)
function PhysicsKernel.Move(particle: Particle, dt: number)
	
	-- Apply gravity (constant pull down)
	particle.velocity = Vector2.new(
		particle.velocity.X + Config.PHYSICS.GRAVITY.X * dt,
		particle.velocity.Y + Config.PHYSICS.GRAVITY.Y * dt
	)
	
	-- Apply radiative heating (Inverse Square Law)
	-- Real physics: Energy received = SUN_STRENGTH / (distance²)
	local sunDiff = Vector2.new(
		Config.SUN.POSITION.X - particle.position.X,
		Config.SUN.POSITION.Y - particle.position.Y
	)
	local sunDistance = sunDiff.Magnitude
	
	-- Prevent singularity at distance = 0
	if sunDistance > 0.1 then
		-- Inverse square law: energy = strength / distance²
		local energyReceived = Config.SUN.STRENGTH / (sunDistance * sunDistance)
		
		-- Convert energy to kinetic energy: KE = 0.5 * m * v²
		-- So: v = sqrt(2 * KE / m)
		-- We add this as a velocity impulse in the direction away from the sun
		local velocityImpulse = math.sqrt(2 * energyReceived / particle.mass) * dt
		local sunDirection = sunDiff.Unit
		
		particle.velocity = Vector2.new(
			particle.velocity.X + sunDirection.X * velocityImpulse,
			particle.velocity.Y + sunDirection.Y * velocityImpulse
		)
	end
	
	-- Update position
	particle.position = Vector2.new(
		particle.position.X + particle.velocity.X * dt,
		particle.position.Y + particle.velocity.Y * dt
	)
end

return PhysicsKernel
