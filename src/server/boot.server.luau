--!native
--!strict
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")

local Shared = ReplicatedStorage:WaitForChild("Shared")
local Config = require(Shared:WaitForChild("Config"))
local Matter = require(Shared:WaitForChild("Matter"))
local PhysicsKernel = require(Shared:WaitForChild("PhysicsKernel"))
local SpatialHashGrid = require(Shared:WaitForChild("SpatialHashGrid"))

type Particle = Matter.Particle

-- 1. CREATE THE UNIVERSE
print("🌌 Creating Universe...")

local AllParticles = {}
local Grid = SpatialHashGrid.new(Config.WORLD.GRID_CELL_SIZE)

-- 2. SEED THE "PLANET" (The Primordial Soup)
print("🌍 Seeding Planet...")

for i = 1, Config.SEED.PLANET_PARTICLES do
	-- Spawn in a dense clump at the bottom
	local pos = Vector2.new(
		math.random(0, Config.WORLD.WIDTH),
		math.random(0, Config.WORLD.HEIGHT / 4) -- Only in bottom 25%
	)
	
	-- 70% Carbon, 30% Oxygen
	local pType = if math.random() < 0.7 then "CARBON" else "OXYGEN"
	local p = Matter.create(pType, pos)
	table.insert(AllParticles, p)
end

-- 3. SEED THE "ATMOSPHERE" (The Fuel)
print("💨 Seeding Atmosphere...")

for i = 1, Config.SEED.ATMOSPHERE_PARTICLES do
	-- Spawn in a thin gas at the top
	local pos = Vector2.new(
		math.random(0, Config.WORLD.WIDTH),
		math.random(Config.WORLD.HEIGHT / 2, Config.WORLD.HEIGHT) -- Only in top 50%
	)
	
	local p = Matter.create("HYDROGEN", pos)
	table.insert(AllParticles, p)
end

print("🔥 IGNITION. Starting Physics Loop...")

-- 4. MOLECULAR DETECTION FUNCTIONS
-- Find all molecules (connected components via bonds)
local function findMolecules(particles: {Particle}): {{Particle}}
	local molecules = {}
	local visited = {}
	
	-- DFS to find all particles in a molecule
	local function dfs(particle: Particle, molecule: {Particle})
		if visited[particle] then return end
		visited[particle] = true
		table.insert(molecule, particle)
		
		-- Visit all bonded particles
		for _, bonded in particle.bonds do
			dfs(bonded, molecule)
		end
	end
	
	-- Find all molecules
	for _, particle in particles do
		if not visited[particle] and #particle.bonds > 0 then
			-- Only count particles that have bonds (molecules, not isolated atoms)
			local molecule = {}
			dfs(particle, molecule)
			if #molecule > 1 then -- Only molecules with 2+ atoms
				table.insert(molecules, molecule)
			end
		end
	end
	
	return molecules
end

-- Identify molecular patterns
local function identifyMolecule(molecule: {Particle}): string
	local carbonCount = 0
	local hydrogenCount = 0
	local oxygenCount = 0
	
	for _, particle in molecule do
		if particle.particleType == "CARBON" then
			carbonCount = carbonCount + 1
		elseif particle.particleType == "HYDROGEN" then
			hydrogenCount = hydrogenCount + 1
		elseif particle.particleType == "OXYGEN" then
			oxygenCount = oxygenCount + 1
		end
	end
	
	-- Match to known patterns
	if carbonCount == 1 and hydrogenCount == 4 then
		return "CH4"
	elseif carbonCount == 1 and oxygenCount == 2 then
		return "CO2"
	elseif carbonCount == 1 and hydrogenCount == 2 and oxygenCount == 1 then
		return "H2CO"
	elseif hydrogenCount == 2 and oxygenCount == 1 then
		return "H2O"
	elseif hydrogenCount == 2 then
		return "H2"
	elseif oxygenCount == 2 then
		return "O2"
	elseif carbonCount == 1 and oxygenCount == 1 then
		return "CO"
	elseif carbonCount > 0 or hydrogenCount > 0 or oxygenCount > 0 then
		-- Unknown pattern, return formula
		local formula = ""
		if carbonCount > 0 then formula = formula .. "C" .. (carbonCount > 1 and tostring(carbonCount) or "") end
		if hydrogenCount > 0 then formula = formula .. "H" .. (hydrogenCount > 1 and tostring(hydrogenCount) or "") end
		if oxygenCount > 0 then formula = formula .. "O" .. (oxygenCount > 1 and tostring(oxygenCount) or "") end
		return formula
	else
		return "UNKNOWN"
	end
end

-- 5. THE MAIN LOOP (Heartbeat of God)
local frameCount = 0
RunService.Heartbeat:Connect(function(dt)
	frameCount = frameCount + 1
	
	-- DEBUG: Confirm loop is running
	if frameCount == 1 then
		print("✅ Physics loop started (frame 1)")
	end
	
	-- Cap dt to prevent physics explosions from lag spikes
	if dt > 0.1 then dt = 0.1 end
	
	-- This must be done every frame
	Grid:Clear()
	
	-- == INSERT PHASE ==
	-- Put all particles into their new buckets
	for _, p in AllParticles do
		Grid:Insert(p)
	end
	
	-- == PHYSICS PHASE (Now Singular) ==
	local processedCount = 0
	for _, p in AllParticles do
		-- 1. Find all neighbors
		local neighbors = Grid:Query(p.position)
		
		-- 2. Interact with neighbors and maintain bonds
		local neighborCount = 0
		for _, neighbor in neighbors do
			if p == neighbor then continue end -- Don't check against self
			if neighborCount >= 20 then break end -- Limit neighbors per particle
			
			-- Apply the Lennard-Jones force (pure physics)
			PhysicsKernel.ApplyAtomicForce(p, neighbor)
			
			-- Track bonds (just metadata - bonds form from physics above)
			local diff = Vector2.new(
				neighbor.position.X - p.position.X,
				neighbor.position.Y - p.position.Y
			)
			local distance = diff.Magnitude
			
			-- Calculate equilibrium distance (average sigma)
			local equilibriumDist = (p.sigma + neighbor.sigma) / 2
			local bondThreshold = equilibriumDist * 1.2 -- Allow 20% tolerance for movement
			
			-- Check relative velocity (stable if moving together, not apart)
			local relativeVel = Vector2.new(
				neighbor.velocity.X - p.velocity.X,
				neighbor.velocity.Y - p.velocity.Y
			)
			local relativeSpeed = relativeVel.Magnitude
			
			-- Bond if: within equilibrium distance AND stable (low relative velocity)
			if distance <= bondThreshold and relativeSpeed < 1.5 then
				if not Matter.isBonded(p, neighbor) then
					Matter.addBond(p, neighbor)
				end
			else
				-- Unbond if: too far apart OR moving apart too fast
				if distance > bondThreshold * 1.5 or relativeSpeed > 3.0 then
					if Matter.isBonded(p, neighbor) then
						Matter.removeBond(p, neighbor)
					end
				end
			end
			
			neighborCount = neighborCount + 1
		end
		
		-- 3. Apply universal laws
		PhysicsKernel.Move(p, dt)
		
		-- 5. Check boundaries (simple "box" collision)
		if p.position.X < 0 or p.position.X > Config.WORLD.WIDTH then
			p.velocity = Vector2.new(-p.velocity.X, p.velocity.Y) -- Bounce off walls
			p.position = Vector2.new(math.clamp(p.position.X, 0, Config.WORLD.WIDTH), p.position.Y)
		end
		if p.position.Y < 0 or p.position.Y > Config.WORLD.HEIGHT then
			p.velocity = Vector2.new(p.velocity.X, -p.velocity.Y) -- Bounce off floor/ceiling
			p.position = Vector2.new(p.position.X, math.clamp(p.position.Y, 0, Config.WORLD.HEIGHT))
		end
		
		-- Yield every 1000 particles to prevent timeout
		processedCount = processedCount + 1
		if processedCount % 1000 == 0 then
			task.wait()
		end
	end
	
	-- == LOGGING PHASE (Count Actual Bonds) ==
	-- Count bonds directly from the bonds lists (no inference, no sampling)
	if frameCount % 60 == 0 then -- ~1 second at 60fps
		local totalBonds = 0
		
		-- Count all bonds (each bond is stored in both particles, so divide by 2)
		for _, p in AllParticles do
			totalBonds = totalBonds + #p.bonds
		end
		
		-- Each bond is counted twice (once per particle), so divide by 2
		local actualBonds = totalBonds / 2
		
		print(string.format("⏱️  Time: %.1fs | Particles: %d | Bonds: %.0f", 
			frameCount / 60, #AllParticles, actualBonds))
	end
	
	-- == MOLECULAR DETECTION PHASE ==
	-- Detect molecular patterns every 5 seconds
	if frameCount % 300 == 0 then -- ~5 seconds at 60fps
		local molecules = findMolecules(AllParticles)
		
		-- Count molecules by type
		local moleculeCounts = {}
		for _, molecule in molecules do
			local pattern = identifyMolecule(molecule)
			moleculeCounts[pattern] = (moleculeCounts[pattern] or 0) + 1
		end
		
		-- Log results
		if next(moleculeCounts) then
			local logParts = {}
			for pattern, count in pairs(moleculeCounts) do
				table.insert(logParts, string.format("%s: %d", pattern, count))
			end
			table.sort(logParts)
			print(string.format("🧪 Molecules detected: %s", table.concat(logParts, ", ")))
		else
			print("🧪 No molecules detected (only isolated atoms)")
		end
	end
end)

