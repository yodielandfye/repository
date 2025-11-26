--!native
--!strict
-- The "bucket" system. Solves lag before it happens.

export type SpatialHashGrid = {
	cellSize: number,
	buckets: {[string]: {any}},
	
	new: (cellSize: number) -> SpatialHashGrid,
	Clear: (self: SpatialHashGrid) -> (),
	Insert: (self: SpatialHashGrid, particle: any) -> (),
	Query: (self: SpatialHashGrid, position: Vector2) -> {any},
}

local function new(cellSize: number): SpatialHashGrid
	local self: SpatialHashGrid = {
		cellSize = cellSize,
		buckets = {},
		
		new = nil :: any,
		Clear = nil :: any,
		Insert = nil :: any,
		Query = nil :: any,
	}
	
	function self:Clear()
		self.buckets = {}
	end
	
	function self:Insert(particle: any)
		local cellX = math.floor(particle.position.X / self.cellSize)
		local cellY = math.floor(particle.position.Y / self.cellSize)
		local key = tostring(cellX) .. "," .. tostring(cellY)
		
		if not self.buckets[key] then
			self.buckets[key] = {}
		end
		
		table.insert(self.buckets[key], particle)
	end
	
	function self:Query(position: Vector2): {any}
		local cellX = math.floor(position.X / self.cellSize)
		local cellY = math.floor(position.Y / self.cellSize)
		local results = {}
		
		-- Check the cell and its 8 neighbors
		for dx = -1, 1 do
			for dy = -1, 1 do
				local key = tostring(cellX + dx) .. "," .. tostring(cellY + dy)
				local bucket = self.buckets[key]
				if bucket then
					for _, particle in bucket do
						table.insert(results, particle)
					end
				end
			end
		end
		
		return results
	end
	
	return self
end

return {
	new = new,
}







