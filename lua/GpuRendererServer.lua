-- Put this file in ServerScriptService.

local HttpService = game:GetService("HttpService")
local Lighting = game:GetService("Lighting")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local GPU_SERVER_URL = "http://127.0.0.1:8000/render"
local GPU_SCENE_URL = "http://127.0.0.1:8000/scene"
local MIN_REQUEST_INTERVAL = 0.13
local lastRequestTime = 0

local remote = ReplicatedStorage:FindFirstChild("GpuPathtraceEvent")
if not remote or not remote:IsA("RemoteEvent") then
	remote = Instance.new("RemoteEvent")
	remote.Name = "GpuPathtraceEvent"
	remote.Parent = ReplicatedStorage
end
assert(remote, "RemoteEvent missing")


local B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
local B64_DECODE = {} :: { [number]: number }
for i = 1, 64 do
	B64_DECODE[string.byte(B64, i)] = i - 1
end
B64_DECODE[string.byte("=")] = 0

local function base64ToBuffer(b64: string): buffer
	b64 = b64:gsub("%s", "")
	local len = #b64
	local padding = 0
	if string.sub(b64, len, len) == "=" then
		padding = 1
		if string.sub(b64, len - 1, len - 1) == "=" then
			padding = 2
		end
	end
	local outLen = (len / 4) * 3 - padding
	local buf = buffer.create(outLen)
	local wi = 0
	for i = 1, len, 4 do
		local a = B64_DECODE[string.byte(b64, i)] or 0
		local b = B64_DECODE[string.byte(b64, i + 1)] or 0
		local c = B64_DECODE[string.byte(b64, i + 2)] or 0
		local d = B64_DECODE[string.byte(b64, i + 3)] or 0
		local triplet = bit32.bor(
			bit32.lshift(a, 18),
			bit32.lshift(b, 12),
			bit32.lshift(c, 6),
			d
		)
		if wi < outLen then
			buffer.writeu8(buf, wi, bit32.rshift(triplet, 16))
			wi += 1
		end
		if wi < outLen then
			buffer.writeu8(buf, wi, bit32.band(bit32.rshift(triplet, 8), 0xFF))
			wi += 1
		end
		if wi < outLen then
			buffer.writeu8(buf, wi, bit32.band(triplet, 0xFF))
			wi += 1
		end
	end
	return buf
end

local function vec3(v: Vector3): { number }
	return { v.X, v.Y, v.Z }
end

local function c3(col: Color3): { number }
	return { col.R, col.G, col.B }
end

local function mat3(cf: CFrame): { { number } }
	local r = cf.RightVector
	local u = cf.UpVector
	local l = cf.LookVector
	return {
		{ r.X, r.Y, r.Z },
		{ u.X, u.Y, u.Z },
		{ l.X, l.Y, l.Z },
	}
end

local function getNum(inst: Instance, name: string, default: number): number
	local v = inst:GetAttribute(name)
	if typeof(v) == "number" then return v end
	return default
end

local function getColor(inst: Instance, name: string, default: Color3): Color3
	local v = inst:GetAttribute(name)
	if typeof(v) == "Color3" then return v end
	return default
end

local function getVec(inst: Instance, name: string, default: Vector3): Vector3
	local v = inst:GetAttribute(name)
	if typeof(v) == "Vector3" then return v end
	return default
end

local ROUGHNESS: { [Enum.Material]: number } = {
	[Enum.Material.SmoothPlastic] = 0.15,
	[Enum.Material.Plastic] = 0.22,
	[Enum.Material.Metal] = 0.30,
	[Enum.Material.DiamondPlate] = 0.35,
	[Enum.Material.Foil] = 0.05,
	[Enum.Material.CorrodedMetal] = 0.75,
	[Enum.Material.Glass] = 0.02,
	[Enum.Material.Neon] = 0.05,
	[Enum.Material.Wood] = 0.60,
	[Enum.Material.WoodPlanks] = 0.65,
	[Enum.Material.Brick] = 0.80,
	[Enum.Material.Concrete] = 0.85,
	[Enum.Material.Slate] = 0.50,
	[Enum.Material.Granite] = 0.55,
	[Enum.Material.Marble] = 0.20,
	[Enum.Material.Cobblestone] = 0.80,
	[Enum.Material.Pebble] = 0.75,
	[Enum.Material.Sand] = 0.90,
	[Enum.Material.Sandstone] = 0.80,
	[Enum.Material.Grass] = 0.80,
	[Enum.Material.LeafyGrass] = 0.85,
	[Enum.Material.Ground] = 0.70,
	[Enum.Material.Ice] = 0.02,
	[Enum.Material.Snow] = 0.95,
	[Enum.Material.Fabric] = 0.90,
	[Enum.Material.Limestone] = 0.70,
	[Enum.Material.Basalt] = 0.65,
	[Enum.Material.Rock] = 0.75,
	[Enum.Material.CrackedLava] = 0.85,
	[Enum.Material.Salt] = 0.70,
	[Enum.Material.Mud] = 0.90,
	[Enum.Material.Pavement] = 0.75,
	[Enum.Material.Asphalt] = 0.80,
	[Enum.Material.Cardboard] = 0.85,
	[Enum.Material.Carpet] = 0.95,
	[Enum.Material.Plaster] = 0.80,
}

local METALLIC: { [Enum.Material]: number } = {
	[Enum.Material.Metal] = 0.9,
	[Enum.Material.DiamondPlate] = 0.85,
	[Enum.Material.Foil] = 1.0,
	[Enum.Material.CorrodedMetal] = 0.7,
}

local TRANSPARENCY: { [Enum.Material]: number } = {
	[Enum.Material.Glass] = 0.8,
	[Enum.Material.Ice] = 0.3,
}

local IOR: { [Enum.Material]: number } = {
	[Enum.Material.Glass] = 1.5,
	[Enum.Material.Ice] = 1.31,
}

local function defaultRoughness(mat: Enum.Material): number
	return ROUGHNESS[mat] or 0.50
end

local function defaultMetallic(mat: Enum.Material): number
	return METALLIC[mat] or 0.0
end

local function defaultTransparency(mat: Enum.Material, partTransp: number): number
	if partTransp > 0 then return partTransp end
	return TRANSPARENCY[mat] or 0.0
end

local function defaultIOR(mat: Enum.Material): number
	return IOR[mat] or 1.5
end

local function classifyPart(obj: BasePart): string?
	if obj:IsA("Terrain") then return nil end
	if not obj:IsA("BasePart") then return nil end
	if obj:IsA("Part") then
		if obj.Shape == Enum.PartType.Ball then return "sphere" end
		if obj.Shape == Enum.PartType.Cylinder then return "cylinder" end
		return "box"
	end
	return "box"
end

local function serializePart(obj: BasePart): { [string]: any }?
	local prim = classifyPart(obj)
	if not prim then return nil end

	local s = obj.Size
	if s.X <= 0 or s.Y <= 0 or s.Z <= 0 then return nil end

	local mat = obj.Material
	local baseCol = obj.Color

	local albedo = getColor(obj, "RT_Albedo", baseCol)
	local rough = getNum(obj, "RT_Roughness", defaultRoughness(mat))
	local metal = getNum(obj, "RT_Metallic", defaultMetallic(mat))
	local refl = getNum(obj, "RT_Reflectance", obj.Reflectance)
	local transp = getNum(obj, "RT_Transparency", defaultTransparency(mat, obj.Transparency))
	local ior = getNum(obj, "RT_IOR", defaultIOR(mat))

	local emCol = getColor(obj, "RT_EmissionColor", Color3.new(0, 0, 0))
	local emStr = getNum(obj, "RT_EmissionStrength", 0)

	if mat == Enum.Material.Neon and emStr == 0 then
		emStr = 12
		emCol = baseCol
	end

	if mat == Enum.Material.CrackedLava and emStr == 0 then
		emStr = 4
		emCol = Color3.new(1.0, 0.3, 0.05)
	end

	local sa = obj:FindFirstChildWhichIsA("SurfaceAppearance")
	if sa then
		local ok, saTint = pcall(function() return sa.Color end)
		if ok and typeof(saTint) == "Color3" then
			albedo = Color3.new(albedo.R * saTint.R, albedo.G * saTint.G, albedo.B * saTint.B)
		end
	end

	local textureId = ""
	for _, child in obj:GetChildren() do
		if (child:IsA("Decal") or child:IsA("Texture")) and child.Texture ~= "" then
			textureId = child.Texture
			break
		end
	end
	if textureId == "" and obj:IsA("MeshPart") then
		local ok2, tid = pcall(function() return obj.TextureID end)
		if ok2 and typeof(tid) == "string" and tid ~= "" then
			textureId = tid
		end
	end

	local entry: { [string]: any } = {
		primitive = prim,
		position = vec3(obj.Position),
		rotation = mat3(obj.CFrame),
		size = vec3(s),
		color = c3(albedo),
		roughness = math.clamp(rough, 0.02, 1.0),
		metallic = math.clamp(metal, 0.0, 1.0),
		reflectance = math.clamp(refl, 0.0, 1.0),
		transparency = math.clamp(transp, 0.0, 1.0),
		ior = math.clamp(ior, 1.0, 3.0),
		emission = {
			emCol.R * emStr,
			emCol.G * emStr,
			emCol.B * emStr,
		},
	}
	if textureId ~= "" then
		entry.textureId = textureId
	end
	return entry
end

local function collectScene(): { [string]: any }
	local objects = {}
	for _, obj in workspace:GetDescendants() do
		if obj:IsA("BasePart") and not obj:IsA("Terrain") then
			local entry = serializePart(obj)
			if entry then
				table.insert(objects, entry)
			end
		end
	end
	return objects
end

local function serializePlayerCharacter(player: Player): { [string]: any }?
	local char = player.Character
	if not char then return nil end
	local root = char:FindFirstChild("HumanoidRootPart")
	if not root or not root:IsA("BasePart") then return nil end
	local parts = {}
	for _, obj in char:GetChildren() do
		if obj:IsA("BasePart") then
			local entry = serializePart(obj)
			if entry then
				entry.isPlayerPart = true
				table.insert(parts, entry)
			end
		end
	end
	return parts
end

local function getFaceDirection(part: BasePart, face: Enum.NormalId): Vector3
	local cf = part.CFrame
	if face == Enum.NormalId.Front then return -cf.LookVector end
	if face == Enum.NormalId.Back then return cf.LookVector end
	if face == Enum.NormalId.Top then return cf.UpVector end
	if face == Enum.NormalId.Bottom then return -cf.UpVector end
	if face == Enum.NormalId.Right then return cf.RightVector end
	if face == Enum.NormalId.Left then return -cf.RightVector end
	return -cf.LookVector
end

local function collectLights(): { [string]: any }
	local lights = {}
	for _, obj in workspace:GetDescendants() do
		if not obj:IsA("Light") then continue end
		if not obj.Enabled then continue end

		local parent = obj.Parent
		if not parent or not parent:IsA("BasePart") then continue end

		local pos = parent.Position

		if obj:IsA("PointLight") then
			table.insert(lights, {
				type = "point",
				position = vec3(pos),
				color = c3(obj.Color),
				brightness = obj.Brightness,
				range = obj.Range,
			})
		elseif obj:IsA("SpotLight") then
			local dir = getFaceDirection(parent, obj.Face)
			table.insert(lights, {
				type = "spot",
				position = vec3(pos),
				direction = vec3(dir),
				color = c3(obj.Color),
				brightness = obj.Brightness,
				range = obj.Range,
				angle = obj.Angle,
			})
		elseif obj:IsA("SurfaceLight") then
			local dir = getFaceDirection(parent, obj.Face)
			table.insert(lights, {
				type = "surface",
				position = vec3(pos),
				direction = vec3(dir),
				color = c3(obj.Color),
				brightness = obj.Brightness,
				range = obj.Range,
				angle = obj.Angle,
			})
		end
	end
	return lights
end


local function collectWorld(): { [string]: any }
	local sunDir = getVec(Lighting, "RT_SunDir", -Lighting:GetSunDirection())
	local sunColor = getColor(Lighting, "RT_SunColor", Color3.new(1, 0.95, 0.8))
	local sunIntensity = getNum(Lighting, "RT_SunIntensity", math.clamp(Lighting.Brightness * 0.5, 0, 2))

	local bgTop = getColor(Lighting, "RT_BGTop", Color3.new(0.4, 0.6, 1.0))
	local bgBottom = getColor(Lighting, "RT_BGBottom", Color3.new(0.9, 0.9, 0.9))
	local envIntensity = getNum(Lighting, "RT_EnvIntensity", 1.0)

	local ambient = getColor(Lighting, "RT_Ambient", Lighting.OutdoorAmbient)
	local ambientIntensity = getNum(Lighting, "RT_AmbientIntensity", 0.3)

	local fogDensity = getNum(Lighting, "RT_FogDensity", 0.0)
	local fogColor = getColor(Lighting, "RT_FogColor", Lighting.FogColor)

	local exposure = getNum(Lighting, "RT_Exposure", 1.0)
	local gamma = getNum(Lighting, "RT_Gamma", 2.2)
	local bloomStrength = getNum(Lighting, "RT_BloomStrength", 0.0)
	local bloomThreshold = getNum(Lighting, "RT_BloomThreshold", 2.0)
	local sunAngularRadius = getNum(Lighting, "RT_SunAngularRadius", 0.02)

	local atmosphere = Lighting:FindFirstChildWhichIsA("Atmosphere")
	if atmosphere then
		local atmDensity = atmosphere.Density
		local atmHaze = atmosphere.Haze
		local atmColor = atmosphere.Color
		local atmDecay = atmosphere.Decay

		if fogDensity == 0 and (atmDensity > 0 or atmHaze > 0) then
			fogDensity = math.clamp(atmDensity * 0.01 + atmHaze * 0.002, 0, 0.01)
			fogColor = atmColor
		end

		local tintStrength = math.clamp(atmDensity * 0.5, 0, 0.5)
		bgTop = bgTop:Lerp(atmDecay, tintStrength)
		bgBottom = bgBottom:Lerp(atmColor, tintStrength)
	end

	local sky = Lighting:FindFirstChildWhichIsA("Sky")
	local skyboxIds = {}
	if sky then
		skyboxIds = {
			front = sky.SkyboxFt,
			back = sky.SkyboxBk,
			left = sky.SkyboxLf,
			right = sky.SkyboxRt,
			up = sky.SkyboxUp,
			down = sky.SkyboxDn,
		}

		if not Lighting:GetAttribute("RT_BGTop") then
			local ct = Lighting.ClockTime
			if ct >= 6 and ct <= 18 then
				bgTop = Color3.new(0.35, 0.55, 0.95)
				bgBottom = Color3.new(0.75, 0.82, 0.95)
			end
		end
	end

	local bloom = Lighting:FindFirstChildWhichIsA("BloomEffect")
	if bloom and bloom.Enabled then
		if bloomStrength == 0 then
			bloomStrength = bloom.Intensity * 0.5
			bloomThreshold = bloom.Threshold
		end
	end

	return {
		sunDirection = { sunDir.X, sunDir.Y, sunDir.Z },
		sunColor = c3(sunColor),
		sunIntensity = sunIntensity,
		sunAngularRadius = sunAngularRadius,
		skyTop = c3(bgTop),
		skyBottom = c3(bgBottom),
		envIntensity = envIntensity,
		ambientColor = c3(ambient),
		ambientIntensity = ambientIntensity,
		fogDensity = fogDensity,
		fogColor = c3(fogColor),
		exposure = exposure,
		gamma = gamma,
		bloomStrength = bloomStrength,
		bloomThreshold = bloomThreshold,
		skybox = skyboxIds,
		clockTime = Lighting.ClockTime,
		brightness = Lighting.Brightness,
	}
end

local cachedSceneJson: string? = nil
local SCENE_REFRESH_INTERVAL = 0.5
local lastSceneRefresh = 0
local sceneUploaded = false

local function refreshSceneIfNeeded()
	local now = os.clock()
	if now - lastSceneRefresh > SCENE_REFRESH_INTERVAL or cachedSceneJson == nil then
		local objects = collectScene()
		local lights = collectLights()
		local world = collectWorld()
		local newJson = HttpService:JSONEncode({
			objects = objects,
			lights = lights,
			world = world,
		})
		if newJson ~= cachedSceneJson then
			cachedSceneJson = newJson
			sceneUploaded = false
		end
		lastSceneRefresh = now
	end
end

local function uploadSceneIfDirty()
	if sceneUploaded or cachedSceneJson == nil then return end
	local ok, response = pcall(function()
		return HttpService:PostAsync(
			GPU_SCENE_URL,
			cachedSceneJson :: string,
			Enum.HttpContentType.ApplicationJson,
			false
		)
	end)
	if ok then
		sceneUploaded = true
	else
		warn("[RoTracer] Scene upload failed:", response)
	end
end

local function getCameraBody(cam: { [string]: any }, seq: number, playerParts: { [string]: any }?): string
	return HttpService:JSONEncode({ camera = cam, seq = seq, playerParts = playerParts })
end

local function getSceneBody(cam: { [string]: any }): string
	refreshSceneIfNeeded()
	local camJson = HttpService:JSONEncode({ camera = cam })
	return '{' .. string.sub(camJson, 2, -2) .. ',' .. string.sub(cachedSceneJson :: string, 2, -2) .. '}'
end


local function handleRequestFrame(player: Player, payload: { [string]: any })
	if typeof(payload) ~= "table" then
		warn("[RoTracer] Invalid payload from", player.Name)
		return
	end

	local cam = payload.camera
	if typeof(cam) ~= "table" then
		warn("[RoTracer] Missing camera from", player.Name)
		return
	end

	local seq = payload.seq or 0

	refreshSceneIfNeeded()
	uploadSceneIfDirty()

	local playerParts = serializePlayerCharacter(player)
	local bodyJson = getCameraBody(cam, seq, playerParts)

	local now = os.clock()
	local elapsed = now - lastRequestTime
	if elapsed < MIN_REQUEST_INTERVAL then
		task.wait(MIN_REQUEST_INTERVAL - elapsed)
	end
	lastRequestTime = os.clock()

	local ok, response = pcall(function()
		return HttpService:PostAsync(
			GPU_SERVER_URL,
			bodyJson,
			Enum.HttpContentType.ApplicationJson,
			false
		)
	end)

	if not ok then
		warn("[RoTracer] HTTP error:", response)
		remote:FireClient(player, { type = "Error", message = "Server unreachable" })
		return
	end

	local ok2, data = pcall(function()
		return HttpService:JSONDecode(response)
	end)

	if not ok2 or type(data) ~= "table" then
		warn("[RoTracer] Bad JSON from GPU server")
		return
	end

	if data.type ~= "frame" then
		warn("[RoTracer] Invalid frame response")
		return
	end

	remote:FireClient(player, {
		type = "Frame",
		sample = data.sample,
		seq = data.seq or 0,
	})
end

local GPU_RECORD_URL = "http://127.0.0.1:8000/record"
local GPU_PHOTO_URL = "http://127.0.0.1:8000/photo"

local function handleToggleRecord(player: Player)
	local ok, response = pcall(function()
		return HttpService:PostAsync(GPU_RECORD_URL, "{}", Enum.HttpContentType.ApplicationJson, false)
	end)
	if not ok then
		warn("[RoTracer] Record toggle failed:", response)
		remote:FireClient(player, { type = "Error", message = "Record toggle failed" })
		return
	end
	local ok2, data = pcall(function() return HttpService:JSONDecode(response) end)
	if ok2 and data then
		print("[RoTracer] Recording:", data.status or "unknown")
		remote:FireClient(player, { type = "RecordStatus", status = data.status or "unknown" })
	end
end

local function handleTakePhoto(player: Player, payload: { [string]: any })
	if typeof(payload) ~= "table" then return end
	local cam = payload.camera
	if typeof(cam) ~= "table" then
		remote:FireClient(player, { type = "PhotoError", message = "Missing camera" })
		return
	end

	local bodyJson = getSceneBody(cam)

	print("[RoTracer] Taking photo for", player.Name)
	local ok, response = pcall(function()
		return HttpService:PostAsync(GPU_PHOTO_URL, bodyJson, Enum.HttpContentType.ApplicationJson, false)
	end)

	if not ok then
		warn("[RoTracer] Photo error:", response)
		remote:FireClient(player, { type = "PhotoError", message = "Server unreachable" })
		return
	end

	local ok2, data = pcall(function() return HttpService:JSONDecode(response) end)
	if not ok2 or type(data) ~= "table" then
		remote:FireClient(player, { type = "PhotoError", message = "Bad response" })
		return
	end

	remote:FireClient(player, {
		type = "PhotoDone",
		samples = data.samples or 0,
		elapsed = data.elapsed or 0,
		path = data.path or "",
	})
end

-- Ignore Photo/Recording. That was something else I was working on.

remote.OnServerEvent:Connect(function(player: Player, messageType: string, payload: any)
	if messageType == "RequestFrame" then
		handleRequestFrame(player, payload)
	elseif messageType == "TakePhoto" then
		handleTakePhoto(player, payload)
	elseif messageType == "ToggleRecord" then
		handleToggleRecord(player)
	elseif messageType == "Ping" then
		remote:FireClient(player, { type = "Pong" })
	end
end)

print("[RoTracer] Server ready -- " .. #collectScene() .. " objects, " .. #collectLights() .. " lights")