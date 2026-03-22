-- Put this in a LocalScript under StarterPlayerScripts
local UserInputService = game:GetService("UserInputService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local camera = workspace.CurrentCamera
local gpuEvent: RemoteEvent = ReplicatedStorage:WaitForChild("GpuPathtraceEvent") :: RemoteEvent

local rendering = false
local waitingForFrame = false
local currentSample = 0
local frameSeq = 0
local lastDisplayedSeq = 0

local function getCameraPayload(): { [string]: any }
	local cf = camera.CFrame
	local pos = cf.Position
	local fwd = cf.LookVector
	local right = cf.RightVector
	local up = cf.UpVector
	local fov = camera.FieldOfView

	return {
		position = { pos.X, pos.Y, pos.Z },
		forward = { fwd.X, fwd.Y, fwd.Z },
		right = { right.X, right.Y, right.Z },
		up = { up.X, up.Y, up.Z },
		fov = fov,
	}
end

local function requestFrame()
	if waitingForFrame then return end
	waitingForFrame = true
	frameSeq += 1

	gpuEvent:FireServer("RequestFrame", {
		camera = getCameraPayload(),
		seq = frameSeq,
	})
end

gpuEvent.OnClientEvent:Connect(function(data: { [string]: any })
	if not data or typeof(data) ~= "table" then return end

	if data.type == "Pong" then return end

	if data.type == "Error" then
		warn("[CTracer] Error:", data.message)
		waitingForFrame = false
		return
	end

	if data.type == "RecordStatus" then
		print("[CTracer] Recording:", data.status or "unknown")
		waitingForFrame = false
		return
	end

	if data.type ~= "Frame" then
		waitingForFrame = false
		return
	end

	local seq = data.seq or 0
	if seq > 0 and seq < lastDisplayedSeq then
		waitingForFrame = false
		if rendering then
			task.defer(requestFrame)
		end
		return
	end
	lastDisplayedSeq = seq

	local sample = data.sample or 0
	currentSample = sample

	waitingForFrame = false

	if rendering then
		task.defer(requestFrame)
	end
end)

UserInputService.InputBegan:Connect(function(input: InputObject, gameProcessed: boolean)
	if gameProcessed then return end

	if input.KeyCode == Enum.KeyCode.P then
		rendering = not rendering
		if rendering then
			print("[CTracer] Rendering started")
			requestFrame()
		else
			print(string.format("[CTracer] Paused at sample %d", currentSample))
		end
	elseif input.KeyCode == Enum.KeyCode.R then
		gpuEvent:FireServer("ToggleRecord")
	end
end)

print("[CTracer] Client ready. Press P to toggle rendering.")
