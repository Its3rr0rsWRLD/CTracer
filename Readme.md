# CTracer

GPU path tracer for Roblox. Runs on your PC as a separate process, gets scene data from Studio over HTTP, and renders it in real time using CUDA.

Some of the CUDA code I found on stackoverflow since I've never done GPU stuff in C before, so don't judge too hard.

---

## Requirements

- Windows 10/11
- NVIDIA GPU (Maxwell or newer — anything from roughly 2014+)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- [CMake 3.18+](https://cmake.org/download/)
- Visual Studio 2022 with the **Desktop development with C++** workload (needed for MSVC and Ninja)

---

## Building

Just run `build.bat` from the repo root. It finds everything automatically (MSVC, CUDA, Ninja) and outputs `build\ctracer.exe`.

```bat
build.bat
```

If it fails, the most common issues are:
- `nvcc not found` — add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your PATH
- `ninja not found` — install Ninja or make sure VS build tools are installed
- `vcvarsall.bat not found` — install the C++ workload in Visual Studio Installer

Or just grab a prebuilt from Releases and skip all of this.

---

## Running

```bat
build\ctracer.exe
```

Flags (all optional):

```
--port 8000       HTTP port Studio connects to (default: 8000)
--width 256       render width
--height 144      render height
--spp 1           samples per pixel per frame
--bounces 4       max ray bounces
```

The render window opens automatically. You can adjust SPP, bounces, exposure, bloom, etc. from the panel on the right side without restarting.

---

## Studio Setup

### 1. Add the scripts

Copy the scripts from `lua/` into your place:

- `lua/GpuRendererServer.lua` → `ServerScriptService` (Script)
- `lua/GpuRendererClient.lua` → `StarterPlayer/StarterPlayerScripts` (LocalScript)

### 2. Enable HTTP requests

In Studio: **Home → Game Settings → Security → Allow HTTP Requests** → on.

(CTracer only listens on `127.0.0.1` so this doesn't expose anything externally.)

### 3. Start rendering

1. Open your place in Studio and hit **Play**
2. Start `ctracer.exe` on your PC
3. Press **P** in-game to toggle the render loop
4. The render window on your desktop should start showing your scene

If Studio can't connect it'll print a warning in the Output window. Make sure ctracer is running before you press P.

---

## How it works

The Studio scripts collect every visible `BasePart` in your workspace — position, size, rotation, color, and material — and sends it to ctracer over HTTP whenever the scene changes. The client script sends the current camera position every frame so the render stays in sync with where you're looking.

CTracer uploads all of that to the GPU and runs a path trace. Every frame it traces 1 sample per pixel and blends it into a history buffer, so the image gets cleaner the longer you stay still (temporal accumulation). Moving the camera resets the accumulation and starts fresh.

At 256×144 with 1 SPP it's basically real time on most NVIDIA GPUs. Cranking up resolution or SPP will slow it down but produce a cleaner image.

---

## Material properties

By default, material properties (roughness, metallic, etc.) are inferred from the Roblox `Material` enum. You can override them per-part using **Attributes**:

| Attribute | Type | Description |
|---|---|---|
| `RT_Roughness` | number | 0 = mirror, 1 = fully diffuse |
| `RT_Metallic` | number | 0 = dielectric, 1 = metal |
| `RT_Reflectance` | number | extra reflectance boost |
| `RT_EmissionStrength` | number | emission brightness multiplier |
| `RT_EmissionColor` | Color3 | emission color (defaults to part color) |
| `RT_IOR` | number | index of refraction for glass (default 1.5) |
| `RT_Transparency` | number | overrides part transparency |
| `RT_Albedo` | Color3 | overrides part color |

---

## HTTP API (if you need it)

ctracer exposes a simple JSON API on `127.0.0.1:8000`:

**`POST /scene`** — upload full static scene (objects + lights + world settings)

**`POST /render`** — send camera + trigger a render frame

**`GET /health`** — returns `{"status":"ok"}` if running

**`POST /reset`** — reset temporal accumulation

Request body shape for `/scene` and `/render`:

```json
{
  "objects": [
    {
      "position": [x, y, z],
      "size": [x, y, z],
      "rotation": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
      "primitive": "box",
      "color": [r, g, b],
      "roughness": 0.5,
      "metallic": 0.0,
      "reflectance": 0.0,
      "transparency": 0.0,
      "ior": 1.5,
      "emission": [0, 0, 0]
    }
  ],
  "lights": [
    {
      "position": [x, y, z],
      "color": [r, g, b],
      "brightness": 1.0,
      "range": 60.0,
      "direction": [x, y, z],
      "angle": 1.57,
      "type": "point"
    }
  ],
  "world": {
    "skyTop": [0.4, 0.6, 1.0],
    "skyBottom": [0.85, 0.85, 0.85],
    "sunDirection": [0.5, 0.8, 0.3],
    "sunColor": [1.0, 0.95, 0.8],
    "sunIntensity": 1.0,
    "ambientColor": [0.02, 0.02, 0.03],
    "ambientIntensity": 0.4,
    "fogDensity": 0.0,
    "exposure": 0.6,
    "gamma": 2.2
  },
  "camera": {
    "position": [x, y, z],
    "forward": [x, y, z],
    "right": [x, y, z],
    "up": [x, y, z],
    "fov": 70.0
  },
  "seq": 0
}
```

`primitive` can be `"box"`, `"sphere"`, `"cylinder"`, or `"mesh"` (mesh requires `vertices` and `indices` arrays).

`light.type` can be `"point"`, `"spot"`, or `"surface"`. Spot and surface use `direction` and `angle` (radians).
