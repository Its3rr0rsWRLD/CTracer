# Getting Started

## 1. Build

Run `build.bat` from the repo root. It handles everything — finds MSVC, CUDA, and Ninja automatically.

```bat
build.bat
```

Output: `build\ctracer.exe`

**If it fails:**

| Error | Fix |
|---|---|
| `nvcc not found` | Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to PATH |
| `ninja not found` | Install Ninja, or make sure VS build tools are installed |
| `vcvarsall.bat not found` | Install the **Desktop development with C++** workload in Visual Studio Installer |
| CMake configure fails | Delete the `build/` folder and try again |

---

## 2. Run

```bat
build\ctracer.exe
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--port 8000` | 8000 | HTTP port Studio connects to |
| `--width 256` | 256 | Render width in pixels |
| `--height 144` | 144 | Render height in pixels |
| `--spp 1` | 1 | Samples per pixel per frame |
| `--bounces 4` | 4 | Max path bounces |

The render window opens immediately. You don't need Studio connected — you can adjust settings from the panel while it's running.

---

## 3. Studio setup

### Add the scripts

Copy these files from `lua/` into your place:

| File | Where it goes | Type |
|---|---|---|
| `GpuRendererServer.lua` | `ServerScriptService` | Script |
| `GpuRendererClient.lua` | `StarterPlayer > StarterPlayerScripts` | LocalScript |

The easiest way is to open each file in a text editor, create a new Script/LocalScript in Studio, and paste.

### Enable HTTP

**Home → Game Settings → Security → Allow HTTP Requests → On**

CTracer only listens on `127.0.0.1` so nothing is exposed externally.

---

## 4. First render

1. Start `ctracer.exe`
2. Open your place in Studio and hit **Play**
3. Press **P** in-game to start the render loop
4. The desktop window should start updating with your scene

If nothing happens, check the Studio Output window for connection errors. The most common cause is ctracer not running before you press P.
