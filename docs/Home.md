# CTracer Wiki

CTracer is a real-time GPU path tracer for Roblox. It runs as a standalone app on your PC, connects to Roblox Studio over HTTP, and renders your scene using CUDA.

The render preview appears in a window on your desktop. Studio just sends scene data — it never displays the render itself.

---

## Pages

| Page | What's in it |
|---|---|
| [Getting Started](Getting-Started) | Build, run, Studio setup |
| [Studio Integration](Studio-Integration) | How the scripts work, where they go, controls |
| [Render Window](Render-Window) | What the UI does, settings panel |
| [Material Attributes](Material-Attributes) | Per-part RT_ attribute overrides |
| [HTTP API](HTTP-API) | Endpoint reference for custom integrations |

---

## Quick start

```bat
git clone https://github.com/its3rr0rswrld/CTracer
cd CTracer
build.bat
build\ctracer.exe
```

Then in Studio: add the scripts from `lua/`, enable HTTP requests, hit Play, press **P**.

---

## Requirements

- Windows 10/11
- NVIDIA GPU (Maxwell or newer)
- CUDA Toolkit 12.x
- CMake 3.18+
- Visual Studio 2022 with **Desktop development with C++**
