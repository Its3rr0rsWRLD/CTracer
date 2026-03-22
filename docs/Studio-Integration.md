# Studio Integration

CTracer uses two regular game scripts — no plugin required. They communicate with the desktop renderer over HTTP using Roblox's `HttpService`.

---

## How it works

```
[Studio: LocalScript]  →  sends camera each frame  →  [Server Script]
                                                              ↓
                                                    checks if scene changed
                                                              ↓
                                                    POST /scene  (if changed)
                                                    POST /render (camera + players)
                                                              ↓
                                                    [ctracer.exe on your PC]
                                                              ↓
                                                    renders frame on GPU
                                                    returns sample count
```

The scene (all BaseParts, lights, world settings) is cached and only re-sent when it actually changes. Camera data goes every frame.

---

## Scripts

### GpuRendererServer (ServerScriptService)

Runs on the server. Responsible for:

- Collecting all `BasePart` instances from workspace and serializing their position, size, rotation, color, and material
- Collecting `PointLight`, `SpotLight`, and `SurfaceLight` instances
- Reading world/atmosphere settings from the `Lighting` service
- Caching the scene JSON and only uploading it to `/scene` when it changes
- Handling render requests from the client — sending camera + player character data to `/render`
- Creating the `GpuPathtraceEvent` RemoteEvent in `ReplicatedStorage`

### GpuRendererClient (StarterPlayerScripts)

Runs on each client. Responsible for:

- Reading `workspace.CurrentCamera` every frame
- Sending camera data to the server via the RemoteEvent
- Handling responses (frame confirmations, errors)
- Input handling

---

## Controls

| Key | Action |
|---|---|
| **P** | Toggle render loop on/off |
| **R** | Toggle recording (if implemented on server) |

Press **P** to start. Press it again to pause. The sample counter in the render window resets when you move the camera.

---

## What gets sent

**Objects:** Every `BasePart` in workspace that isn't `Terrain`. Includes position, size, rotation matrix, color, and material-derived PBR properties. Player character parts are sent separately each frame so they update without triggering a full scene reload.

**Lights:** Any `PointLight`, `SpotLight`, or `SurfaceLight` that's enabled and attached to a BasePart.

**World:** Sun direction/color/intensity, sky gradient colors, ambient light, fog, bloom settings. Pulled from the `Lighting` service. If an `Atmosphere` object is present, its density and haze are mapped to fog. If a `BloomEffect` is present, its settings are used as defaults.

**Camera:** Position, forward/right/up vectors, and FOV. Sent every frame from the client.

---

## Primitives

Parts are classified into three types:

| Roblox shape | Sent as |
|---|---|
| `Part` (default block) | `box` |
| `Part` with `Shape = Ball` | `sphere` |
| `Part` with `Shape = Cylinder` | `cylinder` |
| `MeshPart`, `SpecialMesh`, etc. | `box` (bounding box only) |

Mesh rendering isn't supported yet — mesh parts use their bounding box as a box primitive.

---

## RemoteEvent

The scripts create a `RemoteEvent` named `GpuPathtraceEvent` in `ReplicatedStorage` automatically on server start. You don't need to create it manually.

Message types sent from client → server:

| Type | Payload | Description |
|---|---|---|
| `RequestFrame` | `{ camera, seq }` | Request a render with the current camera |
| `ToggleRecord` | none | Toggle recording on the server |
| `Ping` | none | Health check |
