# HTTP API

CTracer listens on `http://127.0.0.1:8000` by default. The Studio scripts use this to send scene data and trigger renders. You can also hit these endpoints yourself if you want to integrate something custom.

---

## Endpoints

### `POST /scene`

Upload the full scene. Call this whenever the scene changes. CTracer uploads everything to the GPU and resets temporal accumulation.

**Body:**
```json
{
  "objects": [ ... ],
  "lights": [ ... ],
  "world": { ... }
}
```

**Response:**
```json
{ "status": "ok", "objects": 42, "lights": 3 }
```

---

### `POST /render`

Send a camera and trigger a render frame. CTracer renders one frame and returns a confirmation. This call blocks until the frame is done (up to 5 seconds timeout).

**Body:**
```json
{
  "camera": {
    "position": [x, y, z],
    "forward": [x, y, z],
    "right":   [x, y, z],
    "up":      [x, y, z],
    "fov": 70.0
  },
  "seq": 1
}
```

You can also include `objects`, `lights`, and `world` in the same request as `/scene` if you want to do everything in one call.

**Response:**
```json
{ "type": "frame", "width": 256, "height": 144, "sample": 12, "seq": 1 }
```

---

### `GET /health`

Check if ctracer is running.

**Response:**
```json
{ "status": "ok", "device": "cuda", "cuda": true }
```

---

### `POST /reset`

Reset temporal accumulation. Useful if the scene changed but you didn't send a new `/scene` yet.

**Response:**
```json
{ "status": "reset" }
```

---

## Object format

Each entry in `objects`:

```json
{
  "primitive": "box",
  "position":  [x, y, z],
  "size":      [x, y, z],
  "rotation":  [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
  "color":     [r, g, b],
  "roughness":    0.5,
  "metallic":     0.0,
  "reflectance":  0.0,
  "transparency": 0.0,
  "ior":          1.5,
  "emission":     [0, 0, 0]
}
```

`primitive` values: `"box"`, `"sphere"`, `"cylinder"`

`rotation` is a 3×3 row-major matrix. Roblox CFrame rows map directly: right, up, look.

`color` and `emission` are linear RGB (0–1 range). Gamma correction happens on the C side.

---

## Light format

Each entry in `lights`:

```json
{
  "type":       "point",
  "position":   [x, y, z],
  "color":      [r, g, b],
  "brightness": 1.0,
  "range":      60.0,
  "direction":  [x, y, z],
  "angle":      0.785
}
```

`type` values: `"point"`, `"spot"`, `"surface"`

`direction` and `angle` are only used for `"spot"` and `"surface"`. `angle` is in radians.

---

## World format

```json
{
  "sunDirection":    [x, y, z],
  "sunColor":        [r, g, b],
  "sunIntensity":    1.0,
  "sunAngularRadius": 0.02,
  "skyTop":          [r, g, b],
  "skyBottom":       [r, g, b],
  "envIntensity":    1.0,
  "ambientColor":    [r, g, b],
  "ambientIntensity": 0.3,
  "fogDensity":      0.0,
  "fogColor":        [r, g, b],
  "exposure":        1.0,
  "gamma":           2.2,
  "bloomStrength":   0.0,
  "bloomThreshold":  2.0
}
```

All color values are linear RGB. `fogDensity` is an exponential fog coefficient — values above `0.02` get very thick very fast.
