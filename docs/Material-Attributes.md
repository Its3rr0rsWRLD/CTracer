# Material Attributes

CTracer infers PBR material properties from each part's Roblox `Material` enum. You can override any of these per-part using **Instance Attributes** in Studio.

To add an attribute: select a part → Properties panel → Attributes section → click **+** → name it and set the type.

---

## Attribute reference

| Attribute | Type | Description |
|---|---|---|
| `RT_Roughness` | `number` | Surface roughness. `0` = perfect mirror, `1` = fully diffuse |
| `RT_Metallic` | `number` | Metallic factor. `0` = dielectric (plastic/glass), `1` = metal |
| `RT_Reflectance` | `number` | Extra reflectance on top of material defaults |
| `RT_Transparency` | `number` | Overrides the part's built-in Transparency value |
| `RT_IOR` | `number` | Index of refraction for transparent materials. Glass = `1.5`, ice = `1.31`, water = `1.33` |
| `RT_Albedo` | `Color3` | Overrides the part's color |
| `RT_EmissionStrength` | `number` | How bright the part glows. `0` = off, `10`+ = very bright |
| `RT_EmissionColor` | `Color3` | Color of the emission. Defaults to the part's color if not set |

---

## Material defaults

If no `RT_` attribute is set, these defaults are used based on the part's material:

| Material | Roughness | Metallic | Transparency | IOR |
|---|---|---|---|---|
| SmoothPlastic | 0.15 | — | — | — |
| Plastic | 0.22 | — | — | — |
| Metal | 0.30 | 0.9 | — | — |
| DiamondPlate | 0.35 | 0.85 | — | — |
| Foil | 0.05 | 1.0 | — | — |
| CorrodedMetal | 0.75 | 0.7 | — | — |
| Glass | 0.02 | — | 0.8 | 1.5 |
| Ice | 0.02 | — | 0.3 | 1.31 |
| Neon | 0.05 | — | — | — |
| Wood / WoodPlanks | 0.60–0.65 | — | — | — |
| Brick / Concrete | 0.80–0.85 | — | — | — |
| Grass / Sand / Snow | 0.80–0.95 | — | — | — |
| Marble | 0.20 | — | — | — |

Everything else defaults to `roughness = 0.5`.

---

## Special materials

**Neon** — automatically gets `RT_EmissionStrength = 12` using the part's color. Override with `RT_EmissionStrength` if you want a different brightness.

**CrackedLava** — automatically emits orange/red light (`strength = 4`). Override with `RT_EmissionColor` and `RT_EmissionStrength` to change it.

---

## World attributes

You can also set attributes on the `Lighting` service to override environment settings:

| Attribute | Type | Description |
|---|---|---|
| `RT_SunDir` | `Vector3` | Sun direction override |
| `RT_SunColor` | `Color3` | Sun color |
| `RT_SunIntensity` | `number` | Sun brightness |
| `RT_SunAngularRadius` | `number` | Angular size of the sun disc (radians) |
| `RT_BGTop` | `Color3` | Sky gradient top color |
| `RT_BGBottom` | `Color3` | Sky gradient bottom color |
| `RT_EnvIntensity` | `number` | Environment light multiplier |
| `RT_Ambient` | `Color3` | Ambient light color |
| `RT_AmbientIntensity` | `number` | Ambient light brightness |
| `RT_FogDensity` | `number` | Fog density (0 = no fog) |
| `RT_FogColor` | `Color3` | Fog color |
| `RT_Exposure` | `number` | Scene exposure |
| `RT_Gamma` | `number` | Gamma correction |
| `RT_BloomStrength` | `number` | Bloom intensity |
| `RT_BloomThreshold` | `number` | Minimum brightness for bloom |

If none of these are set, CTracer reads the values from Lighting, Atmosphere, Sky, and BloomEffect automatically.
