# Render Window

When you run `ctracer.exe`, a window opens with the render preview on the left and a settings panel on the right.

---

## Preview

The left side shows the current render. It starts noisy and cleans up over time as samples accumulate (temporal accumulation). The longer you hold the camera still, the cleaner it gets.

Moving the camera resets the accumulation and starts fresh from 1 sample.

The sample counter in the panel shows how many frames have been accumulated since the last reset.

---

## Settings panel

All settings take effect immediately without restarting.

### Render

| Setting | Description |
|---|---|
| **SPP** | Samples per pixel per frame. 1 = fast and noisy, higher = slower but cleaner per frame |
| **Bounces** | Max number of ray bounces. More bounces = more accurate indirect light, slower render |
| **Width / Height** | Render resolution. Changing this resets accumulation |

### Post-processing

| Setting | Description |
|---|---|
| **Exposure** | Brightens or darkens the final image |
| **Gamma** | Gamma correction (default 2.2) |
| **Bloom strength** | How strong the bloom glow is (0 = off) |
| **Bloom threshold** | How bright a pixel needs to be before bloom kicks in |

Changes to Exposure and Gamma in the panel override whatever Studio sends — you'll see an indicator when a UI override is active. To go back to Studio-controlled values, reset them.

### Stats

The panel shows:
- Render time (ms per frame)
- GPU memory used
- Current sample count
- Last scene upload time
