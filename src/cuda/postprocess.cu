/* postprocess.cu - bloom and tonemap */

#include "cuda_utils.cuh"
#include "postprocess.cuh"

__global__ void bloom_downsample_kernel(
    const float * __restrict__ src_r,
    const float * __restrict__ src_g,
    const float * __restrict__ src_b,
    float *dst_r, float *dst_g, float *dst_b,
    int src_w, int src_h, int dst_w, int dst_h,
    float threshold, int is_first_level)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_w * dst_h) return;

    int dx = idx % dst_w;
    int dy = idx / dst_w;
    int sx = dx * 2;
    int sy = dy * 2;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    int count = 0;

    for (int oy = 0; oy < 2; oy++) {
        for (int ox = 0; ox < 2; ox++) {
            int px = sx + ox;
            int py = sy + oy;
            if (px < src_w && py < src_h) {
                int si = py * src_w + px;
                float sr = src_r[si];
                float sg = src_g[si];
                float sb = src_b[si];
                if (is_first_level) {
                    sr = fmaxf(sr - threshold, 0.0f);
                    sg = fmaxf(sg - threshold, 0.0f);
                    sb = fmaxf(sb - threshold, 0.0f);
                }
                r += sr; g += sg; b += sb;
                count++;
            }
        }
    }

    float inv = 1.0f / fmaxf((float)count, 1.0f);
    dst_r[idx] = r * inv;
    dst_g[idx] = g * inv;
    dst_b[idx] = b * inv;
}

__global__ void bloom_upsample_add_kernel(
    const float * __restrict__ src_r,
    const float * __restrict__ src_g,
    const float * __restrict__ src_b,
    float *dst_r, float *dst_g, float *dst_b,
    int src_w, int src_h, int dst_w, int dst_h,
    float strength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_w * dst_h) return;

    int dx = idx % dst_w;
    int dy = idx / dst_w;

    /* Bilinear sample from lower-res source */
    float sx = ((float)dx + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * (float)src_h / (float)dst_h - 0.5f;

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    int i00 = y0 * src_w + x0;
    int i10 = y0 * src_w + x1;
    int i01 = y1 * src_w + x0;
    int i11 = y1 * src_w + x1;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    float r = src_r[i00] * w00 + src_r[i10] * w10 + src_r[i01] * w01 + src_r[i11] * w11;
    float g = src_g[i00] * w00 + src_g[i10] * w10 + src_g[i01] * w01 + src_g[i11] * w11;
    float b = src_b[i00] * w00 + src_b[i10] * w10 + src_b[i01] * w01 + src_b[i11] * w11;

    dst_r[idx] += r * strength;
    dst_g[idx] += g * strength;
    dst_b[idx] += b * strength;
}

__global__ void tonemap_gamma_kernel(
    const float * __restrict__ hdr_r,
    const float * __restrict__ hdr_g,
    const float * __restrict__ hdr_b,
    unsigned char *ldr_rgba,
    int total_pixels,
    float exposure, float inv_gamma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float r = fminf(fmaxf(hdr_r[idx] * exposure, 0.0f), 64.0f);
    float g = fminf(fmaxf(hdr_g[idx] * exposure, 0.0f), 64.0f);
    float b = fminf(fmaxf(hdr_b[idx] * exposure, 0.0f), 64.0f);

    r = aces_tonemap(r);
    g = aces_tonemap(g);
    b = aces_tonemap(b);

    r = powf(fminf(fmaxf(r, 0.0f), 1.0f), inv_gamma);
    g = powf(fminf(fmaxf(g, 0.0f), 1.0f), inv_gamma);
    b = powf(fminf(fmaxf(b, 0.0f), 1.0f), inv_gamma);

    int out_idx = idx * 4;
    ldr_rgba[out_idx + 0] = (unsigned char)(r * 255.0f + 0.5f);
    ldr_rgba[out_idx + 1] = (unsigned char)(g * 255.0f + 0.5f);
    ldr_rgba[out_idx + 2] = (unsigned char)(b * 255.0f + 0.5f);
    ldr_rgba[out_idx + 3] = 255;
}


extern "C" {

#define BLOOM_MAX_LEVELS 5

void launch_bloom(
    float *hdr_r, float *hdr_g, float *hdr_b,
    float *scratch, /* must be large enough for all pyramid levels */
    int width, int height,
    float strength, float threshold,
    void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (strength <= 0.0f || width < 4 || height < 4) return;

    int level_w[BLOOM_MAX_LEVELS], level_h[BLOOM_MAX_LEVELS];
    int level_offset[BLOOM_MAX_LEVELS]; /* offset into scratch buffer (in pixels) */
    int num_levels = 0;
    int cur_w = width, cur_h = height;
    int offset = 0;

    for (int i = 0; i < BLOOM_MAX_LEVELS; i++) {
        int nw = cur_w / 2;
        int nh = cur_h / 2;
        if (nw < 2 || nh < 2) break;
        level_w[i] = nw;
        level_h[i] = nh;
        level_offset[i] = offset;
        offset += nw * nh;
        cur_w = nw;
        cur_h = nh;
        num_levels++;
    }

    if (num_levels == 0) return;

    /* scratch layout: [level0_R | level0_G | level0_B | level1_R | ...] */
    int total_scratch = offset;

    /* Downsample pass */
    for (int i = 0; i < num_levels; i++) {
        int dw = level_w[i], dh = level_h[i];
        int dpix = dw * dh;
        int block = 256;
        int grid = (dpix + block - 1) / block;

        float *dr = scratch + level_offset[i];
        float *dg = scratch + total_scratch + level_offset[i];
        float *db = scratch + total_scratch * 2 + level_offset[i];

        if (i == 0) {
            /* First level: downsample from HDR with threshold */
            int sw = width, sh = height;
            bloom_downsample_kernel<<<grid, block, 0, stream>>>(
                hdr_r, hdr_g, hdr_b, dr, dg, db,
                sw, sh, dw, dh, threshold, 1);
        } else {
            /* Subsequent levels: downsample from previous level */
            int sw = level_w[i-1], sh = level_h[i-1];
            float *sr = scratch + level_offset[i-1];
            float *sg = scratch + total_scratch + level_offset[i-1];
            float *sb = scratch + total_scratch * 2 + level_offset[i-1];
            bloom_downsample_kernel<<<grid, block, 0, stream>>>(
                sr, sg, sb, dr, dg, db,
                sw, sh, dw, dh, 0.0f, 0);
        }
    }

    /* Upsample pass */
    for (int i = num_levels - 1; i > 0; i--) {
        int sw = level_w[i], sh = level_h[i];
        int dw = level_w[i-1], dh = level_h[i-1];
        int dpix = dw * dh;
        int block = 256;
        int grid = (dpix + block - 1) / block;

        float *sr = scratch + level_offset[i];
        float *sg = scratch + total_scratch + level_offset[i];
        float *sb = scratch + total_scratch * 2 + level_offset[i];
        float *dr = scratch + level_offset[i-1];
        float *dg = scratch + total_scratch + level_offset[i-1];
        float *db = scratch + total_scratch * 2 + level_offset[i-1];

        bloom_upsample_add_kernel<<<grid, block, 0, stream>>>(
            sr, sg, sb, dr, dg, db,
            sw, sh, dw, dh, 1.0f);
    }

    /* Final upsample: add bloom level 0 back into HDR */
    {
        int dpix = width * height;
        int block = 256;
        int grid = (dpix + block - 1) / block;

        float *sr = scratch + level_offset[0];
        float *sg = scratch + total_scratch + level_offset[0];
        float *sb = scratch + total_scratch * 2 + level_offset[0];

        bloom_upsample_add_kernel<<<grid, block, 0, stream>>>(
            sr, sg, sb, hdr_r, hdr_g, hdr_b,
            level_w[0], level_h[0], width, height, strength);
    }
}

void launch_tonemap(
    const float *hdr_r, const float *hdr_g, const float *hdr_b,
    unsigned char *ldr_rgba,
    int total_pixels,
    float exposure, float gamma,
    void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    float inv_gamma = 1.0f / fmaxf(gamma, 1.0f);
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    tonemap_gamma_kernel<<<grid, block, 0, stream>>>(
        hdr_r, hdr_g, hdr_b, ldr_rgba, total_pixels, exposure, inv_gamma);
}

}
