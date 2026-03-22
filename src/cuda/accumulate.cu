/* accumulate.cu - temporal blend (EMA) */

#include "cuda_utils.cuh"

__global__ void clear_accum_kernel(
    float *accum_r, float *accum_g, float *accum_b,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    accum_r[idx] = 0.0f;
    accum_g[idx] = 0.0f;
    accum_b[idx] = 0.0f;
}

__global__ void temporal_blend_kernel(
    float *hist_r, float *hist_g, float *hist_b,
    const float * __restrict__ new_r,
    const float * __restrict__ new_g,
    const float * __restrict__ new_b,
    float *hdr_r, float *hdr_g, float *hdr_b,
    int total_pixels, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float one_minus_alpha = 1.0f - alpha;

    float r = hist_r[idx] * one_minus_alpha + new_r[idx] * alpha;
    float g = hist_g[idx] * one_minus_alpha + new_g[idx] * alpha;
    float b = hist_b[idx] * one_minus_alpha + new_b[idx] * alpha;

    /* clamp bright pixels */
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float max_lum = 50.0f;
    if (lum > max_lum) {
        float scale = max_lum / lum;
        r *= scale;
        g *= scale;
        b *= scale;
    }

    hist_r[idx] = r;
    hist_g[idx] = g;
    hist_b[idx] = b;

    hdr_r[idx] = r;
    hdr_g[idx] = g;
    hdr_b[idx] = b;
}


extern "C" {

void launch_clear_accum(float *accum_r, float *accum_g, float *accum_b,
                        int total_pixels, void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    clear_accum_kernel<<<grid, block, 0, stream>>>(accum_r, accum_g, accum_b, total_pixels);
}

void launch_temporal_blend(float *hist_r, float *hist_g, float *hist_b,
                           const float *new_r, const float *new_g, const float *new_b,
                           float *hdr_r, float *hdr_g, float *hdr_b,
                           int total_pixels, float alpha, void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    temporal_blend_kernel<<<grid, block, 0, stream>>>(
        hist_r, hist_g, hist_b, new_r, new_g, new_b,
        hdr_r, hdr_g, hdr_b, total_pixels, alpha);
}

}
