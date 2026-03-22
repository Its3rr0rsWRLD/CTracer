/* ray_gen.cuh - primary ray generation */
#ifndef RAY_GEN_CUH
#define RAY_GEN_CUH

#include "cuda_utils.cuh"

__device__ __forceinline__ void generate_primary_ray(
    const KernelCamera *cam,
    int px, int py, int width, int height,
    RNGState *rng,
    float3 *origin, float3 *dir)
{
    float aspect = (float)width / (float)height;
    float half_h = tanf(cam->fov_rad * 0.5f);
    float half_w = half_h * aspect;

    /* Pixel center in NDC [-1, 1] with subpixel jitter */
    float jx = rng_float(rng) - 0.5f;
    float jy = rng_float(rng) - 0.5f;
    float u = (2.0f * ((float)px + 0.5f + jx) / (float)width - 1.0f) * half_w;
    float v = (1.0f - 2.0f * ((float)py + 0.5f + jy) / (float)height) * half_h;

    *origin = cam->position;
    *dir = f3_normalize(f3_add(f3_add(cam->forward, f3_scale(cam->right, u)),
                               f3_scale(cam->up, v)));
}

#endif /* RAY_GEN_CUH */
