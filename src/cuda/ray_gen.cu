/* ray_gen.cu - debug kernel, not used in prod */

#include "cuda_utils.cuh"
#include "ray_gen.cuh"

__global__ void ray_gen_debug_kernel(
    const KernelCamera cam,
    int width, int height,
    float *dir_rgb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int px = idx % width;
    int py = idx / width;

    RNGState rng;
    rng_init(&rng, idx, 0, 0);

    float3 origin, dir;
    generate_primary_ray(&cam, px, py, width, height, &rng, &origin, &dir);

    dir_rgb[idx * 3 + 0] = dir.x * 0.5f + 0.5f;
    dir_rgb[idx * 3 + 1] = dir.y * 0.5f + 0.5f;
    dir_rgb[idx * 3 + 2] = dir.z * 0.5f + 0.5f;
}

extern "C" {
void launch_ray_gen_debug(const KernelCamera *cam, int width, int height,
                          float *dir_rgb, cudaStream_t stream)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    ray_gen_debug_kernel<<<grid, block, 0, stream>>>(*cam, width, height, dir_rgb);
}
}
