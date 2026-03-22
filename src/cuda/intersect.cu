/* intersect.cu - debug depth buffer kernel */

#include "cuda_utils.cuh"
#include "ray_gen.cuh"
#include "intersect.cuh"

__global__ void depth_buffer_kernel(
    const KernelScene  scene,
    const KernelCamera cam,
    int width, int height,
    float max_depth,
    float *depth_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int px = idx % width;
    int py = idx / width;

    RNGState rng;
    rng_init(&rng, idx, 0, 0);

    float3 origin, dir;
    generate_primary_ray(&cam, px, py, width, height, &rng, &origin, &dir);

    float t_hit;
    int hit_id;
    intersect_all(origin, dir, &scene, &t_hit, &hit_id);

    if (hit_id >= 0) {
        depth_out[idx] = fminf(t_hit / max_depth, 1.0f);
    } else {
        depth_out[idx] = 1.0f;
    }
}

extern "C" {
void launch_depth_buffer(const KernelScene *scene, const KernelCamera *cam,
                         int width, int height, float max_depth,
                         float *depth_out, cudaStream_t stream)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    depth_buffer_kernel<<<grid, block, 0, stream>>>(*scene, *cam, width, height,
                                                     max_depth, depth_out);
}
}
