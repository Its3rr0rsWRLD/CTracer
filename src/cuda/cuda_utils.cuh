/* cuda_utils.cuh - math, RNG, and scene types for kernels */
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

/* Constants */
#define CT_PI_D     3.14159265358979323846f
#define CT_EPS_D    1e-4f
#define CT_INF_D    1e30f

/* Primitive type IDs (must match ctracer.h) */
#define PRIM_BOX      0
#define PRIM_SPHERE   1
#define PRIM_CYLINDER 2
#define PRIM_MESH     3

/* RNG (no curand) */
typedef struct {
    unsigned int state;
} RNGState;

__device__ __forceinline__ unsigned int rng_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ void rng_init(RNGState *rng, unsigned int pixel, unsigned int sample, unsigned int frame) {
    rng->state = rng_hash(pixel * 1973u + sample * 9277u + frame * 26699u + 1u);
}

__device__ __forceinline__ float rng_float(RNGState *rng) {
    rng->state = rng_hash(rng->state);
    return (float)(rng->state & 0x00FFFFFFu) / 16777216.0f;  /* [0, 1) */
}

__device__ __forceinline__ float3 f3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 f3_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 f3_mul(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 f3_scale(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float f3_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 f3_cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float f3_length(float3 v) {
    return sqrtf(f3_dot(v, v));
}

__device__ __forceinline__ float3 f3_normalize(float3 v) {
    float inv_len = rsqrtf(fmaxf(f3_dot(v, v), 1e-16f));
    return f3_scale(v, inv_len);
}

__device__ __forceinline__ float3 f3_lerp(float3 a, float3 b, float t) {
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    );
}

__device__ __forceinline__ float3 f3_clamp01(float3 v) {
    return make_float3(
        fminf(fmaxf(v.x, 0.0f), 1.0f),
        fminf(fmaxf(v.y, 0.0f), 1.0f),
        fminf(fmaxf(v.z, 0.0f), 1.0f)
    );
}

__device__ __forceinline__ float3 f3_max(float3 v, float m) {
    return make_float3(fmaxf(v.x, m), fmaxf(v.y, m), fmaxf(v.z, m));
}

__device__ __forceinline__ float3 f3_min(float3 v, float m) {
    return make_float3(fminf(v.x, m), fminf(v.y, m), fminf(v.z, m));
}

__device__ __forceinline__ float f3_max_comp(float3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

__device__ __forceinline__ float f3_luminance(float3 v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

__device__ __forceinline__ float3 f3_reflect(float3 v, float3 n) {
    float d = f3_dot(v, n);
    return f3_sub(v, f3_scale(n, 2.0f * d));
}

__device__ __forceinline__ float3 f3_neg(float3 v) {
    return make_float3(-v.x, -v.y, -v.z);
}

/* tangent space */
__device__ __forceinline__ void build_basis(float3 n, float3 *t, float3 *b) {
    float3 up;
    if (fabsf(n.y) > 0.999f) {
        up = make_float3(1.0f, 0.0f, 0.0f);
    } else {
        up = make_float3(0.0f, 1.0f, 0.0f);
    }
    *t = f3_normalize(f3_cross(up, n));
    *b = f3_cross(n, *t);
}

typedef struct {
    const float * __restrict__ pos_x;
    const float * __restrict__ pos_y;
    const float * __restrict__ pos_z;
    const float * __restrict__ half_x;
    const float * __restrict__ half_y;
    const float * __restrict__ half_z;
    const float * __restrict__ rot;       /* 9*N floats row-major */
    const int   * __restrict__ prim_type;
    const float * __restrict__ color_r;
    const float * __restrict__ color_g;
    const float * __restrict__ color_b;
    const float * __restrict__ roughness;
    const float * __restrict__ metallic;
    const float * __restrict__ reflectance;
    const float * __restrict__ transparency;
    const float * __restrict__ ior;
    const float * __restrict__ emission_r;
    const float * __restrict__ emission_g;
    const float * __restrict__ emission_b;
    int num_objects;

    /* Per-object mesh reference (PRIM_MESH only) */
    const int   * __restrict__ mesh_tri_offset;
    const int   * __restrict__ mesh_tri_count;

    /* Global triangle buffer (all meshes, local-space vertices) */
    const float * __restrict__ tri_v0_x;
    const float * __restrict__ tri_v0_y;
    const float * __restrict__ tri_v0_z;
    const float * __restrict__ tri_v1_x;
    const float * __restrict__ tri_v1_y;
    const float * __restrict__ tri_v1_z;
    const float * __restrict__ tri_v2_x;
    const float * __restrict__ tri_v2_y;
    const float * __restrict__ tri_v2_z;
    int total_triangles;

    /* NEE emissive data */
    const int   * __restrict__ emissive_ids;
    const float * __restrict__ emissive_centers_x;
    const float * __restrict__ emissive_centers_y;
    const float * __restrict__ emissive_centers_z;
    const float * __restrict__ emissive_radii;
    const float * __restrict__ emissive_emission_r;
    const float * __restrict__ emissive_emission_g;
    const float * __restrict__ emissive_emission_b;
    const float * __restrict__ emissive_probs;
    int num_emissive;
} KernelScene;

typedef struct {
    const float * __restrict__ pos_x;
    const float * __restrict__ pos_y;
    const float * __restrict__ pos_z;
    const float * __restrict__ color_r;
    const float * __restrict__ color_g;
    const float * __restrict__ color_b;
    const float * __restrict__ brightness;
    const float * __restrict__ range;
    const float * __restrict__ dir_x;
    const float * __restrict__ dir_y;
    const float * __restrict__ dir_z;
    const float * __restrict__ angle;
    const int   * __restrict__ is_spot;
    int count;
} KernelLights;

typedef struct {
    float3 sky_top, sky_bottom;
    float  env_intensity;
    float3 sun_dir, sun_color;
    float  sun_intensity, sun_angular_radius;
    float3 ambient;
    float  ambient_intensity;
    float  fog_density;
    float3 fog_color;
} KernelWorld;

typedef struct {
    float3 position, forward, right, up;
    float  fov_rad;
} KernelCamera;

typedef struct {
    int  width, height;
    int  spp, max_bounces;
    int  do_shadows, do_reflections, do_gi, do_nee;
    int  do_ao, ao_samples;
    float ao_radius;
    unsigned int frame_seed;
} KernelParams;

#endif /* CUDA_UTILS_CUH */
