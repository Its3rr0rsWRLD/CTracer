/* I literally think this is magic. I don't know what any of this means 😭 */
#ifndef SHADE_CUH
#define SHADE_CUH

#include "cuda_utils.cuh"

/* Fresnel functions */

/* Schlick approximation for conductors/dielectrics */
__device__ __forceinline__ float3 fresnel_schlick(float cos_theta, float3 f0) {
    float t = 1.0f - fminf(fmaxf(cos_theta, 0.0f), 1.0f);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return make_float3(
        f0.x + (1.0f - f0.x) * t5,
        f0.y + (1.0f - f0.y) * t5,
        f0.z + (1.0f - f0.z) * t5
    );
}

/* Fresnel for dielectric glass (returns reflectance probability) */
__device__ __forceinline__ float fresnel_dielectric(float cos_i, float eta) {
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t > 1.0f) return 1.0f;  /* total internal reflection */
    float cos_t = sqrtf(fmaxf(1.0f - sin2_t, 0.0f));
    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t + 1e-8f);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t + 1e-8f);
    return fminf(fmaxf((rs * rs + rp * rp) * 0.5f, 0.0f), 1.0f);
}

/* GGX Microfacet BRDF */

/* GGX normal distribution function */
__device__ __forceinline__ float ggx_D(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0f) + 1.0f;
    return a2 / (CT_PI_D * denom * denom + 1e-6f);
}

/* Smith G1 for GGX */
__device__ __forceinline__ float smith_g1(float ndv, float alpha) {
    float a2 = alpha * alpha;
    return (2.0f * ndv) / (ndv + sqrtf(a2 + (1.0f - a2) * ndv * ndv) + 1e-6f);
}

/* Combined Smith G for GGX */
__device__ __forceinline__ float ggx_G(float n_dot_v, float n_dot_l, float alpha) {
    return smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
}

/* eval_brdf_cos - Evaluate Cook-Torrance BRDF * cos(theta) */
__device__ __forceinline__ float3 eval_brdf_cos(
    float3 base, float rough, float metal, float refl,
    float3 n, float3 v, float3 l)
{
    float alpha = fmaxf(rough * rough, 0.02f);
    float n_dot_l = fmaxf(f3_dot(n, l), 0.0f);
    float n_dot_v = fmaxf(f3_dot(n, v), 0.001f);

    float3 h = f3_normalize(f3_add(l, v));
    float n_dot_h = fmaxf(f3_dot(n, h), 0.0f);
    float v_dot_h = fmaxf(f3_dot(v, h), 0.0f);

    /* F0: blend between dielectric (0.04) and metallic (base color) */
    float3 f0 = f3_lerp(make_float3(0.04f, 0.04f, 0.04f), base, metal);
    /* Reflectance boost */
    f0 = f3_lerp(f0, make_float3(1.0f, 1.0f, 1.0f), refl);

    float3 F = fresnel_schlick(v_dot_h, f0);
    float D = ggx_D(n_dot_h, alpha);
    float G = ggx_G(n_dot_v, n_dot_l, alpha);

    /* Specular: DGF / (4 * NdotV * NdotL) */
    float spec_denom = 4.0f * n_dot_v * n_dot_l + 1e-6f;
    float3 spec = f3_scale(F, D * G / spec_denom);

    /* Diffuse: kd * base / pi */
    float3 kd = make_float3(
        (1.0f - F.x) * (1.0f - metal),
        (1.0f - F.y) * (1.0f - metal),
        (1.0f - F.z) * (1.0f - metal)
    );
    float3 diff = f3_scale(f3_mul(kd, base), 1.0f / CT_PI_D);

    return f3_scale(f3_add(diff, spec), n_dot_l);
}

/* Sampling */

/* Cosine-weighted hemisphere sampling */
__device__ __forceinline__ float3 cosine_hemisphere(float3 n, RNGState *rng) {
    float u1 = rng_float(rng);
    float u2 = rng_float(rng);
    float r = sqrtf(u1);
    float phi = 2.0f * CT_PI_D * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(1.0f - u1, 0.0f));

    float3 t, b;
    build_basis(n, &t, &b);
    return f3_normalize(f3_add(f3_add(f3_scale(t, x), f3_scale(b, y)), f3_scale(n, z)));
}

/* Random direction on unit sphere */
__device__ __forceinline__ float3 random_sphere_dir(RNGState *rng) {
    float u1 = rng_float(rng);
    float u2 = rng_float(rng);
    float theta = 2.0f * CT_PI_D * u1;
    float z = 2.0f * u2 - 1.0f;
    float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    return make_float3(r * cosf(theta), r * sinf(theta), z);
}

/* Sample direction within a cone around a given direction */
__device__ __forceinline__ float3 sample_cone(float3 direction, float half_angle, RNGState *rng) {
    float cos_max = cosf(half_angle);
    float u1 = rng_float(rng);
    float u2 = rng_float(rng);
    float z = 1.0f - u1 * (1.0f - cos_max);
    float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    float phi = 2.0f * CT_PI_D * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    float3 t, b;
    build_basis(direction, &t, &b);
    return f3_normalize(f3_add(f3_add(f3_scale(t, x), f3_scale(b, y)), f3_scale(direction, z)));
}

/* Environment sampling */

/* Sky gradient based on ray Y direction */
__device__ __forceinline__ float3 sample_env(float3 dir, const KernelWorld *world) {
    float y = fminf(fmaxf(dir.y, -1.0f), 1.0f);
    float blend = (y + 1.0f) * 0.5f;
    float3 env = f3_add(
        f3_scale(world->sky_top, blend),
        f3_scale(world->sky_bottom, 1.0f - blend)
    );
    return f3_scale(env, world->env_intensity);
}

#endif