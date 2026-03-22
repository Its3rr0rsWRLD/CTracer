/* postprocess.cuh */
#ifndef POSTPROCESS_CUH
#define POSTPROCESS_CUH

#include "cuda_utils.cuh"

/* ACES filmic tonemapping */
__device__ __forceinline__ float aces_tonemap(float x) {
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return fminf(fmaxf((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f), 1.0f);
}

#endif /* POSTPROCESS_CUH */
