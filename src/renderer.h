#ifndef RENDERER_H
#define RENDERER_H

#include "ctracer.h"


#ifdef __cplusplus
extern "C" {
#endif

void gpu_scene_alloc(GPUScene *gs, int num_objects, int *capacity);
void gpu_scene_upload(GPUScene *gs, const SceneObject *objects, int num_objects, void *stream);
void gpu_scene_upload_player(GPUScene *gs, const SceneObject *objects, int num_objects, void *stream);
void gpu_scene_free(GPUScene *gs);
void gpu_lights_alloc(GPULights *gl, int count, int *capacity);
void gpu_lights_upload(GPULights *gl, const LightData *lights, int count, void *stream);
void gpu_lights_free(GPULights *gl);
void gpu_print_info(void);

void launch_clear_accum(float *r, float *g, float *b, int total, void *stream);
void launch_temporal_blend(float *hist_r, float *hist_g, float *hist_b,
                           const float *new_r, const float *new_g, const float *new_b,
                           float *hdr_r, float *hdr_g, float *hdr_b,
                           int total, float alpha, void *stream);
void launch_bloom(float *hr, float *hg, float *hb, float *scratch,
                  int w, int h, float strength, float threshold, void *stream);
void launch_tonemap(const float *hr, const float *hg, const float *hb,
                    unsigned char *ldr, int total, float exposure, float gamma, void *stream);

#ifdef __cplusplus
}
#endif

#endif
