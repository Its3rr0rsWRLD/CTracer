#include <cuda_runtime_api.h>

#include "renderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CT_MAX_TEMPORAL_FRAMES 128 /* Todo: Let the user decide */

/* Idk I found this on stackoverflow 😭 */
#define CHECK_CUDA_HOST(call) do {
    cudaError_t err = (call);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
} while(0)

typedef struct {
    const float *pos_x, *pos_y, *pos_z;
    const float *half_x, *half_y, *half_z;
    const float *rot;
    const int   *prim_type;
    const float *color_r, *color_g, *color_b;
    const float *roughness, *metallic, *reflectance;
    const float *transparency, *ior;
    const float *emission_r, *emission_g, *emission_b;
    int num_objects;
    const int   *mesh_tri_offset;
    const int   *mesh_tri_count;
    const float *tri_v0_x, *tri_v0_y, *tri_v0_z;
    const float *tri_v1_x, *tri_v1_y, *tri_v1_z;
    const float *tri_v2_x, *tri_v2_y, *tri_v2_z;
    int total_triangles;
    const int   *emissive_ids;
    const float *emissive_centers_x, *emissive_centers_y, *emissive_centers_z;
    const float *emissive_radii;
    const float *emissive_emission_r, *emissive_emission_g, *emissive_emission_b;
    const float *emissive_probs;
    int num_emissive;
} KSceneBridge;

typedef struct {
    const float *pos_x, *pos_y, *pos_z;
    const float *color_r, *color_g, *color_b;
    const float *brightness, *range;
    const float *dir_x, *dir_y, *dir_z;
    const float *angle;
    const int   *is_spot;
    int count;
} KLightsBridge;

typedef struct {
    float sky_top[3];
    float sky_bottom[3];
    float env_intensity;
    float sun_dir[3];
    float sun_color[3];
    float sun_intensity, sun_angular_radius;
    float ambient[3];
    float ambient_intensity;
    float fog_density;
    float fog_color[3];
} KWorldBridge;

typedef struct {
    float position[3];
    float forward[3];
    float right[3];
    float up[3];
    float fov_rad;
} KCameraBridge;

typedef struct {
    int width, height;
    int spp, max_bounces;
    int do_shadows, do_reflections, do_gi, do_nee;
    int do_ao, ao_samples;
    float ao_radius;
    unsigned int frame_seed;
} KParamsBridge;

extern void launch_pathtrace(
    const void *scene_ptr, const void *lights_ptr, const void *world_ptr,
    const void *cam_ptr, const void *params_ptr,
    float *accum_r, float *accum_g, float *accum_b,
    void *stream);

/* Reset detection */
static uint64_t compute_cam_hash(const Camera *cam, int w, int h) {
    uint64_t hash = 14695981039346656037ULL;
    const unsigned char *data = (const unsigned char *)cam;
    for (size_t i = 0; i < sizeof(Camera); i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    hash ^= (uint64_t)w * 2654435761ULL;
    hash ^= (uint64_t)h * 40503ULL;
    return hash;
}

/* Initialization */
void renderer_init(RendererCtx *ctx) {
    memset(ctx, 0, sizeof(RendererCtx));

    /* Default settings */
    ctx->settings.width = CT_DEFAULT_WIDTH;
    ctx->settings.height = CT_DEFAULT_HEIGHT;
    ctx->settings.spp = CT_DEFAULT_SPP;
    ctx->settings.max_bounces = CT_DEFAULT_BOUNCES;
    ctx->settings.shadows = true;
    ctx->settings.reflections = true;
    ctx->settings.gi = true;
    ctx->settings.nee = true;
    ctx->settings.ao = true;
    ctx->settings.ao_samples = 4;
    ctx->settings.ao_radius = 5.0f;
    ctx->settings.bloom_strength = 0.0f;
    ctx->settings.bloom_threshold = 2.0f;
    ctx->settings.exposure = 0.6f;
    ctx->settings.gamma = 2.2f;

    world_settings_defaults(&ctx->world);

    /* Create CUDA streams */
    cudaStream_t rs, us, rbs;
    CHECK_CUDA_HOST(cudaStreamCreate(&rs));
    CHECK_CUDA_HOST(cudaStreamCreate(&us));
    CHECK_CUDA_HOST(cudaStreamCreate(&rbs));
    ctx->render_stream = rs;
    ctx->upload_stream = us;
    ctx->readback_stream = rbs;

    /* Allocate pixel buffers */
    renderer_resize(ctx, ctx->settings.width, ctx->settings.height);

    /* Print GPU info (Sometimes it doesn't like to detect my GPU) */
    gpu_print_info();
}

/* Resize pixel buffers */
void renderer_resize(RendererCtx *ctx, int width, int height) {
    if (width == ctx->buf_width && height == ctx->buf_height) return;

    int total = width * height;

    /* Free old buffers */
    if (ctx->d_history_r)    cudaFree(ctx->d_history_r);
    if (ctx->d_history_g)    cudaFree(ctx->d_history_g);
    if (ctx->d_history_b)    cudaFree(ctx->d_history_b);
    if (ctx->d_scratch_r)    cudaFree(ctx->d_scratch_r);
    if (ctx->d_scratch_g)    cudaFree(ctx->d_scratch_g);
    if (ctx->d_scratch_b)    cudaFree(ctx->d_scratch_b);
    if (ctx->d_hdr_r)        cudaFree(ctx->d_hdr_r);
    if (ctx->d_hdr_g)        cudaFree(ctx->d_hdr_g);
    if (ctx->d_hdr_b)        cudaFree(ctx->d_hdr_b);
    if (ctx->d_bloom_scratch) cudaFree(ctx->d_bloom_scratch);
    if (ctx->d_ldr)          cudaFree(ctx->d_ldr);
    if (ctx->h_pinned_output) cudaFreeHost(ctx->h_pinned_output);
    if (ctx->h_pinned_back)   cudaFreeHost(ctx->h_pinned_back);

    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_history_r, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_history_g, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_history_b, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMemset(ctx->d_history_r, 0, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMemset(ctx->d_history_g, 0, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMemset(ctx->d_history_b, 0, total * sizeof(float)));

    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_scratch_r, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_scratch_g, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_scratch_b, total * sizeof(float)));

    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_hdr_r, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_hdr_g, total * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_hdr_b, total * sizeof(float)));

    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_bloom_scratch, total * 3 * sizeof(float)));
    CHECK_CUDA_HOST(cudaMalloc(&ctx->d_ldr, total * 4));

    CHECK_CUDA_HOST(cudaHostAlloc(&ctx->h_pinned_output, total * 4, cudaHostAllocDefault));
    CHECK_CUDA_HOST(cudaHostAlloc(&ctx->h_pinned_back, total * 4, cudaHostAllocDefault));

    ctx->buf_width = width;
    ctx->buf_height = height;
    ctx->temporal_frame = 0;
    ctx->sample_count = 0;

    /* Track GPU memory usage */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    ctx->gpu_memory_used_mb = (float)(total_mem - free_mem) / 1048576.0f;
}

/* Scene upload */
void renderer_upload_scene(RendererCtx *ctx, const SceneObject *objects, int num_objects,
                           const LightData *lights, int num_lights)
{
    gpu_scene_alloc(&ctx->gpu_scene, num_objects, &ctx->scene_capacity);
    gpu_scene_upload(&ctx->gpu_scene, objects, num_objects, ctx->upload_stream);

    gpu_lights_alloc(&ctx->gpu_lights, num_lights, &ctx->lights_capacity);
    gpu_lights_upload(&ctx->gpu_lights, lights, num_lights, ctx->upload_stream);

    /* Track how many are static for partial updates */
    int static_count = 0;
    for (int i = 0; i < num_objects; i++) {
        if (!objects[i].is_player) static_count++;
    }
    ctx->gpu_scene.num_static_objects = static_count;

    printf("Scene uploaded: %d objects (%d static), %d lights, %d emissive\n",
           num_objects, static_count, num_lights, ctx->gpu_scene.num_emissive);
}

/* Upload player objects */
void renderer_upload_player_objects(RendererCtx *ctx, const SceneObject *objects, int num_objects)
{
    if (num_objects <= 0 || !objects) return;
    gpu_scene_upload_player(&ctx->gpu_scene, objects, num_objects, ctx->upload_stream);
}

/* Set world settings */
void renderer_set_world(RendererCtx *ctx, const WorldSettings *world) {
    /* Save user values */
    float saved_exposure = ctx->world.exposure;
    float saved_gamma = ctx->world.gamma;

    /* Copy all game-controlled values */
    ctx->world = *world;

    /* Restore UI overrides */
    if (ctx->ui_overrides & UI_OVERRIDE_EXPOSURE)
        ctx->world.exposure = saved_exposure;
    if (ctx->ui_overrides & UI_OVERRIDE_GAMMA)
        ctx->world.gamma = saved_gamma;

    /* Sync exposure/gamma to render settings from world */
    if (!(ctx->ui_overrides & UI_OVERRIDE_EXPOSURE))
        ctx->settings.exposure = ctx->world.exposure;
    if (!(ctx->ui_overrides & UI_OVERRIDE_GAMMA))
        ctx->settings.gamma = ctx->world.gamma;

    /* Update bloom*/
    if (world->bloom_strength > 0.0f)
        ctx->settings.bloom_strength = world->bloom_strength;
    if (world->bloom_threshold > 0.0f)
        ctx->settings.bloom_threshold = world->bloom_threshold;
}

void renderer_set_camera(RendererCtx *ctx, const Camera *cam) {
    ctx->camera = *cam;
}

/* Reset temporal accumulation */
void renderer_reset_accumulation(RendererCtx *ctx) {
    ctx->temporal_frame = 0;
}

void renderer_render(RendererCtx *ctx) {
    int w = ctx->settings.width;
    int h = ctx->settings.height;
    int total = w * h;

    if (total == 0 || ctx->gpu_scene.num_objects == 0) return;

    /* Resize if needed */
    if (w != ctx->buf_width || h != ctx->buf_height) {
        renderer_resize(ctx, w, h);
    }

    /* Check camera change */
    uint64_t new_hash = compute_cam_hash(&ctx->camera, w, h);
    static uint64_t s_cam_hash = 0;
    if (new_hash != s_cam_hash) {
        renderer_reset_accumulation(ctx);
        s_cam_hash = new_hash;
    }

    /* Compute temporal blend alpha (idk how this works lwk) */
    int frame = ctx->temporal_frame;
    int effective = (frame + 1 < CT_MAX_TEMPORAL_FRAMES) ? frame + 1 : CT_MAX_TEMPORAL_FRAMES;
    float alpha = 1.0f / (float)effective;

    /* ignore this shit bro */
    KSceneBridge ks;
    memset(&ks, 0, sizeof(ks));
    ks.pos_x = ctx->gpu_scene.pos_x;
    ks.pos_y = ctx->gpu_scene.pos_y;
    ks.pos_z = ctx->gpu_scene.pos_z;
    ks.half_x = ctx->gpu_scene.half_x;
    ks.half_y = ctx->gpu_scene.half_y;
    ks.half_z = ctx->gpu_scene.half_z;
    ks.rot = ctx->gpu_scene.rot;
    ks.prim_type = ctx->gpu_scene.prim_type;
    ks.color_r = ctx->gpu_scene.color_r;
    ks.color_g = ctx->gpu_scene.color_g;
    ks.color_b = ctx->gpu_scene.color_b;
    ks.roughness = ctx->gpu_scene.roughness;
    ks.metallic = ctx->gpu_scene.metallic;
    ks.reflectance = ctx->gpu_scene.reflectance;
    ks.transparency = ctx->gpu_scene.transparency;
    ks.ior = ctx->gpu_scene.ior;
    ks.emission_r = ctx->gpu_scene.emission_r;
    ks.emission_g = ctx->gpu_scene.emission_g;
    ks.emission_b = ctx->gpu_scene.emission_b;
    ks.num_objects = ctx->gpu_scene.num_objects;
    ks.mesh_tri_offset = ctx->gpu_scene.mesh_tri_offset;
    ks.mesh_tri_count = ctx->gpu_scene.mesh_tri_count;
    ks.tri_v0_x = ctx->gpu_scene.tri_v0_x;
    ks.tri_v0_y = ctx->gpu_scene.tri_v0_y;
    ks.tri_v0_z = ctx->gpu_scene.tri_v0_z;
    ks.tri_v1_x = ctx->gpu_scene.tri_v1_x;
    ks.tri_v1_y = ctx->gpu_scene.tri_v1_y;
    ks.tri_v1_z = ctx->gpu_scene.tri_v1_z;
    ks.tri_v2_x = ctx->gpu_scene.tri_v2_x;
    ks.tri_v2_y = ctx->gpu_scene.tri_v2_y;
    ks.tri_v2_z = ctx->gpu_scene.tri_v2_z;
    ks.total_triangles = ctx->gpu_scene.total_triangles;
    ks.emissive_ids = ctx->gpu_scene.emissive_ids;
    ks.emissive_centers_x = ctx->gpu_scene.emissive_centers_x;
    ks.emissive_centers_y = ctx->gpu_scene.emissive_centers_y;
    ks.emissive_centers_z = ctx->gpu_scene.emissive_centers_z;
    ks.emissive_radii = ctx->gpu_scene.emissive_radii;
    ks.emissive_emission_r = ctx->gpu_scene.emissive_emission_r;
    ks.emissive_emission_g = ctx->gpu_scene.emissive_emission_g;
    ks.emissive_emission_b = ctx->gpu_scene.emissive_emission_b;
    ks.emissive_probs = ctx->gpu_scene.emissive_probs;
    ks.num_emissive = ctx->gpu_scene.num_emissive;

    KLightsBridge kl;
    memset(&kl, 0, sizeof(kl));
    kl.pos_x = ctx->gpu_lights.pos_x;
    kl.pos_y = ctx->gpu_lights.pos_y;
    kl.pos_z = ctx->gpu_lights.pos_z;
    kl.color_r = ctx->gpu_lights.color_r;
    kl.color_g = ctx->gpu_lights.color_g;
    kl.color_b = ctx->gpu_lights.color_b;
    kl.brightness = ctx->gpu_lights.brightness;
    kl.range = ctx->gpu_lights.range;
    kl.dir_x = ctx->gpu_lights.dir_x;
    kl.dir_y = ctx->gpu_lights.dir_y;
    kl.dir_z = ctx->gpu_lights.dir_z;
    kl.angle = ctx->gpu_lights.angle;
    kl.is_spot = ctx->gpu_lights.is_spot;
    kl.count = ctx->gpu_lights.count;

    /* Normalize sun direction */
    float sdx = ctx->world.sun_dir[0];
    float sdy = ctx->world.sun_dir[1];
    float sdz = ctx->world.sun_dir[2];
    float slen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
    if (slen > 1e-6f) { sdx /= slen; sdy /= slen; sdz /= slen; }

    KWorldBridge kw;
    kw.sky_top[0] = ctx->world.sky_top[0];
    kw.sky_top[1] = ctx->world.sky_top[1];
    kw.sky_top[2] = ctx->world.sky_top[2];
    kw.sky_bottom[0] = ctx->world.sky_bottom[0];
    kw.sky_bottom[1] = ctx->world.sky_bottom[1];
    kw.sky_bottom[2] = ctx->world.sky_bottom[2];
    kw.env_intensity = ctx->world.env_intensity;
    kw.sun_dir[0] = sdx; kw.sun_dir[1] = sdy; kw.sun_dir[2] = sdz;
    kw.sun_color[0] = ctx->world.sun_color[0];
    kw.sun_color[1] = ctx->world.sun_color[1];
    kw.sun_color[2] = ctx->world.sun_color[2];
    kw.sun_intensity = ctx->world.sun_intensity;
    kw.sun_angular_radius = ctx->world.sun_angular_radius;
    kw.ambient[0] = ctx->world.ambient[0] * ctx->world.ambient_intensity;
    kw.ambient[1] = ctx->world.ambient[1] * ctx->world.ambient_intensity;
    kw.ambient[2] = ctx->world.ambient[2] * ctx->world.ambient_intensity;
    kw.ambient_intensity = 1.0f;
    kw.fog_density = ctx->world.fog_density;
    kw.fog_color[0] = ctx->world.fog_color[0];
    kw.fog_color[1] = ctx->world.fog_color[1];
    kw.fog_color[2] = ctx->world.fog_color[2];

    KCameraBridge kc;
    kc.position[0] = ctx->camera.position[0];
    kc.position[1] = ctx->camera.position[1];
    kc.position[2] = ctx->camera.position[2];
    kc.forward[0] = ctx->camera.forward[0];
    kc.forward[1] = ctx->camera.forward[1];
    kc.forward[2] = ctx->camera.forward[2];
    kc.right[0] = ctx->camera.right[0];
    kc.right[1] = ctx->camera.right[1];
    kc.right[2] = ctx->camera.right[2];
    kc.up[0] = ctx->camera.up[0];
    kc.up[1] = ctx->camera.up[1];
    kc.up[2] = ctx->camera.up[2];
    kc.fov_rad = ctx->camera.fov * (CT_PI / 180.0f);

    KParamsBridge kp;
    kp.width = w;
    kp.height = h;
    kp.spp = ctx->settings.spp;
    kp.max_bounces = ctx->settings.max_bounces;
    kp.do_shadows = ctx->settings.shadows ? 1 : 0;
    kp.do_reflections = ctx->settings.reflections ? 1 : 0;
    kp.do_gi = ctx->settings.gi ? 1 : 0;
    kp.do_nee = ctx->settings.nee ? 1 : 0;
    kp.do_ao = ctx->settings.ao ? 1 : 0;
    kp.ao_samples = ctx->settings.ao_samples;
    kp.ao_radius = ctx->settings.ao_radius;
    kp.frame_seed = (unsigned int)(frame + 1);

    cudaEvent_t render_start, render_end;
    CHECK_CUDA_HOST(cudaEventCreate(&render_start));
    CHECK_CUDA_HOST(cudaEventCreate(&render_end));
    CHECK_CUDA_HOST(cudaEventRecord(render_start, (cudaStream_t)ctx->render_stream));

    launch_clear_accum(ctx->d_scratch_r, ctx->d_scratch_g, ctx->d_scratch_b,
                       total, ctx->render_stream);

    launch_pathtrace(&ks, &kl, &kw, &kc, &kp,
                     ctx->d_scratch_r, ctx->d_scratch_g, ctx->d_scratch_b,
                     ctx->render_stream);

    launch_temporal_blend(ctx->d_history_r, ctx->d_history_g, ctx->d_history_b,
                          ctx->d_scratch_r, ctx->d_scratch_g, ctx->d_scratch_b,
                          ctx->d_hdr_r, ctx->d_hdr_g, ctx->d_hdr_b,
                          total, alpha, ctx->render_stream);

    ctx->temporal_frame++;
    ctx->sample_count = (ctx->temporal_frame < CT_MAX_TEMPORAL_FRAMES)
                        ? ctx->temporal_frame : CT_MAX_TEMPORAL_FRAMES;

    /* Bloom */
    float bloom_str = ctx->settings.bloom_strength;
    float bloom_thr = ctx->settings.bloom_threshold;
    if (bloom_str > 0.0f) {
        launch_bloom(ctx->d_hdr_r, ctx->d_hdr_g, ctx->d_hdr_b,
                     ctx->d_bloom_scratch, w, h,
                     bloom_str, bloom_thr,
                     (cudaStream_t)ctx->render_stream);
    }

    float exposure = ctx->settings.exposure;
    float gamma = ctx->settings.gamma;
    if (exposure <= 0.0f) exposure = 0.6f;
    if (gamma <= 0.0f) gamma = 2.2f;
    launch_tonemap(ctx->d_hdr_r, ctx->d_hdr_g, ctx->d_hdr_b,
                   ctx->d_ldr, total, exposure, gamma,
                   (cudaStream_t)ctx->render_stream);

    CHECK_CUDA_HOST(cudaEventRecord(render_end, (cudaStream_t)ctx->render_stream));

    CHECK_CUDA_HOST(cudaMemcpyAsync(ctx->h_pinned_back, ctx->d_ldr,
                                    total * 4, cudaMemcpyDeviceToHost,
                                    (cudaStream_t)ctx->render_stream));
    CHECK_CUDA_HOST(cudaStreamSynchronize((cudaStream_t)ctx->render_stream));

    uint8_t *tmp = ctx->h_pinned_output;
    ctx->h_pinned_output = ctx->h_pinned_back;
    ctx->h_pinned_back = tmp;

    float ms;
    CHECK_CUDA_HOST(cudaEventElapsedTime(&ms, render_start, render_end));
    ctx->last_render_ms = ms;
    ctx->last_total_ms = ms;

    CHECK_CUDA_HOST(cudaEventDestroy(render_start));
    CHECK_CUDA_HOST(cudaEventDestroy(render_end));

    ct_atomic_store(&ctx->output_ready, 1);
}

/* Shutdown */
void renderer_shutdown(RendererCtx *ctx) {
    if (ctx->render_stream)   cudaStreamDestroy((cudaStream_t)ctx->render_stream);
    if (ctx->upload_stream)   cudaStreamDestroy((cudaStream_t)ctx->upload_stream);
    if (ctx->readback_stream) cudaStreamDestroy((cudaStream_t)ctx->readback_stream);

    if (ctx->d_history_r)    cudaFree(ctx->d_history_r);
    if (ctx->d_history_g)    cudaFree(ctx->d_history_g);
    if (ctx->d_history_b)    cudaFree(ctx->d_history_b);
    if (ctx->d_scratch_r)    cudaFree(ctx->d_scratch_r);
    if (ctx->d_scratch_g)    cudaFree(ctx->d_scratch_g);
    if (ctx->d_scratch_b)    cudaFree(ctx->d_scratch_b);
    if (ctx->d_hdr_r)         cudaFree(ctx->d_hdr_r);
    if (ctx->d_hdr_g)         cudaFree(ctx->d_hdr_g);
    if (ctx->d_hdr_b)         cudaFree(ctx->d_hdr_b);
    if (ctx->d_bloom_scratch) cudaFree(ctx->d_bloom_scratch);
    if (ctx->d_ldr)           cudaFree(ctx->d_ldr);
    if (ctx->h_pinned_output) cudaFreeHost(ctx->h_pinned_output);
    if (ctx->h_pinned_back)   cudaFreeHost(ctx->h_pinned_back);

    gpu_scene_free(&ctx->gpu_scene);
    gpu_lights_free(&ctx->gpu_lights);

    memset(ctx, 0, sizeof(RendererCtx));
    printf("Renderer shutdown complete\n");
}
