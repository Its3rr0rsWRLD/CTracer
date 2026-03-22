#ifndef CTRACER_H
#define CTRACER_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Consts */
#define CT_MAX_OBJECTS     4096
#define CT_MAX_LIGHTS      128
#define CT_MAX_EMISSIVE    256
#define CT_MAX_BOUNCES     16
#define CT_MAX_SPP         64
#define CT_MAX_TRIANGLES   65536
#define CT_DEFAULT_WIDTH   256
#define CT_DEFAULT_HEIGHT  144
#define CT_DEFAULT_SPP     1
#define CT_DEFAULT_BOUNCES 4
#define CT_DEFAULT_PORT    8000
#define CT_RING_SIZE       4
#define CT_PI              3.14159265358979323846f
#define CT_EPS             1e-4f
#define CT_INF             1e30f

/* Primitive type IDs (BOX=0, SPHERE=1, CYLINDER=2, MESH=3) */
#define PRIM_BOX      0
#define PRIM_SPHERE   1
#define PRIM_CYLINDER 2
#define PRIM_MESH     3

/* UI override flags */
#define UI_OVERRIDE_EXPOSURE       (1u << 0)
#define UI_OVERRIDE_GAMMA          (1u << 1)

/* Vector types */
#if !defined(__CUDACC__) && !defined(__VECTOR_TYPES_H__)
typedef struct { float x, y, z; } float3;
static inline float3 make_float3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}
#endif

/* Camera properties */
typedef struct {
    float position[3];
    float forward[3];
    float right[3];
    float up[3];
    float fov;
} Camera;

/* Environment settings */
typedef struct {
    float sky_top[3];
    float sky_bottom[3];
    float env_intensity;
    float sun_dir[3];
    float sun_color[3];
    float sun_intensity;
    float sun_angular_radius;
    float ambient[3];
    float ambient_intensity;
    float fog_density;
    float fog_color[3];
    float exposure;
    float gamma;
    float bloom_strength;
    float bloom_threshold;
} WorldSettings;

/* Scene object properties */
typedef struct {
    float position[3];
    float size[3];
    float rotation[9];
    int prim_type;
    float color[3];
    float roughness;
    float metallic;
    float reflectance;
    float transparency;
    float ior;
    float emission[3];
    int is_player;

    /* Mesh data */
    float *mesh_vertices;
    int mesh_tri_count;
} SceneObject;

/* Light properties */
typedef struct {
    float position[3];
    float color[3];
    float brightness;
    float range;
    float direction[3];
    float angle;
    int is_spot;
} LightData;

/* Parsed scene data (CPU) */
typedef struct {
    SceneObject *objects;
    int num_objects;
    LightData *lights;
    int num_lights;
    WorldSettings world;
    Camera camera;
    int seq;
    bool has_scene;
    bool has_camera;

    SceneObject *player_objects;
    int          num_player_objects;
} ParsedScene;

/* GPU Scene Stuff */
typedef struct {
    float *pos_x, *pos_y, *pos_z;
    float *half_x, *half_y, *half_z;
    float *rot;
    int *prim_type;
    float *color_r, *color_g, *color_b;
    float *roughness, *metallic, *reflectance;
    float *transparency, *ior;
    float *emission_r, *emission_g, *emission_b;
    int num_objects;

    /* Mesh References */
    int *mesh_tri_offset;
    int *mesh_tri_count;

    /* Global triangle buffer */
    float *tri_v0_x, *tri_v0_y, *tri_v0_z;
    float *tri_v1_x, *tri_v1_y, *tri_v1_z;
    float *tri_v2_x, *tri_v2_y, *tri_v2_z;
    int total_triangles;
    int tri_capacity;

    /* NEE emissive object data */
    int *emissive_ids;
    float *emissive_centers_x, *emissive_centers_y, *emissive_centers_z;
    float *emissive_radii;
    float *emissive_emission_r, *emissive_emission_g, *emissive_emission_b;
    float *emissive_probs;
    int num_emissive;
    int  emissive_capacity;

    /* Player object tracking for partial updates */
    int    num_static_objects;   /* non-player objects (at start of arrays) */
} GPUScene;

/* GPU Lights Data */
typedef struct {
    float *pos_x, *pos_y, *pos_z;
    float *color_r, *color_g, *color_b;
    float *brightness, *range;
    float *dir_x, *dir_y, *dir_z;
    float *angle;
    int   *is_spot;
    int    count;
} GPULights;

/* Render Settings */
typedef struct {
    int   width;
    int   height;
    int   spp;
    int   max_bounces;
    bool  shadows;
    bool  reflections;
    bool  gi;
    bool  nee;
    bool  ao;
    int   ao_samples;
    float ao_radius;
    float bloom_strength;
    float bloom_threshold;
    float exposure;
    float gamma;
} RenderSettings;

/* Renderer context (from stack overflow) */
typedef struct {
    /* Temporal accumulation (EMA) */
    float   *d_history_r, *d_history_g, *d_history_b; /* persistent history [W*H] */
    float   *d_scratch_r, *d_scratch_g, *d_scratch_b; /* per-frame scratch [W*H] */
    int      temporal_frame; /* frames since last reset */

    /* Post-processing buffers */
    float   *d_hdr_r, *d_hdr_g, *d_hdr_b; /* blended HDR for post-proc [W*H] */
    float   *d_bloom_scratch; /* bloom scratch buffer */
    uint8_t *d_ldr; /* final RGBA [W*H*4] */
    uint8_t *h_pinned_output; /* pinned host output [W*H*4] */
    uint8_t *h_pinned_back; /* double-buffer back [W*H*4] */

    /* Scene and lights on GPU */
    GPUScene  gpu_scene;
    GPULights gpu_lights;
    int       scene_capacity;
    int       lights_capacity;

    /* World settings (from game via /scene) */
    WorldSettings world;
    Camera        camera;

    /* UI override tracking */
    uint32_t ui_overrides; /* bitmask of UI_OVERRIDE_* flags */

    /* Buffer dimensions */
    int      buf_width, buf_height;

    /* Render settings (UI-controlled) */
    RenderSettings settings;

    /* CUDA streams */
    void *render_stream; /* cudaStream_t */
    void *upload_stream;
    void *readback_stream;

    /* Output state */
    volatile int  output_ready; /* atomic flag */
    int           output_sample;
    int           output_seq;

    /* Stats */
    float last_render_ms;
    float last_readback_ms;
    float last_total_ms;
    float gpu_memory_used_mb;
    int   sample_count;
} RendererCtx;

typedef struct {
    ParsedScene slots[CT_RING_SIZE];
    volatile long head;
    volatile long tail;
} SceneQueue;

/* Server state */
typedef struct {
    int           port;
    volatile int  running;
    void         *mg_mgr;
    SceneQueue   *scene_queue;
    SceneQueue   *render_queue;
    RendererCtx  *renderer;
    int           last_object_count;
    int           last_light_count;
    double        last_scene_time;

    /* Pending render response */
    volatile int  response_ready;
    int           response_sample;
    int           response_seq;
    int           response_width;
    int           response_height;
} ServerState;

/* platform threading stuff */
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  typedef HANDLE ct_thread_t;
  typedef CRITICAL_SECTION ct_mutex_t;
  #define ct_mutex_init(m)    InitializeCriticalSection(m)
  #define ct_mutex_destroy(m) DeleteCriticalSection(m)
  #define ct_mutex_lock(m)    EnterCriticalSection(m)
  #define ct_mutex_unlock(m)  LeaveCriticalSection(m)
  #define ct_atomic_load(p)   InterlockedCompareExchange((volatile long*)(p), 0, 0)
  #define ct_atomic_store(p,v) InterlockedExchange((volatile long*)(p), (long)(v))
  #define ct_atomic_inc(p)    InterlockedIncrement((volatile long*)(p))
#else
  #include <pthread.h>
  typedef pthread_t ct_thread_t;
  typedef pthread_mutex_t ct_mutex_t;
  #define ct_mutex_init(m)    pthread_mutex_init(m, NULL)
  #define ct_mutex_destroy(m) pthread_mutex_destroy(m)
  #define ct_mutex_lock(m)    pthread_mutex_lock(m)
  #define ct_mutex_unlock(m)  pthread_mutex_unlock(m)
  #define ct_atomic_load(p)   __atomic_load_n(p, __ATOMIC_SEQ_CST)
  #define ct_atomic_store(p,v) __atomic_store_n(p, v, __ATOMIC_SEQ_CST)
  #define ct_atomic_inc(p)    __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST)
#endif

/* function declarations */

/* scene.c */
void scene_queue_init(SceneQueue *q);
bool scene_queue_push(SceneQueue *q, const ParsedScene *scene);
bool scene_queue_pop(SceneQueue *q, ParsedScene *out);
void parsed_scene_free(ParsedScene *ps);
bool scene_parse_request(const char *json_data, size_t json_len, ParsedScene *out);
void world_settings_defaults(WorldSettings *w);

/* server.c */
void server_init(ServerState *srv, int port, SceneQueue *scene_q, SceneQueue *render_q, RendererCtx *ctx);
void server_poll(ServerState *srv);
void server_stop(ServerState *srv);

/* renderer.c / renderer.h (CUDA host wrappers) */
void renderer_init(RendererCtx *ctx);
void renderer_upload_scene(RendererCtx *ctx, const SceneObject *objects, int num_objects,
                           const LightData *lights, int num_lights);
void renderer_upload_player_objects(RendererCtx *ctx, const SceneObject *objects, int num_objects);
void renderer_set_world(RendererCtx *ctx, const WorldSettings *world);
void renderer_set_camera(RendererCtx *ctx, const Camera *cam);
void renderer_render(RendererCtx *ctx);
void renderer_resize(RendererCtx *ctx, int width, int height);
void renderer_reset_accumulation(RendererCtx *ctx);
void renderer_shutdown(RendererCtx *ctx);

/* ui.cpp (C interface) */
int  ui_init(int width, int height, RendererCtx *ctx);
void ui_update(RendererCtx *ctx, ServerState *srv);
int  ui_should_close(void);
void ui_shutdown(void);
int  ui_get_preview_width(void);
int  ui_get_preview_height(void);

#ifdef __cplusplus
}
#endif

#endif /* CTRACER_H */
