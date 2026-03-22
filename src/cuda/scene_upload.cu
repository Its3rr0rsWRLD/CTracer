/* scene_upload.cu - upload scene data to GPU */

#include "cuda_utils.cuh"
#include "../ctracer.h"

extern "C" {

/* Tracks the triangle count from the last full scene upload, so player
   mesh uploads know where to append without overwriting static mesh data */
static int s_static_tri_count = 0;

static void ensure_device_float(float **ptr, int old_cap, int new_cap) {
    if (new_cap <= old_cap && *ptr != NULL) return;
    if (*ptr) { CHECK_CUDA(cudaFree(*ptr)); }
    CHECK_CUDA(cudaMalloc(ptr, new_cap * sizeof(float)));
}

static void ensure_device_int(int **ptr, int old_cap, int new_cap) {
    if (new_cap <= old_cap && *ptr != NULL) return;
    if (*ptr) { CHECK_CUDA(cudaFree(*ptr)); }
    CHECK_CUDA(cudaMalloc(ptr, new_cap * sizeof(int)));
}

static void upload_mesh_triangles(GPUScene *gs, const SceneObject *objects, int num_objects,
                                   cudaStream_t stream)
{
    int total_tris = 0;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i].prim_type == PRIM_MESH && objects[i].mesh_vertices && objects[i].mesh_tri_count > 0) {
            total_tris += objects[i].mesh_tri_count;
        }
    }

    if (total_tris == 0) {
        gs->total_triangles = 0;
        return;
    }

    if (total_tris > gs->tri_capacity) {
        int cap = (total_tris + 255) & ~255;  /* round up to 256 */
        if (gs->tri_v0_x) cudaFree(gs->tri_v0_x);
        if (gs->tri_v0_y) cudaFree(gs->tri_v0_y);
        if (gs->tri_v0_z) cudaFree(gs->tri_v0_z);
        if (gs->tri_v1_x) cudaFree(gs->tri_v1_x);
        if (gs->tri_v1_y) cudaFree(gs->tri_v1_y);
        if (gs->tri_v1_z) cudaFree(gs->tri_v1_z);
        if (gs->tri_v2_x) cudaFree(gs->tri_v2_x);
        if (gs->tri_v2_y) cudaFree(gs->tri_v2_y);
        if (gs->tri_v2_z) cudaFree(gs->tri_v2_z);
        CHECK_CUDA(cudaMalloc(&gs->tri_v0_x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v0_y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v0_z, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v1_x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v1_y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v1_z, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v2_x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v2_y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&gs->tri_v2_z, cap * sizeof(float)));
        gs->tri_capacity = cap;
    }

    float *h_v0x, *h_v0y, *h_v0z, *h_v1x, *h_v1y, *h_v1z, *h_v2x, *h_v2y, *h_v2z;
    CHECK_CUDA(cudaHostAlloc(&h_v0x, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v0y, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v0z, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1x, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1y, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1z, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2x, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2y, total_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2z, total_tris * sizeof(float), cudaHostAllocDefault));

    int *h_mto, *h_mtc;
    CHECK_CUDA(cudaHostAlloc(&h_mto, num_objects * sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_mtc, num_objects * sizeof(int), cudaHostAllocDefault));

    /* Transpose AoS triangle data to SoA */
    int tri_idx = 0;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i].prim_type == PRIM_MESH && objects[i].mesh_vertices && objects[i].mesh_tri_count > 0) {
            h_mto[i] = tri_idx;
            h_mtc[i] = objects[i].mesh_tri_count;
            const float *verts = objects[i].mesh_vertices;
            for (int t = 0; t < objects[i].mesh_tri_count; t++) {
                int base = t * 9;  /* 3 vertices * 3 components = 9 floats per tri */
                h_v0x[tri_idx] = verts[base + 0];
                h_v0y[tri_idx] = verts[base + 1];
                h_v0z[tri_idx] = verts[base + 2];
                h_v1x[tri_idx] = verts[base + 3];
                h_v1y[tri_idx] = verts[base + 4];
                h_v1z[tri_idx] = verts[base + 5];
                h_v2x[tri_idx] = verts[base + 6];
                h_v2y[tri_idx] = verts[base + 7];
                h_v2z[tri_idx] = verts[base + 8];
                tri_idx++;
            }
        } else {
            h_mto[i] = 0;
            h_mtc[i] = 0;
        }
    }

    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_x, h_v0x, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_y, h_v0y, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_z, h_v0z, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_x, h_v1x, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_y, h_v1y, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_z, h_v1z, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_x, h_v2x, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_y, h_v2y, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_z, h_v2z, total_tris * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaMemcpyAsync(gs->mesh_tri_offset, h_mto, num_objects * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->mesh_tri_count, h_mtc, num_objects * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    gs->total_triangles = total_tris;

    /* Free staging */
    cudaFreeHost(h_v0x); cudaFreeHost(h_v0y); cudaFreeHost(h_v0z);
    cudaFreeHost(h_v1x); cudaFreeHost(h_v1y); cudaFreeHost(h_v1z);
    cudaFreeHost(h_v2x); cudaFreeHost(h_v2y); cudaFreeHost(h_v2z);
    cudaFreeHost(h_mto); cudaFreeHost(h_mtc);
}

void gpu_scene_alloc(GPUScene *gs, int num_objects, int *capacity) {
    if (num_objects <= *capacity && gs->pos_x != NULL) return;
    int cap = (num_objects + 63) & ~63;  /* round up to 64 */

    ensure_device_float(&gs->pos_x, *capacity, cap);
    ensure_device_float(&gs->pos_y, *capacity, cap);
    ensure_device_float(&gs->pos_z, *capacity, cap);
    ensure_device_float(&gs->half_x, *capacity, cap);
    ensure_device_float(&gs->half_y, *capacity, cap);
    ensure_device_float(&gs->half_z, *capacity, cap);
    ensure_device_float(&gs->rot, *capacity * 9, cap * 9);
    ensure_device_int(&gs->prim_type, *capacity, cap);
    ensure_device_float(&gs->color_r, *capacity, cap);
    ensure_device_float(&gs->color_g, *capacity, cap);
    ensure_device_float(&gs->color_b, *capacity, cap);
    ensure_device_float(&gs->roughness, *capacity, cap);
    ensure_device_float(&gs->metallic, *capacity, cap);
    ensure_device_float(&gs->reflectance, *capacity, cap);
    ensure_device_float(&gs->transparency, *capacity, cap);
    ensure_device_float(&gs->ior, *capacity, cap);
    ensure_device_float(&gs->emission_r, *capacity, cap);
    ensure_device_float(&gs->emission_g, *capacity, cap);
    ensure_device_float(&gs->emission_b, *capacity, cap);

    /* Mesh per-object arrays */
    ensure_device_int(&gs->mesh_tri_offset, *capacity, cap);
    ensure_device_int(&gs->mesh_tri_count, *capacity, cap);

    *capacity = cap;
}

void gpu_scene_upload(GPUScene *gs, const SceneObject *objects, int num_objects,
                      void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (num_objects == 0) {
        gs->num_objects = 0;
        return;
    }

    int n = num_objects;

    float *h_px, *h_py, *h_pz;
    float *h_hx, *h_hy, *h_hz;
    float *h_rot;
    int   *h_pt;
    float *h_cr, *h_cg, *h_cb;
    float *h_rough, *h_metal, *h_refl;
    float *h_transp, *h_ior;
    float *h_er, *h_eg, *h_eb;

    CHECK_CUDA(cudaHostAlloc(&h_px, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_py, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_pz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hx, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hy, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_rot, n * 9 * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_pt, n * sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cr, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cg, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cb, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_rough, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_metal, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_refl, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_transp, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_ior, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_er, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_eg, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_eb, n * sizeof(float), cudaHostAllocDefault));

    /* AoS → SoA transpose */
    for (int i = 0; i < n; i++) {
        const SceneObject *o = &objects[i];
        h_px[i] = o->position[0];
        h_py[i] = o->position[1];
        h_pz[i] = o->position[2];
        h_hx[i] = o->size[0] * 0.5f;
        h_hy[i] = o->size[1] * 0.5f;
        h_hz[i] = o->size[2] * 0.5f;
        for (int j = 0; j < 9; j++) h_rot[i * 9 + j] = o->rotation[j];
        h_pt[i] = o->prim_type;
        h_cr[i] = fminf(fmaxf(o->color[0], 0.0f), 1.0f);
        h_cg[i] = fminf(fmaxf(o->color[1], 0.0f), 1.0f);
        h_cb[i] = fminf(fmaxf(o->color[2], 0.0f), 1.0f);
        h_rough[i] = fminf(fmaxf(o->roughness, 0.02f), 1.0f);
        h_metal[i] = fminf(fmaxf(o->metallic, 0.0f), 1.0f);
        h_refl[i] = fminf(fmaxf(o->reflectance, 0.0f), 1.0f);
        h_transp[i] = fminf(fmaxf(o->transparency, 0.0f), 1.0f);
        h_ior[i] = fminf(fmaxf(o->ior, 1.0f), 3.0f);
        h_er[i] = fminf(fmaxf(o->emission[0], 0.0f), 100.0f);
        h_eg[i] = fminf(fmaxf(o->emission[1], 0.0f), 100.0f);
        h_eb[i] = fminf(fmaxf(o->emission[2], 0.0f), 100.0f);
    }

    CHECK_CUDA(cudaMemcpyAsync(gs->pos_x, h_px, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->pos_y, h_py, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->pos_z, h_pz, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_x, h_hx, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_y, h_hy, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_z, h_hz, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->rot, h_rot, n * 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->prim_type, h_pt, n * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_r, h_cr, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_g, h_cg, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_b, h_cb, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->roughness, h_rough, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->metallic, h_metal, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->reflectance, h_refl, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->transparency, h_transp, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->ior, h_ior, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_r, h_er, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_g, h_eg, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_b, h_eb, n * sizeof(float), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    upload_mesh_triangles(gs, objects, num_objects, stream);

    /* Record the static triangle count so player uploads know where to append */
    s_static_tri_count = gs->total_triangles;

    /* build emissive data for NEE */
    int num_emissive = 0;
    for (int i = 0; i < n; i++) {
        float em = fmaxf(h_er[i], fmaxf(h_eg[i], h_eb[i]));
        if (em > 0.0f) num_emissive++;
    }

    if (num_emissive > 0) {
        if (num_emissive > gs->emissive_capacity) {
            int ecap = (num_emissive + 31) & ~31;
            if (gs->emissive_ids) cudaFree(gs->emissive_ids);
            if (gs->emissive_centers_x) cudaFree(gs->emissive_centers_x);
            if (gs->emissive_centers_y) cudaFree(gs->emissive_centers_y);
            if (gs->emissive_centers_z) cudaFree(gs->emissive_centers_z);
            if (gs->emissive_radii) cudaFree(gs->emissive_radii);
            if (gs->emissive_emission_r) cudaFree(gs->emissive_emission_r);
            if (gs->emissive_emission_g) cudaFree(gs->emissive_emission_g);
            if (gs->emissive_emission_b) cudaFree(gs->emissive_emission_b);
            if (gs->emissive_probs) cudaFree(gs->emissive_probs);
            CHECK_CUDA(cudaMalloc(&gs->emissive_ids, ecap * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_centers_x, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_centers_y, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_centers_z, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_radii, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_emission_r, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_emission_g, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_emission_b, ecap * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gs->emissive_probs, ecap * sizeof(float)));
            gs->emissive_capacity = ecap;
        }

        int *he_ids;
        float *he_cx, *he_cy, *he_cz, *he_r, *he_er, *he_eg, *he_eb, *he_probs;
        CHECK_CUDA(cudaHostAlloc(&he_ids, num_emissive * sizeof(int), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_cx, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_cy, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_cz, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_r, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_er, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_eg, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_eb, num_emissive * sizeof(float), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&he_probs, num_emissive * sizeof(float), cudaHostAllocDefault));

        int ei = 0;
        float total_weight = 0.0f;
        for (int i = 0; i < n; i++) {
            float em = fmaxf(h_er[i], fmaxf(h_eg[i], h_eb[i]));
            if (em > 0.0f) {
                he_ids[ei] = i;
                he_cx[ei] = h_px[i];
                he_cy[ei] = h_py[i];
                he_cz[ei] = h_pz[i];
                float rad = fmaxf(h_hx[i], fmaxf(h_hy[i], h_hz[i]));
                he_r[ei] = fmaxf(rad, 0.01f);
                he_er[ei] = h_er[i];
                he_eg[ei] = h_eg[i];
                he_eb[ei] = h_eb[i];
                float lum = 0.2126f * h_er[i] + 0.7152f * h_eg[i] + 0.0722f * h_eb[i];
                float w = fmaxf(lum * 4.0f * CT_PI * rad * rad, 1e-8f);
                he_probs[ei] = w;
                total_weight += w;
                ei++;
            }
        }
        /* Normalize probabilities */
        for (int i = 0; i < num_emissive; i++) {
            he_probs[i] /= total_weight;
        }

        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_ids, he_ids, num_emissive * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_centers_x, he_cx, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_centers_y, he_cy, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_centers_z, he_cz, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_radii, he_r, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_emission_r, he_er, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_emission_g, he_eg, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_emission_b, he_eb, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(gs->emissive_probs, he_probs, num_emissive * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cudaFreeHost(he_ids); cudaFreeHost(he_cx); cudaFreeHost(he_cy); cudaFreeHost(he_cz);
        cudaFreeHost(he_r); cudaFreeHost(he_er); cudaFreeHost(he_eg); cudaFreeHost(he_eb);
        cudaFreeHost(he_probs);
    }
    gs->num_emissive = num_emissive;
    gs->num_objects = n;

    /* Free staging buffers */
    cudaFreeHost(h_px); cudaFreeHost(h_py); cudaFreeHost(h_pz);
    cudaFreeHost(h_hx); cudaFreeHost(h_hy); cudaFreeHost(h_hz);
    cudaFreeHost(h_rot); cudaFreeHost(h_pt);
    cudaFreeHost(h_cr); cudaFreeHost(h_cg); cudaFreeHost(h_cb);
    cudaFreeHost(h_rough); cudaFreeHost(h_metal); cudaFreeHost(h_refl);
    cudaFreeHost(h_transp); cudaFreeHost(h_ior);
    cudaFreeHost(h_er); cudaFreeHost(h_eg); cudaFreeHost(h_eb);
}

static void upload_player_mesh_triangles(GPUScene *gs, const SceneObject *objects,
                                          int num_objects, int obj_offset,
                                          cudaStream_t stream)
{
    int player_tris = 0;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i].prim_type == PRIM_MESH && objects[i].mesh_vertices && objects[i].mesh_tri_count > 0) {
            player_tris += objects[i].mesh_tri_count;
        }
    }

    /* Figure out where static triangles end (preserve them) */
    /* We track the static triangle count: on first full upload it's set in gs->total_triangles.
       Player triangles are appended after. We need to keep a "static triangle count" separate
       from total_triangles. We'll use the existing static triangle data as base. */
    int static_tris = 0;
    /* Scan: the static scene's total_triangles was set during full upload.
       But on repeated player uploads, total_triangles includes previous player tris.
       We need to recompute: static tris = total_triangles at last full upload.
       Since we don't track that separately, we'll just re-upload ALL player mesh tris
       starting right after static mesh data. We can compute static mesh tris from the
       mesh_tri_offset/count of static objects, but those are on GPU. Instead, let's
       track it: the static mesh tri count was whatever total_triangles was set to
       by upload_mesh_triangles() during the last full upload. We'll store it. */
    /* For now: rebuild the entire mesh tri buffer with static tris preserved.
       The simplest correct approach: just re-upload mesh data for ALL objects
       (static + player) but that requires having static object data on host.
       We don't have that here. So instead, append player tris after the current
       static tri data. We need to know where static tris end. */
    /* Solution: use a player_tri_base field. On full upload, set it to total_triangles.
       On player upload, append starting at player_tri_base. */
    /* Since we don't have player_tri_base yet, approximate: the first call to this
       function after a full scene upload can infer static_tris from the current
       total_triangles (which was just set by upload_mesh_triangles for the full scene). */
    /* Actually, simpler: we re-upload ALL mesh data (both static and player).
       But we don't have static objects here. So let's just set a convention:
       player_tri_base = total_triangles as set by the most recent full scene upload.
       We store this in a static variable since GPUScene doesn't have the field. */
    /* Best approach: add a field. But to minimize changes, use gs->total_triangles
       BEFORE we modify it — it currently holds the count from the last upload
       (which could be a previous player upload or the full upload). We need to know
       which. Let's use num_static_objects to figure it out: if this is the first
       player upload after a full scene upload, total_triangles = static mesh tris. */

    /* SIMPLE FIX: Just track player_tri_base alongside the static object count.
       On full upload, we set total_triangles = static mesh tris.
       On player upload, we append at that base. But on repeated player uploads,
       total_triangles grows. We need the original base. */

    /* The cleanest solution without adding struct fields: on each player upload,
       recompute from scratch. We know obj_offset = num_static_objects, so we can
       re-derive the static triangle count. But the static mesh data is on GPU only.
       So just keep a local static variable tracking the base. */
    int tri_base = s_static_tri_count;
    int total_tris_needed = tri_base + player_tris;

    if (total_tris_needed > gs->tri_capacity) {
        /* Need to grow: allocate new buffers, copy old static data, then add player */
        int cap = (total_tris_needed + 255) & ~255;
        float *new_v0x, *new_v0y, *new_v0z, *new_v1x, *new_v1y, *new_v1z, *new_v2x, *new_v2y, *new_v2z;
        CHECK_CUDA(cudaMalloc(&new_v0x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v0y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v0z, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v1x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v1y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v1z, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v2x, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v2y, cap * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&new_v2z, cap * sizeof(float)));

        /* Copy existing static tri data */
        if (tri_base > 0 && gs->tri_v0_x) {
            CHECK_CUDA(cudaMemcpyAsync(new_v0x, gs->tri_v0_x, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v0y, gs->tri_v0_y, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v0z, gs->tri_v0_z, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v1x, gs->tri_v1_x, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v1y, gs->tri_v1_y, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v1z, gs->tri_v1_z, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v2x, gs->tri_v2_x, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v2y, gs->tri_v2_y, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_v2z, gs->tri_v2_z, tri_base * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        /* Free old and swap */
        if (gs->tri_v0_x) cudaFree(gs->tri_v0_x);
        if (gs->tri_v0_y) cudaFree(gs->tri_v0_y);
        if (gs->tri_v0_z) cudaFree(gs->tri_v0_z);
        if (gs->tri_v1_x) cudaFree(gs->tri_v1_x);
        if (gs->tri_v1_y) cudaFree(gs->tri_v1_y);
        if (gs->tri_v1_z) cudaFree(gs->tri_v1_z);
        if (gs->tri_v2_x) cudaFree(gs->tri_v2_x);
        if (gs->tri_v2_y) cudaFree(gs->tri_v2_y);
        if (gs->tri_v2_z) cudaFree(gs->tri_v2_z);
        gs->tri_v0_x = new_v0x; gs->tri_v0_y = new_v0y; gs->tri_v0_z = new_v0z;
        gs->tri_v1_x = new_v1x; gs->tri_v1_y = new_v1y; gs->tri_v1_z = new_v1z;
        gs->tri_v2_x = new_v2x; gs->tri_v2_y = new_v2y; gs->tri_v2_z = new_v2z;
        gs->tri_capacity = cap;
    }

    if (player_tris == 0) {
        /* No player meshes — just zero out player mesh_tri_count entries */
        int *h_mtc;
        CHECK_CUDA(cudaHostAlloc(&h_mtc, num_objects * sizeof(int), cudaHostAllocDefault));
        for (int i = 0; i < num_objects; i++) h_mtc[i] = 0;
        CHECK_CUDA(cudaMemcpyAsync(gs->mesh_tri_count + obj_offset, h_mtc,
                                    num_objects * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        cudaFreeHost(h_mtc);
        gs->total_triangles = tri_base;
        return;
    }

    /* Build SoA triangle data + per-object offset/count for player objects */
    float *h_v0x, *h_v0y, *h_v0z, *h_v1x, *h_v1y, *h_v1z, *h_v2x, *h_v2y, *h_v2z;
    int *h_mto, *h_mtc;
    CHECK_CUDA(cudaHostAlloc(&h_v0x, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v0y, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v0z, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1x, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1y, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v1z, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2x, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2y, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_v2z, player_tris * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_mto, num_objects * sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_mtc, num_objects * sizeof(int), cudaHostAllocDefault));

    int tri_idx = 0;
    for (int i = 0; i < num_objects; i++) {
        if (objects[i].prim_type == PRIM_MESH && objects[i].mesh_vertices && objects[i].mesh_tri_count > 0) {
            h_mto[i] = tri_base + tri_idx;  /* absolute offset in global tri buffer */
            h_mtc[i] = objects[i].mesh_tri_count;
            const float *verts = objects[i].mesh_vertices;
            for (int t = 0; t < objects[i].mesh_tri_count; t++) {
                int base = t * 9;
                h_v0x[tri_idx] = verts[base + 0];
                h_v0y[tri_idx] = verts[base + 1];
                h_v0z[tri_idx] = verts[base + 2];
                h_v1x[tri_idx] = verts[base + 3];
                h_v1y[tri_idx] = verts[base + 4];
                h_v1z[tri_idx] = verts[base + 5];
                h_v2x[tri_idx] = verts[base + 6];
                h_v2y[tri_idx] = verts[base + 7];
                h_v2z[tri_idx] = verts[base + 8];
                tri_idx++;
            }
        } else {
            h_mto[i] = 0;
            h_mtc[i] = 0;
        }
    }

    /* Upload player triangle data at tri_base offset */
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_x + tri_base, h_v0x, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_y + tri_base, h_v0y, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v0_z + tri_base, h_v0z, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_x + tri_base, h_v1x, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_y + tri_base, h_v1y, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v1_z + tri_base, h_v1z, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_x + tri_base, h_v2x, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_y + tri_base, h_v2y, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->tri_v2_z + tri_base, h_v2z, player_tris * sizeof(float), cudaMemcpyHostToDevice, stream));

    /* Upload per-object mesh offset/count for player objects (at obj_offset in GPU arrays) */
    CHECK_CUDA(cudaMemcpyAsync(gs->mesh_tri_offset + obj_offset, h_mto, num_objects * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->mesh_tri_count + obj_offset, h_mtc, num_objects * sizeof(int), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    gs->total_triangles = tri_base + player_tris;

    cudaFreeHost(h_v0x); cudaFreeHost(h_v0y); cudaFreeHost(h_v0z);
    cudaFreeHost(h_v1x); cudaFreeHost(h_v1y); cudaFreeHost(h_v1z);
    cudaFreeHost(h_v2x); cudaFreeHost(h_v2y); cudaFreeHost(h_v2z);
    cudaFreeHost(h_mto); cudaFreeHost(h_mtc);
}

void gpu_scene_upload_player(GPUScene *gs, const SceneObject *objects, int num_objects,
                              void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (num_objects <= 0 || !objects) return;

    int offset = gs->num_static_objects;
    int total_needed = offset + num_objects;
    int n = num_objects;

    /* Allocate pinned staging */
    float *h_px, *h_py, *h_pz;
    float *h_hx, *h_hy, *h_hz;
    float *h_rot;
    int   *h_pt;
    float *h_cr, *h_cg, *h_cb;
    float *h_rough, *h_metal, *h_refl;
    float *h_transp, *h_ior;
    float *h_er, *h_eg, *h_eb;

    CHECK_CUDA(cudaHostAlloc(&h_px, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_py, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_pz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hx, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hy, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_hz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_rot, n * 9 * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_pt, n * sizeof(int), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cr, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cg, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cb, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_rough, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_metal, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_refl, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_transp, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_ior, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_er, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_eg, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_eb, n * sizeof(float), cudaHostAllocDefault));

    /* AoS → SoA transpose */
    for (int i = 0; i < n; i++) {
        const SceneObject *o = &objects[i];
        h_px[i] = o->position[0];
        h_py[i] = o->position[1];
        h_pz[i] = o->position[2];
        h_hx[i] = o->size[0] * 0.5f;
        h_hy[i] = o->size[1] * 0.5f;
        h_hz[i] = o->size[2] * 0.5f;
        for (int j = 0; j < 9; j++) h_rot[i * 9 + j] = o->rotation[j];
        h_pt[i] = o->prim_type;
        h_cr[i] = fminf(fmaxf(o->color[0], 0.0f), 1.0f);
        h_cg[i] = fminf(fmaxf(o->color[1], 0.0f), 1.0f);
        h_cb[i] = fminf(fmaxf(o->color[2], 0.0f), 1.0f);
        h_rough[i] = fminf(fmaxf(o->roughness, 0.02f), 1.0f);
        h_metal[i] = fminf(fmaxf(o->metallic, 0.0f), 1.0f);
        h_refl[i] = fminf(fmaxf(o->reflectance, 0.0f), 1.0f);
        h_transp[i] = fminf(fmaxf(o->transparency, 0.0f), 1.0f);
        h_ior[i] = fminf(fmaxf(o->ior, 1.0f), 3.0f);
        h_er[i] = fminf(fmaxf(o->emission[0], 0.0f), 100.0f);
        h_eg[i] = fminf(fmaxf(o->emission[1], 0.0f), 100.0f);
        h_eb[i] = fminf(fmaxf(o->emission[2], 0.0f), 100.0f);
    }

    /* Upload at offset (after static objects) */
    size_t n_bytes = n * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(gs->pos_x + offset, h_px, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->pos_y + offset, h_py, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->pos_z + offset, h_pz, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_x + offset, h_hx, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_y + offset, h_hy, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->half_z + offset, h_hz, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->rot + offset * 9, h_rot, n * 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->prim_type + offset, h_pt, n * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_r + offset, h_cr, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_g + offset, h_cg, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->color_b + offset, h_cb, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->roughness + offset, h_rough, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->metallic + offset, h_metal, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->reflectance + offset, h_refl, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->transparency + offset, h_transp, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->ior + offset, h_ior, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_r + offset, h_er, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_g + offset, h_eg, n_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gs->emission_b + offset, h_eb, n_bytes, cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    /* Update total object count */
    gs->num_objects = total_needed;

    /* Upload mesh triangle data for player PRIM_MESH objects */
    upload_player_mesh_triangles(gs, objects, num_objects, offset, stream);

    /* Free staging */
    cudaFreeHost(h_px); cudaFreeHost(h_py); cudaFreeHost(h_pz);
    cudaFreeHost(h_hx); cudaFreeHost(h_hy); cudaFreeHost(h_hz);
    cudaFreeHost(h_rot); cudaFreeHost(h_pt);
    cudaFreeHost(h_cr); cudaFreeHost(h_cg); cudaFreeHost(h_cb);
    cudaFreeHost(h_rough); cudaFreeHost(h_metal); cudaFreeHost(h_refl);
    cudaFreeHost(h_transp); cudaFreeHost(h_ior);
    cudaFreeHost(h_er); cudaFreeHost(h_eg); cudaFreeHost(h_eb);
}


void gpu_lights_alloc(GPULights *gl, int count, int *capacity) {
    if (count <= *capacity && gl->pos_x != NULL) return;
    int cap = (count + 31) & ~31;

    ensure_device_float(&gl->pos_x, *capacity, cap);
    ensure_device_float(&gl->pos_y, *capacity, cap);
    ensure_device_float(&gl->pos_z, *capacity, cap);
    ensure_device_float(&gl->color_r, *capacity, cap);
    ensure_device_float(&gl->color_g, *capacity, cap);
    ensure_device_float(&gl->color_b, *capacity, cap);
    ensure_device_float(&gl->brightness, *capacity, cap);
    ensure_device_float(&gl->range, *capacity, cap);
    ensure_device_float(&gl->dir_x, *capacity, cap);
    ensure_device_float(&gl->dir_y, *capacity, cap);
    ensure_device_float(&gl->dir_z, *capacity, cap);
    ensure_device_float(&gl->angle, *capacity, cap);
    ensure_device_int(&gl->is_spot, *capacity, cap);

    *capacity = cap;
}

void gpu_lights_upload(GPULights *gl, const LightData *lights, int count,
                       void *stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (count == 0) {
        gl->count = 0;
        return;
    }

    int n = count;
    float *h_px, *h_py, *h_pz, *h_cr, *h_cg, *h_cb;
    float *h_bright, *h_range, *h_dx, *h_dy, *h_dz, *h_angle;
    int   *h_spot;

    CHECK_CUDA(cudaHostAlloc(&h_px, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_py, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_pz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cr, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cg, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_cb, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_bright, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_range, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_dx, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_dy, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_dz, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_angle, n * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_spot, n * sizeof(int), cudaHostAllocDefault));

    for (int i = 0; i < n; i++) {
        const LightData *l = &lights[i];
        h_px[i] = l->position[0];
        h_py[i] = l->position[1];
        h_pz[i] = l->position[2];
        h_cr[i] = l->color[0];
        h_cg[i] = l->color[1];
        h_cb[i] = l->color[2];
        h_bright[i] = l->brightness;
        h_range[i] = l->range;
        /* Normalize direction */
        float dx = l->direction[0], dy = l->direction[1], dz = l->direction[2];
        float len = sqrtf(dx*dx + dy*dy + dz*dz);
        if (len > 1e-6f) { dx /= len; dy /= len; dz /= len; }
        h_dx[i] = dx; h_dy[i] = dy; h_dz[i] = dz;
        h_angle[i] = l->angle * (CT_PI / 180.0f);  /* degrees → radians */
        h_spot[i] = l->is_spot;
    }

    CHECK_CUDA(cudaMemcpyAsync(gl->pos_x, h_px, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->pos_y, h_py, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->pos_z, h_pz, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->color_r, h_cr, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->color_g, h_cg, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->color_b, h_cb, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->brightness, h_bright, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->range, h_range, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->dir_x, h_dx, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->dir_y, h_dy, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->dir_z, h_dz, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->angle, h_angle, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(gl->is_spot, h_spot, n * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    gl->count = n;

    cudaFreeHost(h_px); cudaFreeHost(h_py); cudaFreeHost(h_pz);
    cudaFreeHost(h_cr); cudaFreeHost(h_cg); cudaFreeHost(h_cb);
    cudaFreeHost(h_bright); cudaFreeHost(h_range);
    cudaFreeHost(h_dx); cudaFreeHost(h_dy); cudaFreeHost(h_dz);
    cudaFreeHost(h_angle); cudaFreeHost(h_spot);
}

void gpu_scene_free(GPUScene *gs) {
    if (gs->pos_x) cudaFree(gs->pos_x);
    if (gs->pos_y) cudaFree(gs->pos_y);
    if (gs->pos_z) cudaFree(gs->pos_z);
    if (gs->half_x) cudaFree(gs->half_x);
    if (gs->half_y) cudaFree(gs->half_y);
    if (gs->half_z) cudaFree(gs->half_z);
    if (gs->rot) cudaFree(gs->rot);
    if (gs->prim_type) cudaFree(gs->prim_type);
    if (gs->color_r) cudaFree(gs->color_r);
    if (gs->color_g) cudaFree(gs->color_g);
    if (gs->color_b) cudaFree(gs->color_b);
    if (gs->roughness) cudaFree(gs->roughness);
    if (gs->metallic) cudaFree(gs->metallic);
    if (gs->reflectance) cudaFree(gs->reflectance);
    if (gs->transparency) cudaFree(gs->transparency);
    if (gs->ior) cudaFree(gs->ior);
    if (gs->emission_r) cudaFree(gs->emission_r);
    if (gs->emission_g) cudaFree(gs->emission_g);
    if (gs->emission_b) cudaFree(gs->emission_b);
    /* Mesh buffers */
    if (gs->mesh_tri_offset) cudaFree(gs->mesh_tri_offset);
    if (gs->mesh_tri_count) cudaFree(gs->mesh_tri_count);
    if (gs->tri_v0_x) cudaFree(gs->tri_v0_x);
    if (gs->tri_v0_y) cudaFree(gs->tri_v0_y);
    if (gs->tri_v0_z) cudaFree(gs->tri_v0_z);
    if (gs->tri_v1_x) cudaFree(gs->tri_v1_x);
    if (gs->tri_v1_y) cudaFree(gs->tri_v1_y);
    if (gs->tri_v1_z) cudaFree(gs->tri_v1_z);
    if (gs->tri_v2_x) cudaFree(gs->tri_v2_x);
    if (gs->tri_v2_y) cudaFree(gs->tri_v2_y);
    if (gs->tri_v2_z) cudaFree(gs->tri_v2_z);
    /* Emissive */
    if (gs->emissive_ids) cudaFree(gs->emissive_ids);
    if (gs->emissive_centers_x) cudaFree(gs->emissive_centers_x);
    if (gs->emissive_centers_y) cudaFree(gs->emissive_centers_y);
    if (gs->emissive_centers_z) cudaFree(gs->emissive_centers_z);
    if (gs->emissive_radii) cudaFree(gs->emissive_radii);
    if (gs->emissive_emission_r) cudaFree(gs->emissive_emission_r);
    if (gs->emissive_emission_g) cudaFree(gs->emissive_emission_g);
    if (gs->emissive_emission_b) cudaFree(gs->emissive_emission_b);
    if (gs->emissive_probs) cudaFree(gs->emissive_probs);
    memset(gs, 0, sizeof(*gs));
}

void gpu_lights_free(GPULights *gl) {
    if (gl->pos_x) cudaFree(gl->pos_x);
    if (gl->pos_y) cudaFree(gl->pos_y);
    if (gl->pos_z) cudaFree(gl->pos_z);
    if (gl->color_r) cudaFree(gl->color_r);
    if (gl->color_g) cudaFree(gl->color_g);
    if (gl->color_b) cudaFree(gl->color_b);
    if (gl->brightness) cudaFree(gl->brightness);
    if (gl->range) cudaFree(gl->range);
    if (gl->dir_x) cudaFree(gl->dir_x);
    if (gl->dir_y) cudaFree(gl->dir_y);
    if (gl->dir_z) cudaFree(gl->dir_z);
    if (gl->angle) cudaFree(gl->angle);
    if (gl->is_spot) cudaFree(gl->is_spot);
    memset(gl, 0, sizeof(*gl));
}

void gpu_print_info(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.0f MB, SM %d.%d)\n",
           prop.name, prop.totalGlobalMem / 1048576.0,
           prop.major, prop.minor);
}

} /* extern "C" */
