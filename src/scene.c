#include "scene.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <yyjson.h>

/* Default world settings */
void world_settings_defaults(WorldSettings *w) {
    w->sky_top[0] = 0.4f; w->sky_top[1] = 0.6f; w->sky_top[2] = 1.0f;
    w->sky_bottom[0] = 0.9f; w->sky_bottom[1] = 0.9f; w->sky_bottom[2] = 0.9f;
    w->env_intensity = 0.8f;
    w->sun_dir[0] = 0.5f; w->sun_dir[1] = 0.8f; w->sun_dir[2] = 0.3f;
    w->sun_color[0] = 1.0f; w->sun_color[1] = 0.95f; w->sun_color[2] = 0.8f;
    w->sun_intensity = 1.0f;
    w->sun_angular_radius = 0.02f;
    w->ambient[0] = 0.02f; w->ambient[1] = 0.02f; w->ambient[2] = 0.03f;
    w->ambient_intensity = 0.4f;
    w->fog_density = 0.0f;
    w->fog_color[0] = 0.7f; w->fog_color[1] = 0.7f; w->fog_color[2] = 0.7f;
    w->exposure = 0.6f;
    w->gamma = 2.2f;
    w->bloom_strength = 0.0f;
    w->bloom_threshold = 2.0f;
}

/* Initialize the scene queue */
void scene_queue_init(SceneQueue *q) {
    memset(q, 0, sizeof(SceneQueue));
}

bool scene_queue_push(SceneQueue *q, const ParsedScene *scene) {
    long head = ct_atomic_load(&q->head);
    long tail = ct_atomic_load(&q->tail);
    long next = (head + 1) % CT_RING_SIZE;
    if (next == tail) return false;

    ParsedScene *slot = &q->slots[head];
    parsed_scene_free(slot);

    *slot = *scene;

    /* Deep copy object and light arrays */
    if (scene->num_objects > 0 && scene->objects) {
        slot->objects = (SceneObject *)malloc(scene->num_objects * sizeof(SceneObject));
        memcpy(slot->objects, scene->objects, scene->num_objects * sizeof(SceneObject));
        /* Deep copy mesh vertex data */
        for (int i = 0; i < scene->num_objects; i++) {
            if (scene->objects[i].mesh_vertices && scene->objects[i].mesh_tri_count > 0) {
                int vsize = scene->objects[i].mesh_tri_count * 9 * sizeof(float);
                slot->objects[i].mesh_vertices = (float *)malloc(vsize);
                memcpy(slot->objects[i].mesh_vertices, scene->objects[i].mesh_vertices, vsize);
            } else {
                slot->objects[i].mesh_vertices = NULL;
            }
        }
    }
    if (scene->num_lights > 0 && scene->lights) {
        slot->lights = (LightData *)malloc(scene->num_lights * sizeof(LightData));
        memcpy(slot->lights, scene->lights, scene->num_lights * sizeof(LightData));
    }

    /* Deep copy player objects */
    if (scene->num_player_objects > 0 && scene->player_objects) {
        slot->player_objects = (SceneObject *)malloc(scene->num_player_objects * sizeof(SceneObject));
        memcpy(slot->player_objects, scene->player_objects, scene->num_player_objects * sizeof(SceneObject));
        for (int i = 0; i < scene->num_player_objects; i++) {
            if (scene->player_objects[i].mesh_vertices && scene->player_objects[i].mesh_tri_count > 0) {
                int vsize = scene->player_objects[i].mesh_tri_count * 9 * sizeof(float);
                slot->player_objects[i].mesh_vertices = (float *)malloc(vsize);
                memcpy(slot->player_objects[i].mesh_vertices, scene->player_objects[i].mesh_vertices, vsize);
            } else {
                slot->player_objects[i].mesh_vertices = NULL;
            }
        }
    }

    ct_atomic_store(&q->head, next);
    return true;
}

bool scene_queue_pop(SceneQueue *q, ParsedScene *out) {
    long head = ct_atomic_load(&q->head);
    long tail = ct_atomic_load(&q->tail);
    if (head == tail) return false;

    *out = q->slots[tail];
    memset(&q->slots[tail], 0, sizeof(ParsedScene));

    ct_atomic_store(&q->tail, (tail + 1) % CT_RING_SIZE);
    return true;
}

void parsed_scene_free(ParsedScene *ps) {
    if (ps->objects) {
        for (int i = 0; i < ps->num_objects; i++) {
            if (ps->objects[i].mesh_vertices) {
                free(ps->objects[i].mesh_vertices);
            }
        }
        free(ps->objects);
        ps->objects = NULL;
    }
    if (ps->lights)  { free(ps->lights);  ps->lights = NULL; }
    if (ps->player_objects) {
        for (int i = 0; i < ps->num_player_objects; i++) {
            if (ps->player_objects[i].mesh_vertices) {
                free(ps->player_objects[i].mesh_vertices);
            }
        }
        free(ps->player_objects);
        ps->player_objects = NULL;
    }
    ps->num_objects = 0;
    ps->num_lights = 0;
    ps->num_player_objects = 0;
}

/* JSON Parsing */
static float json_float(yyjson_val *obj, const char *key, float def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) return def;
    if (yyjson_is_real(v)) return (float)yyjson_get_real(v);
    if (yyjson_is_int(v))  return (float)yyjson_get_int(v);
    return def;
}

static void json_float3(yyjson_val *obj, const char *key, float *out, float dx, float dy, float dz) {
    out[0] = dx; out[1] = dy; out[2] = dz;
    yyjson_val *arr = yyjson_obj_get(obj, key);
    if (!arr || !yyjson_is_arr(arr)) return;
    size_t len = yyjson_arr_size(arr);
    if (len >= 1) {
        yyjson_val *e = yyjson_arr_get_first(arr);
        out[0] = (float)(yyjson_is_real(e) ? yyjson_get_real(e) : yyjson_get_int(e));
        if (len >= 2) {
            e = yyjson_arr_get(arr, 1);
            out[1] = (float)(yyjson_is_real(e) ? yyjson_get_real(e) : yyjson_get_int(e));
        }
        if (len >= 3) {
            e = yyjson_arr_get(arr, 2);
            out[2] = (float)(yyjson_is_real(e) ? yyjson_get_real(e) : yyjson_get_int(e));
        }
    }
}

static const char *json_str(yyjson_val *obj, const char *key, const char *def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v || !yyjson_is_str(v)) return def;
    return yyjson_get_str(v);
}

static int json_int(yyjson_val *obj, const char *key, int def) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v) return def;
    if (yyjson_is_int(v)) return (int)yyjson_get_int(v);
    if (yyjson_is_real(v)) return (int)yyjson_get_real(v);
    return def;
}

/* Single object parsing */
static void parse_single_object(yyjson_val *obj, SceneObject *o) {
    memset(o, 0, sizeof(SceneObject));

    json_float3(obj, "position", o->position, 0, 0, 0);
    json_float3(obj, "size", o->size, 1, 1, 1);

    /* Rotation: 3x3 matrix */
    yyjson_val *rot_arr = yyjson_obj_get(obj, "rotation");
    if (rot_arr && yyjson_is_arr(rot_arr) && yyjson_arr_size(rot_arr) >= 3) {
        for (int r = 0; r < 3; r++) {
            yyjson_val *row = yyjson_arr_get(rot_arr, r);
            if (row && yyjson_is_arr(row) && yyjson_arr_size(row) >= 3) {
                for (int c = 0; c < 3; c++) {
                    yyjson_val *v = yyjson_arr_get(row, c);
                    o->rotation[r * 3 + c] = (float)(yyjson_is_real(v) ? yyjson_get_real(v) :
                                                      yyjson_is_int(v) ? yyjson_get_int(v) : (r == c ? 1 : 0));
                }
            }
        }
    } else {
        o->rotation[0] = 1; o->rotation[4] = 1; o->rotation[8] = 1;
    }

    /* Primitive type (Todo: Allow custom objects) */
    const char *prim = json_str(obj, "primitive", "box");
    if (strcmp(prim, "sphere") == 0 || strcmp(prim, "ball") == 0) {
        o->prim_type = PRIM_SPHERE;
    } else if (strcmp(prim, "cylinder") == 0) {
        o->prim_type = PRIM_CYLINDER;
    } else if (strcmp(prim, "mesh") == 0) {
        o->prim_type = PRIM_MESH;
    } else {
        o->prim_type = PRIM_BOX;
    }

    /* Color */
    float color[3];
    json_float3(obj, "color", color, 1, 1, 1);
    o->color[0] = powf(color[0], 2.2f);
    o->color[1] = powf(color[1], 2.2f);
    o->color[2] = powf(color[2], 2.2f);

    /* Materials */
    o->roughness   = json_float(obj, "roughness", 0.5f);
    o->metallic    = json_float(obj, "metallic", 0.0f);
    o->reflectance = json_float(obj, "reflectance", 0.0f);
    o->transparency = json_float(obj, "transparency", 0.0f);
    o->ior         = json_float(obj, "ior", 1.5f);

    /* Emission */
    yyjson_val *em_val = yyjson_obj_get(obj, "emission");
    if (em_val) {
        if (yyjson_is_arr(em_val)) {
            json_float3(obj, "emission", o->emission, 0, 0, 0);
        } else {
            float e = (float)(yyjson_is_real(em_val) ? yyjson_get_real(em_val) : yyjson_get_int(em_val));
            o->emission[0] = e; o->emission[1] = e; o->emission[2] = e;
        }
    }

    /* Player flag */
    o->is_player = json_int(obj, "isPlayer", 0);

    /* Mesh triangle data (PRIM_MESH only) */
    if (o->prim_type == PRIM_MESH) {
        yyjson_val *verts_arr = yyjson_obj_get(obj, "vertices");
        yyjson_val *indices_arr = yyjson_obj_get(obj, "indices");

        if (verts_arr && yyjson_is_arr(verts_arr) && indices_arr && yyjson_is_arr(indices_arr)) {
            int num_verts = (int)yyjson_arr_size(verts_arr) / 3;
            int num_indices = (int)yyjson_arr_size(indices_arr);
            int num_tris = num_indices / 3;

            if (num_tris > 0 && num_verts > 0) {
                /* Read vertex positions */
                float *verts = (float *)malloc(num_verts * 3 * sizeof(float));
                for (int vi = 0; vi < num_verts * 3; vi++) {
                    yyjson_val *v = yyjson_arr_get(verts_arr, vi);
                    verts[vi] = (float)(yyjson_is_real(v) ? yyjson_get_real(v) : yyjson_get_int(v));
                }

                /* Build triangle vertex buffer (3 vertices per triangle, 3 floats each = 9 per tri) */
                o->mesh_vertices = (float *)malloc(num_tris * 9 * sizeof(float));
                o->mesh_tri_count = num_tris;

                for (int ti = 0; ti < num_tris; ti++) {
                    yyjson_val *i0 = yyjson_arr_get(indices_arr, ti * 3);
                    yyjson_val *i1 = yyjson_arr_get(indices_arr, ti * 3 + 1);
                    yyjson_val *i2 = yyjson_arr_get(indices_arr, ti * 3 + 2);
                    int idx0 = (int)(yyjson_is_int(i0) ? yyjson_get_int(i0) : 0);
                    int idx1 = (int)(yyjson_is_int(i1) ? yyjson_get_int(i1) : 0);
                    int idx2 = (int)(yyjson_is_int(i2) ? yyjson_get_int(i2) : 0);

                    /* Clamp indices */
                    if (idx0 >= num_verts) idx0 = 0;
                    if (idx1 >= num_verts) idx1 = 0;
                    if (idx2 >= num_verts) idx2 = 0;

                    float *dst = &o->mesh_vertices[ti * 9];
                    dst[0] = verts[idx0 * 3];     dst[1] = verts[idx0 * 3 + 1]; dst[2] = verts[idx0 * 3 + 2];
                    dst[3] = verts[idx1 * 3];     dst[4] = verts[idx1 * 3 + 1]; dst[5] = verts[idx1 * 3 + 2];
                    dst[6] = verts[idx2 * 3];     dst[7] = verts[idx2 * 3 + 1]; dst[8] = verts[idx2 * 3 + 2];
                }

                free(verts);
            }
        }
    }
}

/* Scene object parsing */
static void parse_objects(yyjson_val *arr, SceneObject **out, int *count) {
    *count = 0;
    *out = NULL;
    if (!arr || !yyjson_is_arr(arr)) return;

    int n = (int)yyjson_arr_size(arr);
    if (n == 0) return;

    *out = (SceneObject *)calloc(n, sizeof(SceneObject));
    *count = n;

    size_t idx, max;
    yyjson_val *obj;
    int i = 0;
    yyjson_arr_foreach(arr, idx, max, obj) {
        parse_single_object(obj, &(*out)[i++]);
    }
}

/* Light parsing */
static void parse_lights(yyjson_val *arr, LightData **out, int *count) {
    *count = 0;
    *out = NULL;
    if (!arr || !yyjson_is_arr(arr)) return;

    int n = (int)yyjson_arr_size(arr);
    if (n == 0) return;

    *out = (LightData *)calloc(n, sizeof(LightData));
    *count = n;

    size_t idx, max;
    yyjson_val *obj;
    int i = 0;
    yyjson_arr_foreach(arr, idx, max, obj) {
        LightData *l = &(*out)[i++];

        json_float3(obj, "position", l->position, 0, 0, 0);
        json_float3(obj, "color", l->color, 1, 1, 1);
        l->brightness = json_float(obj, "brightness", 1.0f);
        l->range      = json_float(obj, "range", 60.0f);
        json_float3(obj, "direction", l->direction, 0, -1, 0);
        l->angle      = json_float(obj, "angle", 180.0f);

        const char *type = json_str(obj, "type", "point");
        l->is_spot = (strcmp(type, "spot") == 0 || strcmp(type, "surface") == 0) ? 1 : 0;
    }
}

/* World settings parsing */
static void parse_world(yyjson_val *obj, WorldSettings *w) {
    world_settings_defaults(w);
    if (!obj || !yyjson_is_obj(obj)) return;

    json_float3(obj, "skyTop", w->sky_top, w->sky_top[0], w->sky_top[1], w->sky_top[2]);
    json_float3(obj, "skyBottom", w->sky_bottom, w->sky_bottom[0], w->sky_bottom[1], w->sky_bottom[2]);
    w->env_intensity = json_float(obj, "envIntensity", w->env_intensity);
    json_float3(obj, "sunDirection", w->sun_dir, w->sun_dir[0], w->sun_dir[1], w->sun_dir[2]);
    json_float3(obj, "sunColor", w->sun_color, w->sun_color[0], w->sun_color[1], w->sun_color[2]);
    w->sun_intensity = json_float(obj, "sunIntensity", w->sun_intensity);
    w->sun_angular_radius = json_float(obj, "sunAngularRadius", w->sun_angular_radius);
    json_float3(obj, "ambientColor", w->ambient, w->ambient[0], w->ambient[1], w->ambient[2]);
    w->ambient_intensity = json_float(obj, "ambientIntensity", w->ambient_intensity);
    w->fog_density = json_float(obj, "fogDensity", w->fog_density);
    json_float3(obj, "fogColor", w->fog_color, w->fog_color[0], w->fog_color[1], w->fog_color[2]);
    w->exposure = json_float(obj, "exposure", w->exposure);
    w->gamma = json_float(obj, "gamma", w->gamma);
    w->bloom_strength = json_float(obj, "bloomStrength", w->bloom_strength);
    w->bloom_threshold = json_float(obj, "bloomThreshold", w->bloom_threshold);
}

/* Camera parsing */
static bool parse_camera(yyjson_val *obj, Camera *cam) {
    if (!obj || !yyjson_is_obj(obj)) return false;

    json_float3(obj, "position", cam->position, 0, 0, 0);
    json_float3(obj, "forward", cam->forward, 0, 0, -1);
    json_float3(obj, "right", cam->right, 1, 0, 0);
    json_float3(obj, "up", cam->up, 0, 1, 0);
    cam->fov = json_float(obj, "fov", 70.0f);
    return true;
}

/* Scene parsing */
bool scene_parse_request(const char *json_data, size_t json_len, ParsedScene *out) {
    memset(out, 0, sizeof(ParsedScene));
    world_settings_defaults(&out->world);

    yyjson_doc *doc = yyjson_read(json_data, json_len, 0);
    if (!doc) {
        fprintf(stderr, "JSON parse error\n");
        return false;
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val *objects_arr = yyjson_obj_get(root, "objects");
    if (objects_arr && yyjson_is_arr(objects_arr) && yyjson_arr_size(objects_arr) > 0) {
        parse_objects(objects_arr, &out->objects, &out->num_objects);
        out->has_scene = true;
    }

    yyjson_val *lights_arr = yyjson_obj_get(root, "lights");
    if (lights_arr && yyjson_is_arr(lights_arr)) {
        parse_lights(lights_arr, &out->lights, &out->num_lights);
    }

    /* Parse player_objects so we can render the player faster (Broken somehow, I'll fix it later)*/
    yyjson_val *player_arr = yyjson_obj_get(root, "player_objects");
    if (player_arr && yyjson_is_arr(player_arr) && yyjson_arr_size(player_arr) > 0) {
        parse_objects(player_arr, &out->player_objects, &out->num_player_objects);
        for (int i = 0; i < out->num_player_objects; i++) {
            out->player_objects[i].is_player = 1;
        }
    }

    yyjson_val *world_obj = yyjson_obj_get(root, "world");
    if (world_obj) {
        parse_world(world_obj, &out->world);
    }

    yyjson_val *cam_obj = yyjson_obj_get(root, "camera");
    if (cam_obj) {
        out->has_camera = parse_camera(cam_obj, &out->camera);
    }

    out->seq = json_int(root, "seq", 0);

    yyjson_doc_free(doc);
    return true;
}
