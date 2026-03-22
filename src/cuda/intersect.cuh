/* intersect.cuh - ray-primitive intersection */
#ifndef INTERSECT_CUH
#define INTERSECT_CUH

#include "cuda_utils.cuh"

__device__ __forceinline__ float intersect_sphere(
    float3 ro, float3 rd,
    float cx, float cy, float cz, float radius)
{
    float3 oc = make_float3(ro.x - cx, ro.y - cy, ro.z - cz);
    float b = f3_dot(rd, oc);
    float c = f3_dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return CT_INF_D;
    float sq = sqrtf(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    if (t0 > CT_EPS_D) return t0;
    if (t1 > CT_EPS_D) return t1;
    return CT_INF_D;
}

__device__ __forceinline__ float intersect_obb(
    float3 ro, float3 rd,
    float cx, float cy, float cz,
    float hx, float hy, float hz,
    const float *rot)
{
    float3 delta = make_float3(ro.x - cx, ro.y - cy, ro.z - cz);
    float3 lo = make_float3(
        rot[0] * delta.x + rot[3] * delta.y + rot[6] * delta.z,
        rot[1] * delta.x + rot[4] * delta.y + rot[7] * delta.z,
        rot[2] * delta.x + rot[5] * delta.y + rot[8] * delta.z
    );
    float3 ld = make_float3(
        rot[0] * rd.x + rot[3] * rd.y + rot[6] * rd.z,
        rot[1] * rd.x + rot[4] * rd.y + rot[7] * rd.z,
        rot[2] * rd.x + rot[5] * rd.y + rot[8] * rd.z
    );

    float idx = (fabsf(ld.x) > 1e-6f) ? (1.0f / ld.x) : ((ld.x >= 0.0f) ? 1e6f : -1e6f);
    float idy = (fabsf(ld.y) > 1e-6f) ? (1.0f / ld.y) : ((ld.y >= 0.0f) ? 1e6f : -1e6f);
    float idz = (fabsf(ld.z) > 1e-6f) ? (1.0f / ld.z) : ((ld.z >= 0.0f) ? 1e6f : -1e6f);

    float tx1 = (-hx - lo.x) * idx;
    float tx2 = ( hx - lo.x) * idx;
    float ty1 = (-hy - lo.y) * idy;
    float ty2 = ( hy - lo.y) * idy;
    float tz1 = (-hz - lo.z) * idz;
    float tz2 = ( hz - lo.z) * idz;

    float t_enter = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
    float t_exit  = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));

    if (t_exit < t_enter || t_exit < CT_EPS_D) return CT_INF_D;
    if (t_enter > CT_EPS_D) return t_enter;
    return t_exit;
}

__device__ __forceinline__ float intersect_cylinder(
    float3 ro, float3 rd,
    float cx, float cy, float cz,
    float hx, float hy, float hz,
    const float *rot)
{
    float3 delta = make_float3(ro.x - cx, ro.y - cy, ro.z - cz);
    float3 lo = make_float3(
        rot[0] * delta.x + rot[3] * delta.y + rot[6] * delta.z,
        rot[1] * delta.x + rot[4] * delta.y + rot[7] * delta.z,
        rot[2] * delta.x + rot[5] * delta.y + rot[8] * delta.z
    );
    float3 ld = make_float3(
        rot[0] * rd.x + rot[3] * rd.y + rot[6] * rd.z,
        rot[1] * rd.x + rot[4] * rd.y + rot[7] * rd.z,
        rot[2] * rd.x + rot[5] * rd.y + rot[8] * rd.z
    );

    float half_len = hx;
    float radius = hy;
    float t_best = CT_INF_D;

    float a = ld.y * ld.y + ld.z * ld.z;
    float b = 2.0f * (lo.y * ld.y + lo.z * ld.z);
    float c = lo.y * lo.y + lo.z * lo.z - radius * radius;
    float disc = b * b - 4.0f * a * c;

    if (disc >= 0.0f && a > 1e-8f) {
        float sq = sqrtf(disc);
        float inv2a = 0.5f / a;
        float t0 = (-b - sq) * inv2a;
        float t1 = (-b + sq) * inv2a;

        float x0 = lo.x + t0 * ld.x;
        if (t0 > CT_EPS_D && fabsf(x0) <= half_len)
            t_best = fminf(t_best, t0);
        float x1 = lo.x + t1 * ld.x;
        if (t1 > CT_EPS_D && fabsf(x1) <= half_len)
            t_best = fminf(t_best, t1);
    }

    if (fabsf(ld.x) > 1e-6f) {
        float inv_dx = 1.0f / ld.x;
        float tc_pos = (half_len - lo.x) * inv_dx;
        if (tc_pos > CT_EPS_D) {
            float yc = lo.y + tc_pos * ld.y;
            float zc = lo.z + tc_pos * ld.z;
            if (yc * yc + zc * zc <= radius * radius)
                t_best = fminf(t_best, tc_pos);
        }
        float tc_neg = (-half_len - lo.x) * inv_dx;
        if (tc_neg > CT_EPS_D) {
            float yc = lo.y + tc_neg * ld.y;
            float zc = lo.z + tc_neg * ld.z;
            if (yc * yc + zc * zc <= radius * radius)
                t_best = fminf(t_best, tc_neg);
        }
    }

    return t_best;
}

/* Moller-Trumbore */
__device__ __forceinline__ float intersect_triangle(
    float3 lo, float3 ld,
    float3 v0, float3 v1, float3 v2)
{
    float3 e1 = f3_sub(v1, v0);
    float3 e2 = f3_sub(v2, v0);
    float3 pvec = f3_cross(ld, e2);
    float det = f3_dot(e1, pvec);

    if (fabsf(det) < 1e-8f) return CT_INF_D;

    float inv_det = 1.0f / det;
    float3 tvec = f3_sub(lo, v0);
    float u = f3_dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return CT_INF_D;

    float3 qvec = f3_cross(tvec, e1);
    float v = f3_dot(ld, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return CT_INF_D;

    float t = f3_dot(e2, qvec) * inv_det;
    return (t > CT_EPS_D) ? t : CT_INF_D;
}

__device__ __forceinline__ float intersect_mesh(
    float3 ro, float3 rd,
    float cx, float cy, float cz,
    float hx, float hy, float hz,
    const float *rot,
    const KernelScene *scene,
    int obj_id,
    int *tri_idx)
{
    int offset = __ldg(&scene->mesh_tri_offset[obj_id]);
    int count  = __ldg(&scene->mesh_tri_count[obj_id]);
    if (count <= 0) return CT_INF_D;

    float3 delta = make_float3(ro.x - cx, ro.y - cy, ro.z - cz);
    float3 lo = make_float3(
        rot[0] * delta.x + rot[3] * delta.y + rot[6] * delta.z,
        rot[1] * delta.x + rot[4] * delta.y + rot[7] * delta.z,
        rot[2] * delta.x + rot[5] * delta.y + rot[8] * delta.z
    );
    float3 ld = make_float3(
        rot[0] * rd.x + rot[3] * rd.y + rot[6] * rd.z,
        rot[1] * rd.x + rot[4] * rd.y + rot[7] * rd.z,
        rot[2] * rd.x + rot[5] * rd.y + rot[8] * rd.z
    );

    /* Scale ray origin to account for half-extents (mesh verts are in unit space) */
    lo.x /= hx; lo.y /= hy; lo.z /= hz;
    ld.x /= hx; ld.y /= hy; ld.z /= hz;

    float best_t = CT_INF_D;
    int best_tri = -1;

    for (int i = 0; i < count; i++) {
        int ti = offset + i;
        float3 v0 = make_float3(
            __ldg(&scene->tri_v0_x[ti]),
            __ldg(&scene->tri_v0_y[ti]),
            __ldg(&scene->tri_v0_z[ti]));
        float3 v1 = make_float3(
            __ldg(&scene->tri_v1_x[ti]),
            __ldg(&scene->tri_v1_y[ti]),
            __ldg(&scene->tri_v1_z[ti]));
        float3 v2 = make_float3(
            __ldg(&scene->tri_v2_x[ti]),
            __ldg(&scene->tri_v2_y[ti]),
            __ldg(&scene->tri_v2_z[ti]));

        float t = intersect_triangle(lo, ld, v0, v1, v2);
        if (t < best_t) {
            best_t = t;
            best_tri = ti;
        }
    }

    *tri_idx = best_tri;

    if (best_t < CT_INF_D) {
        float3 lp = f3_add(lo, f3_scale(ld, best_t));
        lp.x *= hx; lp.y *= hy; lp.z *= hz;
        /* R * lp + center */
        float3 wp = make_float3(
            rot[0] * lp.x + rot[1] * lp.y + rot[2] * lp.z + cx,
            rot[3] * lp.x + rot[4] * lp.y + rot[5] * lp.z + cy,
            rot[6] * lp.x + rot[7] * lp.y + rot[8] * lp.z + cz
        );
        return f3_length(f3_sub(wp, ro));
    }
    return CT_INF_D;
}

__device__ __forceinline__ void intersect_all(
    float3 ro, float3 rd,
    const KernelScene *scene,
    float *t_hit, int *hit_id)
{
    float best_t = CT_INF_D;
    int best_id = -1;

    for (int i = 0; i < scene->num_objects; i++) {
        float px = __ldg(&scene->pos_x[i]);
        float py = __ldg(&scene->pos_y[i]);
        float pz = __ldg(&scene->pos_z[i]);
        float hx = __ldg(&scene->half_x[i]);
        float hy = __ldg(&scene->half_y[i]);
        float hz = __ldg(&scene->half_z[i]);
        int pt = __ldg(&scene->prim_type[i]);

        float t;
        if (pt == PRIM_SPHERE) {
            float r = fmaxf(hx, fmaxf(hy, hz));
            t = intersect_sphere(ro, rd, px, py, pz, r);
        } else if (pt == PRIM_CYLINDER) {
            t = intersect_cylinder(ro, rd, px, py, pz, hx, hy, hz,
                                   &scene->rot[i * 9]);
        } else if (pt == PRIM_MESH) {
            int tri_idx;
            t = intersect_mesh(ro, rd, px, py, pz, hx, hy, hz,
                              &scene->rot[i * 9], scene, i, &tri_idx);
        } else {
            t = intersect_obb(ro, rd, px, py, pz, hx, hy, hz,
                              &scene->rot[i * 9]);
        }

        if (t < best_t) {
            best_t = t;
            best_id = i;
        }
    }

    *t_hit = best_t;
    *hit_id = best_id;
}

__device__ __forceinline__ float3 compute_normal(
    float3 hp, int obj_id, const KernelScene *scene)
{
    float cx = __ldg(&scene->pos_x[obj_id]);
    float cy = __ldg(&scene->pos_y[obj_id]);
    float cz = __ldg(&scene->pos_z[obj_id]);
    float hx = __ldg(&scene->half_x[obj_id]);
    float hy = __ldg(&scene->half_y[obj_id]);
    float hz = __ldg(&scene->half_z[obj_id]);
    int pt = __ldg(&scene->prim_type[obj_id]);

    const float *rot = &scene->rot[obj_id * 9];
    float r0 = __ldg(&rot[0]), r1 = __ldg(&rot[1]), r2 = __ldg(&rot[2]);
    float r3 = __ldg(&rot[3]), r4 = __ldg(&rot[4]), r5 = __ldg(&rot[5]);
    float r6 = __ldg(&rot[6]), r7 = __ldg(&rot[7]), r8 = __ldg(&rot[8]);

    if (pt == PRIM_SPHERE) {
        return f3_normalize(make_float3(hp.x - cx, hp.y - cy, hp.z - cz));
    }

    float3 delta = make_float3(hp.x - cx, hp.y - cy, hp.z - cz);
    float3 p_local = make_float3(
        r0 * delta.x + r3 * delta.y + r6 * delta.z,
        r1 * delta.x + r4 * delta.y + r7 * delta.z,
        r2 * delta.x + r5 * delta.y + r8 * delta.z
    );

    float3 n_local;

    if (pt == PRIM_MESH) {
        int offset = __ldg(&scene->mesh_tri_offset[obj_id]);
        int count  = __ldg(&scene->mesh_tri_count[obj_id]);

        float3 p_unit = make_float3(p_local.x / hx, p_local.y / hy, p_local.z / hz);
        float best_dist = CT_INF_D;
        n_local = make_float3(0.0f, 1.0f, 0.0f);

        for (int i = 0; i < count; i++) {
            int ti = offset + i;
            float3 v0 = make_float3(
                __ldg(&scene->tri_v0_x[ti]),
                __ldg(&scene->tri_v0_y[ti]),
                __ldg(&scene->tri_v0_z[ti]));
            float3 v1 = make_float3(
                __ldg(&scene->tri_v1_x[ti]),
                __ldg(&scene->tri_v1_y[ti]),
                __ldg(&scene->tri_v1_z[ti]));
            float3 v2 = make_float3(
                __ldg(&scene->tri_v2_x[ti]),
                __ldg(&scene->tri_v2_y[ti]),
                __ldg(&scene->tri_v2_z[ti]));

            float3 e1 = f3_sub(v1, v0);
            float3 e2 = f3_sub(v2, v0);
            float3 fn = f3_cross(e1, e2);
            float d = fabsf(f3_dot(f3_sub(p_unit, v0), f3_normalize(fn)));
            if (d < best_dist) {
                best_dist = d;
                n_local = f3_normalize(fn);
            }
        }

        n_local = f3_normalize(make_float3(
            n_local.x / hx, n_local.y / hy, n_local.z / hz));
    } else if (pt == PRIM_CYLINDER) {
        float half_len = hx;
        int is_cap = (fabsf(fabsf(p_local.x) - half_len) < 0.01f) ? 1 : 0;
        if (is_cap) {
            n_local = make_float3((p_local.x > 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else {
            n_local = f3_normalize(make_float3(0.0f, p_local.y, p_local.z));
        }
    } else {
        /* Box: find nearest face */
        float ax = fabsf(p_local.x) / (hx + 1e-6f);
        float ay = fabsf(p_local.y) / (hy + 1e-6f);
        float az = fabsf(p_local.z) / (hz + 1e-6f);

        if (ax >= ay && ax >= az) {
            n_local = make_float3((p_local.x > 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (ay >= az) {
            n_local = make_float3(0.0f, (p_local.y > 0.0f) ? 1.0f : -1.0f, 0.0f);
        } else {
            n_local = make_float3(0.0f, 0.0f, (p_local.z > 0.0f) ? 1.0f : -1.0f);
        }
    }

    /* Transform normal back to world space: R * n_local */
    float3 n_world = make_float3(
        r0 * n_local.x + r1 * n_local.y + r2 * n_local.z,
        r3 * n_local.x + r4 * n_local.y + r5 * n_local.z,
        r6 * n_local.x + r7 * n_local.y + r8 * n_local.z
    );
    return f3_normalize(n_world);
}

__device__ __forceinline__ bool shadow_test(
    float3 ro, float3 rd, float max_t,
    const KernelScene *scene)
{
    for (int i = 0; i < scene->num_objects; i++) {
        float px = __ldg(&scene->pos_x[i]);
        float py = __ldg(&scene->pos_y[i]);
        float pz = __ldg(&scene->pos_z[i]);
        float hx = __ldg(&scene->half_x[i]);
        float hy = __ldg(&scene->half_y[i]);
        float hz = __ldg(&scene->half_z[i]);
        int pt = __ldg(&scene->prim_type[i]);

        float t;
        if (pt == PRIM_SPHERE) {
            float r = fmaxf(hx, fmaxf(hy, hz));
            t = intersect_sphere(ro, rd, px, py, pz, r);
        } else if (pt == PRIM_CYLINDER) {
            t = intersect_cylinder(ro, rd, px, py, pz, hx, hy, hz,
                                   &scene->rot[i * 9]);
        } else if (pt == PRIM_MESH) {
            int tri_idx;
            t = intersect_mesh(ro, rd, px, py, pz, hx, hy, hz,
                              &scene->rot[i * 9], scene, i, &tri_idx);
        } else {
            t = intersect_obb(ro, rd, px, py, pz, hx, hy, hz,
                              &scene->rot[i * 9]);
        }

        if (t < max_t) return true;
    }
    return false;
}

#endif /* INTERSECT_CUH */
