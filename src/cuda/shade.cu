/* shade.cu - path trace kernel */

#include "cuda_utils.cuh"
#include "ray_gen.cuh"
#include "intersect.cuh"
#include "shade.cuh"

__global__ void pathtrace_kernel(
    const KernelScene   scene,
    const KernelLights  lights,
    const KernelWorld   world,
    const KernelCamera  cam,
    const KernelParams  params,
    float * __restrict__ accum_r,
    float * __restrict__ accum_g,
    float * __restrict__ accum_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = params.width * params.height;

    for (int pixel = idx; pixel < total_pixels; pixel += blockDim.x * gridDim.x) {
        int px = pixel % params.width;
        int py = pixel / params.width;

        float sample_r = 0.0f, sample_g = 0.0f, sample_b = 0.0f;

        for (int s = 0; s < params.spp; s++) {
            RNGState rng;
            rng_init(&rng, pixel, s, params.frame_seed);

            /* Generate primary ray with subpixel jitter */
            float3 origin, dir;
            generate_primary_ray(&cam, px, py, params.width, params.height, &rng, &origin, &dir);

            float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
            float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
            bool alive = true;

            for (int bounce = 0; bounce < params.max_bounces && alive; bounce++) {
                float t_hit;
                int hit_id;
                intersect_all(origin, dir, &scene, &t_hit, &hit_id);

                if (hit_id < 0) {
                    /* Miss: sample environment */
                    float3 env = sample_env(dir, &world);

                    /* Fog on miss */
                    if (world.fog_density > 0.0f) {
                        float tr = expf(-world.fog_density * 50.0f);
                        env = f3_add(f3_scale(env, tr), f3_scale(world.fog_color, 1.0f - tr));
                    }

                    radiance = f3_add(radiance, f3_mul(throughput, env));
                    break;
                }

                float3 hp = f3_add(origin, f3_scale(dir, t_hit));
                float3 normal = compute_normal(hp, hit_id, &scene);

                float3 base_color = make_float3(
                    __ldg(&scene.color_r[hit_id]),
                    __ldg(&scene.color_g[hit_id]),
                    __ldg(&scene.color_b[hit_id])
                );
                float rough = __ldg(&scene.roughness[hit_id]);
                float metal = __ldg(&scene.metallic[hit_id]);
                float refl  = __ldg(&scene.reflectance[hit_id]);
                float transp = __ldg(&scene.transparency[hit_id]);
                float obj_ior = __ldg(&scene.ior[hit_id]);
                float3 emiss = make_float3(
                    __ldg(&scene.emission_r[hit_id]),
                    __ldg(&scene.emission_g[hit_id]),
                    __ldg(&scene.emission_b[hit_id])
                );

                /* Flip normal for backface */
                float dot_dn = f3_dot(dir, normal);
                bool entering = dot_dn < 0.0f;
                if (!entering) normal = f3_neg(normal);

                /* fog */
                if (world.fog_density > 0.0f) {
                    float fog_tr = expf(-world.fog_density * t_hit);
                    radiance = f3_add(radiance, f3_mul(throughput,
                        f3_scale(world.fog_color, 1.0f - fog_tr)));
                    throughput = f3_scale(throughput, fog_tr);
                }

                /* emissive */
                float em_mag = fmaxf(emiss.x, fmaxf(emiss.y, emiss.z));
                if (em_mag > 0.0f) {
                    radiance = f3_add(radiance, f3_mul(throughput, emiss));
                    alive = false;
                    break;
                }

                float3 v = f3_normalize(f3_neg(dir));

                /* refraction */
                bool is_transparent = transp > 0.01f;

                if (is_transparent) {
                    float eta = entering ? (1.0f / fminf(fmaxf(obj_ior, 1.0f), 3.0f))
                                         : fminf(fmaxf(obj_ior, 1.0f), 3.0f);
                    float cos_i = fmaxf(f3_dot(f3_neg(dir), normal), 0.0f);
                    float F_glass = fresnel_dielectric(cos_i, eta);

                    if (rng_float(&rng) < F_glass) {
                        /* Reflect */
                        dir = f3_reflect(dir, normal);
                        dir = f3_normalize(dir);
                        origin = f3_add(hp, f3_scale(normal, 0.002f));
                    } else {
                        /* Refract (Snell's law) */
                        float ct = sqrtf(fmaxf(1.0f - eta * eta * (1.0f - cos_i * cos_i), 0.0f));
                        dir = f3_normalize(f3_add(
                            f3_scale(dir, eta),
                            f3_scale(normal, eta * cos_i - ct)
                        ));
                        origin = f3_sub(hp, f3_scale(normal, 0.002f));
                        throughput = f3_mul(throughput, base_color);
                    }
                    continue;  /* next bounce */
                }

                float3 shadow_o = f3_add(hp, f3_scale(normal, 0.002f));

                /* Ambient (with optional AO on first bounce) */
                float ao_factor = 1.0f;
                if (params.do_ao && bounce == 0 && params.ao_samples > 0) {
                    int ao_hits = 0;
                    for (int ai = 0; ai < params.ao_samples; ai++) {
                        float3 ao_dir = cosine_hemisphere(normal, &rng);
                        float ao_t;
                        int ao_id;
                        intersect_all(shadow_o, ao_dir, &scene, &ao_t, &ao_id);
                        if (ao_id >= 0 && ao_t < params.ao_radius) {
                            ao_hits++;
                        }
                    }
                    ao_factor = 1.0f - ((float)ao_hits / (float)params.ao_samples) * 0.8f;
                }

                float3 ambient_contrib = f3_scale(
                    f3_mul(base_color, world.ambient),
                    world.ambient_intensity * ao_factor
                );
                radiance = f3_add(radiance, f3_mul(throughput, ambient_contrib));

                /* sun */
                float3 sun_l;
                if (world.sun_angular_radius > 0.0f) {
                    sun_l = sample_cone(world.sun_dir, world.sun_angular_radius, &rng);
                } else {
                    sun_l = world.sun_dir;
                }

                float sun_ndotl = fmaxf(f3_dot(normal, sun_l), 0.0f);
                if (sun_ndotl > 0.0f && params.do_shadows) {
                    bool sun_occluded = shadow_test(shadow_o, sun_l, CT_INF_D, &scene);
                    if (!sun_occluded) {
                        float3 sun_brdf = eval_brdf_cos(base_color, rough, metal, refl, normal, v, sun_l);
                        radiance = f3_add(radiance, f3_mul(throughput,
                            f3_scale(f3_mul(sun_brdf, world.sun_color), world.sun_intensity)));
                    }
                } else if (sun_ndotl > 0.0f && !params.do_shadows) {
                    float3 sun_brdf = eval_brdf_cos(base_color, rough, metal, refl, normal, v, sun_l);
                    radiance = f3_add(radiance, f3_mul(throughput,
                        f3_scale(f3_mul(sun_brdf, world.sun_color), world.sun_intensity)));
                }

                /* nee */
                if (params.do_nee && scene.num_emissive > 0) {
                    float pick_r = rng_float(&rng);
                    int pick_idx = 0;
                    float prob_sum = 0.0f;
                    for (int ei = 0; ei < scene.num_emissive; ei++) {
                        prob_sum += __ldg(&scene.emissive_probs[ei]);
                        if (pick_r < prob_sum) { pick_idx = ei; break; }
                    }
                    float p_light = fmaxf(__ldg(&scene.emissive_probs[pick_idx]), 1e-8f);

                    float3 lc = make_float3(
                        __ldg(&scene.emissive_centers_x[pick_idx]),
                        __ldg(&scene.emissive_centers_y[pick_idx]),
                        __ldg(&scene.emissive_centers_z[pick_idx])
                    );
                    float lr = fmaxf(__ldg(&scene.emissive_radii[pick_idx]), 0.01f);
                    float3 le = make_float3(
                        __ldg(&scene.emissive_emission_r[pick_idx]),
                        __ldg(&scene.emissive_emission_g[pick_idx]),
                        __ldg(&scene.emissive_emission_b[pick_idx])
                    );

                    /* Random point on emissive sphere */
                    float3 nL = random_sphere_dir(&rng);
                    float3 xL = f3_add(lc, f3_scale(nL, lr));
                    float3 toL = f3_sub(xL, hp);
                    float nee_dist2 = fmaxf(f3_dot(toL, toL), 1e-8f);
                    float nee_dist = sqrtf(nee_dist2);
                    float3 nee_wi = f3_scale(toL, 1.0f / nee_dist);

                    float nee_ndotl = fmaxf(f3_dot(normal, nee_wi), 0.0f);
                    float nL_dot_wi = fmaxf(f3_dot(nL, f3_neg(nee_wi)), 0.0f);

                    if (nee_ndotl > 0.0f && nL_dot_wi > 0.0f) {
                        /* Shadow test for NEE */
                        float sh_t;
                        int sh_id;
                        intersect_all(shadow_o, nee_wi, &scene, &sh_t, &sh_id);
                        bool nee_visible = (sh_id < 0) || (sh_t >= nee_dist - 0.01f);

                        if (nee_visible) {
                            float area = 4.0f * CT_PI_D * lr * lr;
                            float pdf_dir = nee_dist2 / (area * fmaxf(nL_dot_wi, 1e-8f));
                            float pdf = fmaxf(p_light * pdf_dir, 1e-12f);

                            float3 nee_brdf = eval_brdf_cos(base_color, rough, metal, refl,
                                                             normal, v, nee_wi);
                            float3 nee_contrib = f3_scale(f3_mul(nee_brdf, le), 1.0f / pdf);
                            radiance = f3_add(radiance, f3_mul(throughput, nee_contrib));
                        }
                    }
                }

                /* point/spot lights */
                if (lights.count > 0) {
                    for (int li = 0; li < lights.count; li++) {
                        float3 lpos = make_float3(
                            __ldg(&lights.pos_x[li]),
                            __ldg(&lights.pos_y[li]),
                            __ldg(&lights.pos_z[li])
                        );
                        float3 lcol = make_float3(
                            __ldg(&lights.color_r[li]),
                            __ldg(&lights.color_g[li]),
                            __ldg(&lights.color_b[li])
                        );
                        float lbright = __ldg(&lights.brightness[li]);
                        float lrange  = __ldg(&lights.range[li]);

                        float3 to_light = f3_sub(lpos, hp);
                        float light_dist2 = fmaxf(f3_dot(to_light, to_light), 1e-4f);
                        float light_dist = sqrtf(light_dist2);
                        float3 li_wi = f3_scale(to_light, 1.0f / light_dist);

                        /* Range attenuation */
                        float range_atten = fmaxf(1.0f - light_dist / lrange, 0.0f);
                        float li_atten = (range_atten * range_atten) / light_dist2 * lbright;

                        /* Spot light attenuation */
                        if (__ldg(&lights.is_spot[li])) {
                            float3 ldir = make_float3(
                                __ldg(&lights.dir_x[li]),
                                __ldg(&lights.dir_y[li]),
                                __ldg(&lights.dir_z[li])
                            );
                            float li_angle = __ldg(&lights.angle[li]);
                            float cos_angle = f3_dot(f3_neg(li_wi), ldir);
                            float cos_outer = cosf(li_angle);
                            float cos_inner = cosf(li_angle * 0.8f);
                            float spot_factor = fminf(fmaxf(
                                (cos_angle - cos_outer) / (cos_inner - cos_outer + 1e-6f),
                                0.0f), 1.0f);
                            li_atten *= spot_factor;
                        }

                        float li_ndotl = fmaxf(f3_dot(normal, li_wi), 0.0f);
                        if (li_ndotl <= 0.0f || li_atten < 1e-6f) continue;

                        /* Shadow test */
                        bool li_occluded = false;
                        if (params.do_shadows) {
                            float sh_t;
                            int sh_id;
                            intersect_all(shadow_o, li_wi, &scene, &sh_t, &sh_id);
                            li_occluded = (sh_id >= 0) && (sh_t < light_dist - 0.01f);
                        }

                        if (!li_occluded) {
                            float3 li_brdf = eval_brdf_cos(base_color, rough, metal, refl,
                                                            normal, v, li_wi);
                            radiance = f3_add(radiance, f3_mul(throughput,
                                f3_scale(f3_mul(li_brdf, lcol), li_atten)));
                        }
                    }
                }

                if (!params.do_gi && !params.do_reflections) {
                    alive = false;
                    break;
                }

                /* Fresnel-based specular/diffuse split */
                float3 f0 = f3_lerp(make_float3(0.04f, 0.04f, 0.04f), base_color, metal);
                f0 = f3_lerp(f0, make_float3(1.0f, 1.0f, 1.0f), refl);
                float ndv = fmaxf(f3_dot(normal, v), 0.0f);
                float3 F_o = fresnel_schlick(ndv, f0);
                float spec_prob = fminf(fmaxf((F_o.x + F_o.y + F_o.z) / 3.0f, 0.05f), 0.95f);

                bool do_spec;
                if (params.do_reflections && params.do_gi) {
                    do_spec = rng_float(&rng) < spec_prob;
                } else if (params.do_reflections) {
                    do_spec = true;
                } else {
                    do_spec = false;
                }

                float3 new_dir;
                if (do_spec) {
                    /* Specular: reflect + roughness perturbation */
                    float3 rd = f3_reflect(v, normal);  /* Note: reflect(-v, n) = v - 2(v.n)n */
                    /* Actually we want reflect(dir, normal) equivalent:
                       reflected_view = v - 2*(v.n)*n, but we want outgoing direction */
                    rd = f3_sub(v, f3_scale(normal, 2.0f * f3_dot(v, normal)));
                    rd = f3_normalize(f3_add(rd, f3_scale(random_sphere_dir(&rng), rough)));
                    /* Ensure above surface */
                    if (f3_dot(rd, normal) < 0.0f) {
                        rd = f3_sub(rd, f3_scale(normal, 2.0f * f3_dot(rd, normal)));
                        rd = f3_normalize(rd);
                    }
                    new_dir = rd;

                    /* Specular throughput */
                    throughput = f3_mul(throughput,
                        f3_scale(F_o, 1.0f / fmaxf(spec_prob, 0.05f)));
                } else {
                    /* Diffuse: cosine-weighted hemisphere */
                    new_dir = cosine_hemisphere(normal, &rng);

                    /* Diffuse throughput */
                    float3 diff_tp = f3_scale(base_color, (1.0f - metal) / fmaxf(1.0f - spec_prob, 0.05f));
                    throughput = f3_mul(throughput, diff_tp);
                }

                /* Clamp throughput to prevent fireflies */
                throughput = f3_min(throughput, 50.0f);

                origin = shadow_o;
                dir = new_dir;

                /* russian roulette */
                if (bounce >= 2) {
                    float lum = f3_luminance(throughput);
                    float surv = fminf(fmaxf(lum, 0.1f), 0.9f);
                    if (rng_float(&rng) > surv) {
                        alive = false;
                    } else {
                        throughput = f3_scale(throughput, 1.0f / fmaxf(surv, 0.1f));
                    }
                }
            } /* end bounce loop */

            /* Firefly clamp: cap per-sample radiance */
            float sample_lum = f3_luminance(radiance);
            if (sample_lum > 10.0f) {
                radiance = f3_scale(radiance, 10.0f / fmaxf(sample_lum, 1e-8f));
            }

            sample_r += radiance.x;
            sample_g += radiance.y;
            sample_b += radiance.z;
        } /* end SPP loop */

        float inv_spp = 1.0f / (float)params.spp;
        accum_r[pixel] += sample_r * inv_spp;
        accum_g[pixel] += sample_g * inv_spp;
        accum_b[pixel] += sample_b * inv_spp;
    }
}


extern "C" {

void launch_pathtrace(
    const void *scene_ptr,
    const void *lights_ptr,
    const void *world_ptr,
    const void *cam_ptr,
    const void *params_ptr,
    float *accum_r, float *accum_g, float *accum_b,
    void *stream_ptr)
{
    const KernelScene  *scene  = (const KernelScene *)scene_ptr;
    const KernelLights *lights = (const KernelLights *)lights_ptr;
    const KernelWorld  *world  = (const KernelWorld *)world_ptr;
    const KernelCamera *cam    = (const KernelCamera *)cam_ptr;
    const KernelParams *params = (const KernelParams *)params_ptr;
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    int total_pixels = params->width * params->height;
    if (total_pixels <= 0) return;

    int block_size = 0, min_grid = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
                                       pathtrace_kernel, 0, 0);
    int grid_size = (total_pixels + block_size - 1) / block_size;

    pathtrace_kernel<<<grid_size, block_size, 0, stream>>>(
        *scene, *lights, *world, *cam, *params,
        accum_r, accum_g, accum_b
    );
}

} /* extern "C" */
