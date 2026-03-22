#include "ctracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CMD Parcing */
typedef struct {
    int port;
    int width;
    int height;
    int spp;
    int bounces;
} CmdArgs;

static CmdArgs parse_args(int argc, char **argv) {
    CmdArgs args;
    args.port = CT_DEFAULT_PORT;
    args.width = CT_DEFAULT_WIDTH;
    args.height = CT_DEFAULT_HEIGHT;
    args.spp = CT_DEFAULT_SPP;
    args.bounces = CT_DEFAULT_BOUNCES;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            args.port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            args.width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            args.height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) {
            args.spp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bounces") == 0 && i + 1 < argc) {
            args.bounces = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("CTracer - GPU Path Tracer for Roblox\n");
            printf("Usage: ctracer [options]\n");
            printf("  --port PORT      HTTP server port (default: %d)\n", CT_DEFAULT_PORT);
            printf("  --width W        Render width (default: %d)\n", CT_DEFAULT_WIDTH);
            printf("  --height H       Render height (default: %d)\n", CT_DEFAULT_HEIGHT);
            printf("  --spp S          Samples per pixel per frame (default: %d)\n", CT_DEFAULT_SPP);
            printf("  --bounces B      Max path bounces (default: %d)\n", CT_DEFAULT_BOUNCES);
            exit(0);
        }
    }
    return args;
}

/* Render thread and state */
static RendererCtx g_renderer;
static SceneQueue  g_scene_queue;
static SceneQueue  g_render_queue;
static ServerState g_server;
static volatile int g_running = 1;

#ifdef _WIN32
static DWORD WINAPI render_thread_func(LPVOID arg) {
#else
static void *render_thread_func(void *arg) {
#endif
    (void)arg;

    while (g_running) {
        /* Process scene updates */
        ParsedScene ps;
        while (scene_queue_pop(&g_scene_queue, &ps)) {
            if (ps.has_scene) {
                renderer_upload_scene(&g_renderer, ps.objects, ps.num_objects,
                                     ps.lights, ps.num_lights);
                renderer_set_world(&g_renderer, &ps.world);
                renderer_reset_accumulation(&g_renderer);
            }
            parsed_scene_free(&ps);
        }

        /* Process render requests */
        bool did_render = false;
        while (scene_queue_pop(&g_render_queue, &ps)) {
            if (ps.has_camera) {
                renderer_set_camera(&g_renderer, &ps.camera);
            }
            /* Also handle inline scene data in render requests */
            if (ps.has_scene) {
                renderer_upload_scene(&g_renderer, ps.objects, ps.num_objects,
                                     ps.lights, ps.num_lights);
                renderer_set_world(&g_renderer, &ps.world);
                renderer_reset_accumulation(&g_renderer);
            }

            /* Upload player objects*/
            if (ps.num_player_objects > 0 && ps.player_objects) {
                renderer_upload_player_objects(&g_renderer, ps.player_objects,
                                              ps.num_player_objects);
            }

            g_renderer.output_seq = ps.seq;
            parsed_scene_free(&ps);
            did_render = true;
        }

        if (did_render || g_renderer.gpu_scene.num_objects > 0) {
            renderer_render(&g_renderer);

            /* Signal response to the server */
            g_server.response_width = g_renderer.buf_width;
            g_server.response_height = g_renderer.buf_height;
            g_server.response_sample = g_renderer.sample_count;
            g_server.response_seq = g_renderer.output_seq;
            ct_atomic_store(&g_server.response_ready, 1);
        } else {
            #ifdef _WIN32
            Sleep(1);
            #else
            usleep(1000);
            #endif
        }
    }

    #ifdef _WIN32
    return 0;
    #else
    return NULL;
    #endif
}

/* Main shit */
int main(int argc, char **argv) {
    CmdArgs args = parse_args(argc, argv);

    printf("=== CTracer - C/CUDA Path Tracer ===\n");
    printf("Port: %d | Resolution: %dx%d | SPP: %d | Bounces: %d\n",
           args.port, args.width, args.height, args.spp, args.bounces);

    /* Initialize renderer (CUDA) */
    renderer_init(&g_renderer);
    g_renderer.settings.width = args.width;
    g_renderer.settings.height = args.height;
    g_renderer.settings.spp = args.spp;
    g_renderer.settings.max_bounces = args.bounces;
    renderer_resize(&g_renderer, args.width, args.height);

    /* Initialize queues */
    scene_queue_init(&g_scene_queue);
    scene_queue_init(&g_render_queue);

    /* Initialize HTTP server */
    server_init(&g_server, args.port, &g_scene_queue, &g_render_queue, &g_renderer);

    /* Initialize UI */
    if (ui_init(1280, 720, &g_renderer) != 0) {
        fprintf(stderr, "Failed to initialize UI\n");
        renderer_shutdown(&g_renderer);
        return 1;
    }

    /* Start render thread */
    ct_thread_t render_thread;
    #ifdef _WIN32
    render_thread = CreateThread(NULL, 0, render_thread_func, NULL, 0, NULL);
    if (!render_thread) {
        fprintf(stderr, "Failed to create render thread\n");
        return 1;
    }
    #else
    if (pthread_create(&render_thread, NULL, render_thread_func, NULL) != 0) {
        fprintf(stderr, "Failed to create render thread\n");
        return 1;
    }
    #endif

    printf("CTracer running. Waiting for scene data on port %d...\n", args.port);

    /* Loop */
    while (!ui_should_close() && g_running) {
        server_poll(&g_server);

        ui_update(&g_renderer, &g_server);
    }

    printf("Shutting down...\n");
    g_running = 0;

    #ifdef _WIN32
    WaitForSingleObject(render_thread, 5000);
    CloseHandle(render_thread);
    #else
    pthread_join(render_thread, NULL);
    #endif

    server_stop(&g_server);
    ui_shutdown();
    renderer_shutdown(&g_renderer);

    printf("CTracer exited cleanly.\n");
    return 0;
}
