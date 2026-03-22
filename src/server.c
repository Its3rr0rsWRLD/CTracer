#define _CRT_SECURE_NO_WARNINGS
#include "server.h"
#include <stdio.h>
#include <string.h>
#include <mongoose.h>

/* Cache scene */
static WorldSettings s_cached_world;
static bool s_world_cached = false;

/* Formatting */
static int format_frame_response(char *buf, int bufsize, int width, int height, int sample, int seq) {
    return snprintf(buf, bufsize,
        "{\"type\":\"frame\",\"width\":%d,\"height\":%d,\"sample\":%d,\"seq\":%d}",
        width, height, sample, seq);
}

static int format_scene_response(char *buf, int bufsize, int objects, int lights) {
    return snprintf(buf, bufsize,
        "{\"status\":\"ok\",\"objects\":%d,\"lights\":%d}",
        objects, lights);
}

static int format_health_response(char *buf, int bufsize) {
    return snprintf(buf, bufsize,
        "{\"status\":\"ok\",\"device\":\"cuda\",\"cuda\":true}");
}

static int format_reset_response(char *buf, int bufsize) {
    return snprintf(buf, bufsize, "{\"status\":\"reset\"}");
}

static int format_error_response(char *buf, int bufsize, const char *msg) {
    return snprintf(buf, bufsize, "{\"error\":\"%s\"}", msg);
}

/* Handle Requests */
static void http_handler(struct mg_connection *c, int ev, void *ev_data) {
    if (ev != MG_EV_HTTP_MSG) return;

    struct mg_http_message *hm = (struct mg_http_message *)ev_data;
    ServerState *srv = (ServerState *)c->fn_data;
    char resp[512];
    int resp_len;

    /* POST /render */
    if (mg_match(hm->uri, mg_str("/render"), NULL) && mg_match(hm->method, mg_str("POST"), NULL)) {
        ParsedScene ps;
        if (!scene_parse_request(hm->body.buf, hm->body.len, &ps)) {
            resp_len = format_error_response(resp, sizeof(resp), "Invalid JSON");
            mg_http_reply(c, 400, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);
            return;
        }

        /* Push scene to scene queue */
        if (ps.has_scene) {
            s_cached_world = ps.world;
            s_world_cached = true;
            scene_queue_push(srv->scene_queue, &ps);
            srv->last_object_count = ps.num_objects;
            srv->last_light_count = ps.num_lights;
        }

        /* Push camera to render queue */
        if (ps.has_camera) {
            /* Attach cached world if this request didn't include world */
            if (!ps.has_scene && s_world_cached) {
                ps.world = s_cached_world;
            }
            scene_queue_push(srv->render_queue, &ps);
        }

        /* Wait for render */
        int timeout_ms = 5000;
        int waited = 0;
        while (!ct_atomic_load(&srv->response_ready) && waited < timeout_ms) {
            #ifdef _WIN32
            Sleep(1);
            #else
            usleep(1000);
            #endif
            waited++;
        }

        if (ct_atomic_load(&srv->response_ready)) {
            resp_len = format_frame_response(resp, sizeof(resp),
                srv->response_width, srv->response_height,
                srv->response_sample, srv->response_seq);
            ct_atomic_store(&srv->response_ready, 0);
        } else {
            resp_len = format_frame_response(resp, sizeof(resp),
                srv->renderer->settings.width, srv->renderer->settings.height,
                srv->renderer->sample_count, ps.seq);
        }

        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);

        parsed_scene_free(&ps);
        return;
    }

    /* POST /scene */
    if (mg_match(hm->uri, mg_str("/scene"), NULL) && mg_match(hm->method, mg_str("POST"), NULL)) {
        ParsedScene ps;
        if (!scene_parse_request(hm->body.buf, hm->body.len, &ps)) {
            resp_len = format_error_response(resp, sizeof(resp), "Invalid JSON");
            mg_http_reply(c, 400, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);
            return;
        }

        s_cached_world = ps.world;
        s_world_cached = true;
        srv->last_object_count = ps.num_objects;
        srv->last_light_count = ps.num_lights;

        #ifdef _WIN32
        {
            SYSTEMTIME st;
            GetLocalTime(&st);
            srv->last_scene_time = st.wHour * 3600.0 + st.wMinute * 60.0 + st.wSecond + st.wMilliseconds / 1000.0;
        }
        #else
        {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            srv->last_scene_time = ts.tv_sec + ts.tv_nsec / 1e9;
        }
        #endif

        scene_queue_push(srv->scene_queue, &ps);

        /* Also trigger accumulation reset (I should probably use a different method, this isn't clean) */
        renderer_reset_accumulation(srv->renderer);

        resp_len = format_scene_response(resp, sizeof(resp), ps.num_objects, ps.num_lights);
        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);

        parsed_scene_free(&ps);
        return;
    }

    /* GET /health */
    if (mg_match(hm->uri, mg_str("/health"), NULL)) {
        resp_len = format_health_response(resp, sizeof(resp));
        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);
        return;
    }

    /* POST /reset */
    if (mg_match(hm->uri, mg_str("/reset"), NULL) && mg_match(hm->method, mg_str("POST"), NULL)) {
        renderer_reset_accumulation(srv->renderer);
        resp_len = format_reset_response(resp, sizeof(resp));
        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%.*s", resp_len, resp);
        return;
    }

    /* 404 */
    mg_http_reply(c, 404, "", "Not found");
}

/* Initialize the server */
void server_init(ServerState *srv, int port, SceneQueue *scene_q, SceneQueue *render_q, RendererCtx *ctx) {
    memset(srv, 0, sizeof(ServerState));
    srv->port = port;
    srv->scene_queue = scene_q;
    srv->render_queue = render_q;
    srv->renderer = ctx;
    srv->running = 1;

    world_settings_defaults(&s_cached_world);

    struct mg_mgr *mgr = (struct mg_mgr *)calloc(1, sizeof(struct mg_mgr));
    mg_mgr_init(mgr);

    char listen_url[64];
    snprintf(listen_url, sizeof(listen_url), "http://127.0.0.1:%d", port);

    struct mg_connection *conn = mg_http_listen(mgr, listen_url, http_handler, srv);
    if (!conn) {
        fprintf(stderr, "Failed to start HTTP server on port %d\n", port);
        free(mgr);
        return;
    }

    srv->mg_mgr = mgr;
    printf("CTracer HTTP server listening on %s\n", listen_url);
}

void server_poll(ServerState *srv) {
    if (!srv->mg_mgr || !srv->running) return;
    mg_mgr_poll((struct mg_mgr *)srv->mg_mgr, 1);
}

void server_stop(ServerState *srv) {
    srv->running = 0;
    if (srv->mg_mgr) {
        mg_mgr_free((struct mg_mgr *)srv->mg_mgr);
        free(srv->mg_mgr);
        srv->mg_mgr = NULL;
    }
}
