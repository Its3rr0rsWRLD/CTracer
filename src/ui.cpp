extern "C" {
#include "ctrace.h"
}

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>
#include <math.h>

static GLFWwindow *s_window = nullptr;
static GLuint      s_preview_tex = 0;
static int         s_tex_width = 0;
static int         s_tex_height = 0;
static double      s_last_time = 0.0;
static float       s_fps = 0.0f;
static int         s_frame_count = 0;
static double      s_fps_timer = 0.0;

static double      s_scale_timer = 0.0;
static int         s_preview_width = 0;
static int         s_preview_height = 0;

static void upload_preview_texture(const uint8_t *rgba, int width, int height) {
    if (width <= 0 || height <= 0 || !rgba) return;

    if (s_preview_tex == 0 || width != s_tex_width || height != s_tex_height) {
        if (s_preview_tex) glDeleteTextures(1, &s_preview_tex);
        glGenTextures(1, &s_preview_tex);
        glBindTexture(GL_TEXTURE_2D, s_preview_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        s_tex_width = width;
        s_tex_height = height;
    } else {
        glBindTexture(GL_TEXTURE_2D, s_preview_tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    }
}

static void update_live_scaling(RendererCtx *ctx, double now) {
    if (now - s_scale_timer < 1.0) return;
    s_scale_timer = now;

    if (s_preview_width <= 0 || s_preview_height <= 0) return;

    int new_w = ((s_preview_width + 15) / 16) * 16;
    int new_h = ((s_preview_height + 15) / 16) * 16;
    if (new_w < 16) new_w = 16;
    if (new_h < 16) new_h = 16;
    if (new_w > 3840) new_w = 3840;
    if (new_h > 2160) new_h = 2160;

    if (new_w != ctx->settings.width || new_h != ctx->settings.height) {
        ctx->settings.width = new_w;
        ctx->settings.height = new_h;
    }
}

extern "C" int ui_init(int width, int height, RendererCtx *ctx) {
    (void)ctx;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    s_window = glfwCreateWindow(1280, 720, "CTrace", nullptr, nullptr);
    if (!s_window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(s_window);
    glfwSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 2.0f;
    style.WindowBorderSize = 0.0f;
    style.FramePadding = ImVec2(6, 3);
    style.ItemSpacing = ImVec2(8, 4);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.18f, 0.20f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.25f, 1.0f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.28f, 0.32f, 1.0f);

    io.Fonts->AddFontDefault();

    ImGui_ImplGlfw_InitForOpenGL(s_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    s_last_time = glfwGetTime();
    s_fps_timer = s_last_time;
    s_scale_timer = s_last_time;

    (void)width; (void)height;

    printf("UI initialized (GLFW + OpenGL 3.3 + Dear ImGui)\n");
    return 0;
}

extern "C" void ui_update(RendererCtx *ctx, ServerState *srv) {
    glfwPollEvents();

    double now = glfwGetTime();
    s_frame_count++;
    if (now - s_fps_timer >= 1.0) {
        s_fps = (float)s_frame_count / (float)(now - s_fps_timer);
        s_frame_count = 0;
        s_fps_timer = now;
    }
    s_last_time = now;

    if (ctx->h_pinned_output && ctx->buf_width > 0 && ctx->buf_height > 0) {
        upload_preview_texture(ctx->h_pinned_output, ctx->buf_width, ctx->buf_height);
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int win_w, win_h;
    glfwGetFramebufferSize(s_window, &win_w, &win_h);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(280, (float)win_h));
    ImGui::Begin("Settings", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "CTrace");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Resolution", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Render: %d x %d", ctx->settings.width, ctx->settings.height);
        ImGui::Text("Preview: %d x %d", s_preview_width, s_preview_height);
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Auto-tracks preview size");
    }

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("SPP", &ctx->settings.spp, 1, 16);
        ImGui::SliderInt("Bounces", &ctx->settings.max_bounces, 1, 12);
        ImGui::Checkbox("Shadows", &ctx->settings.shadows);
        ImGui::Checkbox("Reflections", &ctx->settings.reflections);
        ImGui::Checkbox("GI", &ctx->settings.gi);
        ImGui::Checkbox("NEE", &ctx->settings.nee);
        ImGui::Checkbox("AO", &ctx->settings.ao);
        if (ctx->settings.ao) {
            ImGui::SliderInt("AO Samples", &ctx->settings.ao_samples, 1, 16);
            ImGui::SliderFloat("AO Radius", &ctx->settings.ao_radius, 0.5f, 20.0f);
        }
    }

    if (ImGui::CollapsingHeader("Post-Processing")) {
        ImGui::SliderFloat("Bloom", &ctx->settings.bloom_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("Bloom Thr", &ctx->settings.bloom_threshold, 0.0f, 5.0f);

        float prev_exposure = ctx->settings.exposure;
        ImGui::SliderFloat("Exposure", &ctx->settings.exposure, 0.05f, 5.0f);
        if (ctx->settings.exposure != prev_exposure) {
            ctx->ui_overrides |= UI_OVERRIDE_EXPOSURE;
        }

        float prev_gamma = ctx->settings.gamma;
        ImGui::SliderFloat("Gamma", &ctx->settings.gamma, 1.0f, 3.0f);
        if (ctx->settings.gamma != prev_gamma) {
            ctx->ui_overrides |= UI_OVERRIDE_GAMMA;
        }
    }

    if (ImGui::CollapsingHeader("Environment")) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Controlled by game");
        ImGui::Text("Sun Dir: %.2f, %.2f, %.2f",
            ctx->world.sun_dir[0], ctx->world.sun_dir[1], ctx->world.sun_dir[2]);
        ImGui::Text("Sun Int: %.2f", ctx->world.sun_intensity);
        ImGui::Text("Env Int: %.2f", ctx->world.env_intensity);
        ImGui::Text("Ambient Int: %.2f", ctx->world.ambient_intensity);
        ImGui::Text("Fog: %.4f", ctx->world.fog_density);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset Accumulation")) {
        renderer_reset_accumulation(ctx);
    }
    if (ImGui::Button("Reset Overrides")) {
        ctx->ui_overrides = 0;
        ctx->settings.exposure = ctx->world.exposure;
        ctx->settings.gamma = ctx->world.gamma;
    }

    ImGui::End();

    float panel_w = 280.0f;
    float preview_x = panel_w;
    float preview_w = (float)win_w - panel_w;
    float stats_h = 30.0f;
    float preview_h = (float)win_h - stats_h;

    ImGui::SetNextWindowPos(ImVec2(preview_x, 0));
    ImGui::SetNextWindowSize(ImVec2(preview_w, preview_h));
    ImGui::Begin("Preview", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    s_preview_width = (int)avail.x;
    s_preview_height = (int)avail.y;

    if (s_preview_tex) {
        float tex_aspect = (float)s_tex_width / (float)(s_tex_height > 0 ? s_tex_height : 1);
        float fit_w = avail.x;
        float fit_h = avail.x / tex_aspect;
        if (fit_h > avail.y) {
            fit_h = avail.y;
            fit_w = avail.y * tex_aspect;
        }
        float offset_x = (avail.x - fit_w) * 0.5f;
        float offset_y = (avail.y - fit_h) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offset_y);
        ImGui::Image((ImTextureID)(intptr_t)s_preview_tex, ImVec2(fit_w, fit_h));
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Waiting for scene...");
    }

    ImGui::End();

    update_live_scaling(ctx, now);

    ImGui::SetNextWindowPos(ImVec2(panel_w, (float)win_h - stats_h));
    ImGui::SetNextWindowSize(ImVec2(preview_w, stats_h));
    ImGui::Begin("Stats", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar);

    ImVec4 fps_color = (s_fps >= 10.0f) ? ImVec4(0.3f, 1.0f, 0.4f, 1.0f) : (s_fps >= 2.0f)  ? ImVec4(1.0f, 0.7f, 0.0f, 1.0f) : ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    ImGui::TextColored(fps_color, "%.1f FPS", s_fps);
    ImGui::SameLine(0, 20);

    ImGui::Text("%dx%d", ctx->buf_width, ctx->buf_height);
    ImGui::SameLine(0, 20);
    ImGui::Text("Sample %d", ctx->sample_count);
    ImGui::SameLine(0, 20);
    ImGui::Text("%.1f ms", ctx->last_render_ms);
    ImGui::SameLine(0, 20);
    ImGui::Text("%.0f MB GPU", ctx->gpu_memory_used_mb);
    ImGui::SameLine(0, 20);

    ImGui::Text("Port %d", srv ? srv->port : 0);
    ImGui::SameLine(0, 10);
    ImGui::Text("Obj: %d  Lights: %d",
        srv ? srv->last_object_count : 0,
        srv ? srv->last_light_count : 0);

    ImGui::End();

    ImGui::Render();
    glViewport(0, 0, win_w, win_h);
    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(s_window);
}

extern "C" int ui_should_close(void) {
    return s_window ? glfwWindowShouldClose(s_window) : 1;
}

extern "C" int ui_get_preview_width(void) {
    return s_preview_width;
}

extern "C" int ui_get_preview_height(void) {
    return s_preview_height;
}

extern "C" void ui_shutdown(void) {
    if (s_preview_tex) {
        glDeleteTextures(1, &s_preview_tex);
        s_preview_tex = 0;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (s_window) {
        glfwDestroyWindow(s_window);
        s_window = nullptr;
    }
    glfwTerminate();
    printf("UI shutdown complete\n");
}
