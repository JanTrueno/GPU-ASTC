// gpuastc: encode a PNG to .astc using a GLES 3.1 compute shader.
// The GLES context comes from SDL2, so it works on KMSDRM/offscreen with no window system.
#include "astc_gles.h"

#include <SDL.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage()
{
    std::fprintf(stderr,
        "usage: gpuastc input.png output.astc [options]\n"
        "  -b 4|5|6          block size (default 4)\n"
        "  -alpha auto|on|off  encode alpha (default auto: on if any alpha < 255)\n"
        "  -norm             normal map (B and A forced to 1)\n"
        "  -upload           also upload the result into an ASTC texture on the GPU (self-test)\n"
        "  -v                verbose\n");
}

static bool has_extension(const char* name)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; ++i) {
        const char* e = (const char*)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && std::strcmp(e, name) == 0)
            return true;
    }
    return false;
}

static bool save_astc(const char* path, int bdim, int w, int h, const void* data, size_t size)
{
    uint8_t hdr[16] = {0x13, 0xAB, 0xA1, 0x5C, (uint8_t)bdim, (uint8_t)bdim, 1,
                       (uint8_t)w, (uint8_t)(w >> 8), (uint8_t)(w >> 16),
                       (uint8_t)h, (uint8_t)(h >> 8), (uint8_t)(h >> 16), 1, 0, 0};
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr) &&
              std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    return ok;
}

int main(int argc, char** argv)
{
    if (argc < 3) { usage(); return 1; }

    gpuastc::Options opt;
    int alpha_mode = 0; // 0 auto, 1 on, 2 off
    bool verbose = false, upload = false;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-b") && i + 1 < argc) opt.block = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-alpha") && i + 1 < argc) {
            const char* v = argv[++i];
            alpha_mode = !std::strcmp(v, "on") ? 1 : !std::strcmp(v, "off") ? 2 : 0;
        }
        else if (!std::strcmp(argv[i], "-norm")) opt.normal_map = true;
        else if (!std::strcmp(argv[i], "-upload")) upload = true;
        else if (!std::strcmp(argv[i], "-v")) verbose = true;
        else { usage(); return 1; }
    }
    if (opt.block != 4 && opt.block != 6) { std::fprintf(stderr, "block must be 4 or 6\n"); return 1; }

    // Load as RGBA8: grey, grey+alpha, RGB, palette and 16-bit PNGs are all converted.
    int w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load(argv[1], &w, &h, &comp, 4);
    if (!pixels) { std::fprintf(stderr, "cannot load %s: %s\n", argv[1], stbi_failure_reason()); return 1; }

    bool any_alpha = false;
    for (size_t i = 3; i < (size_t)w * h * 4; i += 4)
        if (pixels[i] != 255) { any_alpha = true; break; }
    opt.alpha = alpha_mode == 1 || (alpha_mode == 0 && any_alpha);
    if (verbose)
        std::printf("%dx%d, %d source channels, alpha %s\n", w, h, comp, opt.alpha ? "on" : "off");

    // GLES 3.1 context through SDL2 (hidden window; works with offscreen/KMSDRM drivers).
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_Window* win = SDL_CreateWindow("gpuastc", 0, 0, 64, 64, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) { std::fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) { std::fprintf(stderr, "SDL_GL_CreateContext (GLES 3.1): %s\n", SDL_GetError()); return 1; }
    if (verbose)
        std::printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    gpuastc::Encoder enc;
    std::string log;
    if (!enc.create(opt, &log)) { std::fprintf(stderr, "shader error:\n%s\n", log.c_str()); return 1; }

    GLuint tex = 0, buf = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    stbi_image_free(pixels);

    glGenBuffers(1, &buf);
    Uint64 t0 = SDL_GetPerformanceCounter();
    if (!enc.encode(tex, w, h, buf)) { std::fprintf(stderr, "dispatch failed (GL error 0x%x)\n", glGetError()); return 1; }
    glFinish();
    double ms = 1000.0 * (double)(SDL_GetPerformanceCounter() - t0) / (double)SDL_GetPerformanceFrequency();

    const int bx = gpuastc::Encoder::blocks(w, opt.block), by = gpuastc::Encoder::blocks(h, opt.block);
    const size_t size = (size_t)bx * by * 16;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    const void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)size, GL_MAP_READ_BIT);
    if (!mapped) { std::fprintf(stderr, "map failed (GL error 0x%x)\n", glGetError()); return 1; }
    bool saved = save_astc(argv[2], opt.block, w, h, mapped, size);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    if (!saved) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    std::printf("%s: %dx%d -> %dx%d blocks (%zu bytes) in %.2f ms\n", argv[2], w, h, bx, by, size, ms);

    if (upload) {
        // Zero-copy path for real use: SSBO -> PBO unpack -> ASTC texture.
        if (!has_extension("GL_KHR_texture_compression_astc_ldr")) {
            std::fprintf(stderr, "upload: GL_KHR_texture_compression_astc_ldr not available\n");
        } else {
            GLuint ctex = 0;
            glGenTextures(1, &ctex);
            glBindTexture(GL_TEXTURE_2D, ctex);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf);
            glCompressedTexImage2D(GL_TEXTURE_2D, 0, gpuastc::Encoder::internal_format(opt.block),
                                   w, h, 0, (GLsizei)size, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            GLenum err = glGetError();
            std::printf("upload to ASTC texture: %s\n", err == GL_NO_ERROR ? "ok" : "failed");
            if (err != GL_NO_ERROR) std::printf("  GL error 0x%x\n", err);
            glDeleteTextures(1, &ctex);
        }
    }

    glDeleteBuffers(1, &buf);
    glDeleteTextures(1, &tex);
    enc.destroy();
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
