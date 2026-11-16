// GPU-ASTC: ASTC LDR encoder running as an OpenGL ES 3.1 compute shader.
// Requires a current GLES 3.1 context. No windowing code in here.
#pragma once

#include <GLES3/gl31.h>
#include <cstdint>
#include <string>

namespace gpuastc {

struct Options {
    int block = 4;           // 4 (4x4) or 6 (6x6)
    bool alpha = false;      // encode alpha (RGBA endpoints)
    bool normal_map = false; // force B and A to 1.0
};

class Encoder {
public:
    ~Encoder();

    // Compiles the compute shader variant for these options.
    bool create(const Options& opt, std::string* log = nullptr);
    void destroy();

    // Encodes an RGBA8 texture. out_buffer must be a GL buffer; it is
    // (re)allocated to blocks_x * blocks_y * 16 bytes and filled via SSBO binding 0.
    // Asynchronous: issue glMemoryBarrier / glFinish before reading it.
    bool encode(GLuint rgba8_texture, int width, int height, GLuint out_buffer);

    static int blocks(int texels, int block) { return (texels + block - 1) / block; }
    static GLenum internal_format(int block); // GL_COMPRESSED_RGBA_ASTC_*_KHR

private:
    GLuint program_ = 0;
    GLint loc_tex_size_ = -1;
    GLint loc_blocks_ = -1;
    Options opt_;
};

} // namespace gpuastc
