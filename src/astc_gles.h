// GPU-ASTC: ASTC LDR encoder running as an OpenGL ES 3.1 compute shader.
// Requires a current GLES 3.1 context. No windowing code in here.
#pragma once

#include <GLES3/gl31.h>
#include <cstdint>
#include <string>

namespace gpuastc {

struct Options {
    int block = 4;           // 4 (4x4), 5 (5x5) or 6 (6x6)
    bool alpha = false;      // encode alpha (RGBA endpoints)
    bool premultiplied = false; // colour already multiplied by alpha
    bool normal_map = false; // force B and A to 1.0
    std::string cache_dir;   // where compiled shaders are kept between runs; empty = no cache
    bool input_buffer = false; // the shader reads the pixels from a buffer (encode_pixels), not a texture
    int quality = 1;         // 4x4 search effort: 0 one colour line per block, 1 also two partitions
                             // for blocks that fit badly, 2 more candidates and a lower threshold
};

class Encoder {
public:
    ~Encoder();

    // Compiles the compute shader variant for these options, or loads the compiled program a
    // previous run left in Options::cache_dir (same shader source, same driver).
    bool create(const Options& opt, std::string* log = nullptr);
    bool from_cache() const { return from_cache_; }
    void destroy();

    // Encodes an RGBA8 texture. out_buffer must be a GL buffer; it is
    // (re)allocated to blocks_x * blocks_y * 16 bytes and filled via SSBO binding 0.
    // Asynchronous: issue glMemoryBarrier / glFinish before reading it.
    bool encode(GLuint rgba8_texture, int width, int height, GLuint out_buffer);

    // The same from RGBA8 pixels in memory (Options::input_buffer). The pixels are copied into a
    // buffer the shader reads, which skips the driver's texture upload.
    bool encode_pixels(const uint8_t* rgba, int width, int height, GLuint out_buffer);

    // Whether a width x height image fits the buffer path on this GL context.
    static bool buffer_input_fits(int width, int height);

    static int blocks(int texels, int block) { return (texels + block - 1) / block; }
    static GLenum internal_format(int block); // GL_COMPRESSED_RGBA_ASTC_*_KHR

    // Chooses a block size (4, 5 or 6) from the texture size:
    //  - 4x4 for textures up to 256x256, thin textures (shortest side <= 64, e.g. 1000x16)
    //    and anything with alpha (block edges hurt alpha the most)
    //  - 5x5 up to 2048x2048, 6x6 above that
    static int pick_block(int width, int height, bool alpha);

private:
    GLuint program_ = 0;
    GLuint in_buffer_ = 0;
    bool dispatch(int width, int height, GLuint out_buffer);
    bool from_cache_ = false;
    GLint loc_tex_size_ = -1;
    GLint loc_blocks_ = -1;
    GLint loc_count_ = -1;
    GLint loc_row0_ = -1;
    Options opt_;
};

} // namespace gpuastc
