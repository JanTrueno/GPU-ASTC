// PNG loading for gpuastc.
//
// decode_png() uses a fast libdeflate-based decoder when the project is built with
// libdeflate (GPUASTC_HAVE_LIBDEFLATE) and falls back to stb_image for everything else
// (interlaced files, unusual headers, or when libdeflate is not available).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpuastc {

struct Image {
    int w = 0, h = 0;
    bool alpha = false;          // true if any pixel has alpha < 255
    std::vector<uint8_t> rgba;   // w * h * 4, row-major, top row first
    bool premultiplied = false;  // colour already multiplied by alpha (XNA content)
    std::vector<uint8_t> meta;   // source-format data a writer needs (e.g. the XNB header)
};

bool read_file(const char* path, std::vector<uint8_t>& out);

// Any PNG -> RGBA8 (grey, grey+alpha, RGB, RGBA, palette, 1-16 bit, tRNS).
bool decode_png(const char* path, Image& out, std::string* err = nullptr);

// Same as decode_png() for a file that is already in memory.
bool decode_png_memory(const std::vector<uint8_t>& file, Image& out, std::string* err = nullptr);

// The two back ends, exposed so tests can compare them.
bool decode_png_stb(const std::vector<uint8_t>& file, Image& out, std::string* err = nullptr);
// Returns false if the file is not handled by the fast path (or it was not built).
bool decode_png_fast(const std::vector<uint8_t>& file, Image& out);

} // namespace gpuastc
