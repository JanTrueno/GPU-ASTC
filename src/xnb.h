// XNA/FNA .xnb content files that hold a single Texture2D (the usual case for game art).
//
// Reading: uncompressed or LZX-compressed (flag 0x80); surface format Color (RGBA8) for now.
// Writing: the same file with the pixel data replaced by ASTC blocks and an ASTC surface format,
// written uncompressed (the patched FNA reads that directly).
#pragma once

#include "png_decode.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpuastc {

// Fills `out` (pixels + meta needed to write the file back). Returns false and sets *err for
// anything that is not a plain Texture2D XNB (fonts, models, sounds, unsupported formats ...).
bool decode_xnb_memory(const std::vector<uint8_t>& file, Image& out, std::string* err = nullptr);

// Message set when a file is an XNB that already holds ASTC data (so re-running is harmless).
inline const char* xnb_already_astc() { return "already ASTC"; }

// Surface format value the patched FNA uses for ASTC blocks of the given size (4, 5, 6 or 8).
int xnb_astc_surface_format(int block);

// Builds the replacement .xnb for a texture read by decode_xnb_memory(); `meta` is Image::meta.
std::vector<uint8_t> build_astc_xnb(const std::vector<uint8_t>& meta, int block, int w, int h, const uint8_t* blocks, size_t size);

} // namespace gpuastc
