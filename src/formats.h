// Input format registry. To support another source format, write a function
//     bool decode_xyz(const std::vector<uint8_t>& file, Image& out, std::string* err)
// that fills Image::w/h/alpha/rgba (and optionally premultiplied/meta), and add one line to the
// table in formats.cpp. Everything after that (batching, GPU encode, archives) is format independent.
#pragma once

#include "png_decode.h"

#include <string>
#include <vector>

namespace gpuastc {

struct InputFormat {
    const char* ext; // lower case, with the dot
    bool (*decode)(const std::vector<uint8_t>& file, Image& out, std::string* err);
    bool writes_same_container; // true: output is the same container type with ASTC inside (.xnb)
};

// The format for a file name (by extension, case-insensitive), or nullptr if not supported.
const InputFormat* input_format_for(const std::string& path);

} // namespace gpuastc
