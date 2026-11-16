#include "formats.h"
#include "xnb.h"

#include <cctype>
#include <cstring>

namespace gpuastc {
namespace {

bool decode_png_entry(const std::vector<uint8_t>& file, Image& out, std::string* err)
{
    return decode_png_memory(file, out, err);
}

const InputFormat kFormats[] = {
    {".png", decode_png_entry, false},
    {".xnb", decode_xnb_memory, true},
};

} // namespace

const InputFormat* input_format_for(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return nullptr;
    std::string e = path.substr(dot);
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    for (const InputFormat& f : kFormats)
        if (e == f.ext) return &f;
    return nullptr;
}

} // namespace gpuastc
