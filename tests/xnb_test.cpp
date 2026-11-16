// Decodes .xnb files and reports what happened.
// usage: xnb_test [-png outdir] files...
//   Prints a count per failure reason and the pixel total; -png writes each decoded texture as an
//   (uncompressed) PNG so it can be looked at.
#include "png_decode.h"
#include "xnb.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static uint32_t crc_table[256];
static void init_crc()
{
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
}
static uint32_t crc(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu)
{
    for (size_t i = 0; i < n; ++i) c = crc_table[(c ^ p[i]) & 255] ^ (c >> 8);
    return c;
}
static void be32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 3; i >= 0; --i) v.push_back((uint8_t)(x >> (8 * i))); }
static void chunk(std::vector<uint8_t>& o, const char* type, const std::vector<uint8_t>& d)
{
    be32(o, (uint32_t)d.size());
    std::vector<uint8_t> t(type, type + 4);
    t.insert(t.end(), d.begin(), d.end());
    o.insert(o.end(), t.begin(), t.end());
    be32(o, crc(t.data(), t.size()) ^ 0xFFFFFFFFu);
}
static bool write_png(const char* path, const gpuastc::Image& im)
{
    std::vector<uint8_t> raw;
    for (int y = 0; y < im.h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), im.rgba.begin() + (size_t)y * im.w * 4, im.rgba.begin() + (size_t)(y + 1) * im.w * 4);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t p = 0; p < raw.size(); p += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - p);
        z.push_back(p + n >= raw.size() ? 1 : 0);
        z.push_back((uint8_t)n); z.push_back((uint8_t)(n >> 8)); z.push_back((uint8_t)~n); z.push_back((uint8_t)(~n >> 8));
        z.insert(z.end(), raw.begin() + p, raw.begin() + p + n);
    }
    be32(z, (b << 16) | a);
    std::vector<uint8_t> o = {0x89, 'P', 'N', 'G', 13, 10, 26, 10}, ihdr;
    be32(ihdr, im.w); be32(ihdr, im.h);
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});
    chunk(o, "IHDR", ihdr);
    chunk(o, "IDAT", z);
    chunk(o, "IEND", {});
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fwrite(o.data(), 1, o.size(), f);
    std::fclose(f);
    return true;
}

int main(int argc, char** argv)
{
    init_crc();
    const char* png_dir = nullptr;
    int i = 1;
    if (argc > 2 && !std::strcmp(argv[1], "-png")) { png_dir = argv[2]; i = 3; }
    std::map<std::string, int> reasons;
    int ok = 0, n = 0;
    long long px = 0;
    double ms = 0;
    for (; i < argc; ++i, ++n) {
        std::vector<uint8_t> file;
        if (!gpuastc::read_file(argv[i], file)) { ++reasons["unreadable"]; continue; }
        gpuastc::Image im;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        const bool good = gpuastc::decode_xnb_memory(file, im, &err);
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!good) { ++reasons[err]; if (std::getenv("XNB_FAILS")) std::printf("FAIL %s %zu\n", argv[i], file.size()); continue; }
        ++ok;
        px += (long long)im.w * im.h;
        if (png_dir) {
            std::string name = argv[i];
            name = name.substr(name.find_last_of("/\\") + 1);
            write_png((std::string(png_dir) + "/" + name + ".png").c_str(), im);
        }
    }
    std::printf("%d of %d decoded, %.2f Mpx, %.0f ms\n", ok, n, px / 1e6, ms);
    for (auto& kv : reasons) std::printf("  %5d  %s\n", kv.second, kv.first.c_str());
    return 0;
}
