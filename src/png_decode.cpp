#include "png_decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#ifdef GPUASTC_HAVE_LIBDEFLATE
#include <libdeflate.h>
#endif

namespace gpuastc {

bool read_file(const char* path, std::vector<uint8_t>& out)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); return false; }
    out.resize((size_t)n);
    bool ok = std::fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    std::fclose(f);
    return ok;
}

static void finish_alpha(Image& im)
{
    im.alpha = false;
    const uint8_t* p = im.rgba.data();
    const size_t n = (size_t)im.w * (size_t)im.h;
    for (size_t i = 0; i < n; ++i) {
        if (p[i * 4 + 3] != 255) { im.alpha = true; return; }
    }
}

bool decode_png_stb(const std::vector<uint8_t>& file, Image& out, std::string* err)
{
    int w = 0, h = 0, c = 0;
    stbi_uc* px = stbi_load_from_memory(file.data(), (int)file.size(), &w, &h, &c, 4);
    if (!px) {
        if (err) *err = stbi_failure_reason() ? stbi_failure_reason() : "decode failed";
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgba.assign(px, px + (size_t)w * (size_t)h * 4);
    stbi_image_free(px);
    finish_alpha(out);
    return true;
}

#ifdef GPUASTC_HAVE_LIBDEFLATE

namespace {

struct Inflater {
    libdeflate_decompressor* d;
    Inflater() : d(libdeflate_alloc_decompressor()) {}
    ~Inflater() { if (d) libdeflate_free_decompressor(d); }
};

inline uint32_t be32(const uint8_t* p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

inline uint32_t be16(const uint8_t* p)
{
    return ((uint32_t)p[0] << 8) | (uint32_t)p[1];
}

// Paeth predictor. p - a = b - c, p - b = a - c, p - c = (a - c) + (b - c).
inline uint8_t paeth(int a, int b, int c)
{
    const int pa = std::abs(b - c);
    const int pb = std::abs(a - c);
    const int pc = std::abs(a + b - c - c);
    return (uint8_t)((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
}

// Undo the PNG filter of one row in place (cur and prev both point past the filter byte).
bool unfilter_row(uint8_t ft, uint8_t* cur, const uint8_t* prev, size_t stride, size_t bpp)
{
    switch (ft) {
    case 0:
        return true;
    case 1:
        for (size_t i = bpp; i < stride; ++i) cur[i] = (uint8_t)(cur[i] + cur[i - bpp]);
        return true;
    case 2:
        for (size_t i = 0; i < stride; ++i) cur[i] = (uint8_t)(cur[i] + prev[i]);
        return true;
    case 3:
        for (size_t i = 0; i < bpp && i < stride; ++i) cur[i] = (uint8_t)(cur[i] + (prev[i] >> 1));
        for (size_t i = bpp; i < stride; ++i)
            cur[i] = (uint8_t)(cur[i] + (((int)cur[i - bpp] + (int)prev[i]) >> 1));
        return true;
    case 4: {
        for (size_t i = 0; i < bpp && i < stride; ++i) cur[i] = (uint8_t)(cur[i] + prev[i]);
        if (bpp == 3) {
            // keep the left and upper-left pixels in registers
            int a0 = cur[0], a1 = cur[1], a2 = cur[2];
            int c0 = prev[0], c1 = prev[1], c2 = prev[2];
            for (size_t i = 3; i + 3 <= stride; i += 3) {
                const int b0 = prev[i], b1 = prev[i + 1], b2 = prev[i + 2];
                a0 = cur[i] = (uint8_t)(cur[i] + paeth(a0, b0, c0));
                a1 = cur[i + 1] = (uint8_t)(cur[i + 1] + paeth(a1, b1, c1));
                a2 = cur[i + 2] = (uint8_t)(cur[i + 2] + paeth(a2, b2, c2));
                c0 = b0; c1 = b1; c2 = b2;
            }
        } else if (bpp == 4) {
            int a0 = cur[0], a1 = cur[1], a2 = cur[2], a3 = cur[3];
            int c0 = prev[0], c1 = prev[1], c2 = prev[2], c3 = prev[3];
            for (size_t i = 4; i + 4 <= stride; i += 4) {
                const int b0 = prev[i], b1 = prev[i + 1], b2 = prev[i + 2], b3 = prev[i + 3];
                a0 = cur[i] = (uint8_t)(cur[i] + paeth(a0, b0, c0));
                a1 = cur[i + 1] = (uint8_t)(cur[i + 1] + paeth(a1, b1, c1));
                a2 = cur[i + 2] = (uint8_t)(cur[i + 2] + paeth(a2, b2, c2));
                a3 = cur[i + 3] = (uint8_t)(cur[i + 3] + paeth(a3, b3, c3));
                c0 = b0; c1 = b1; c2 = b2; c3 = b3;
            }
        } else {
            for (size_t i = bpp; i < stride; ++i)
                cur[i] = (uint8_t)(cur[i] + paeth(cur[i - bpp], prev[i], prev[i - bpp]));
        }
        return true;
    }
    default:
        return false;
    }
}

struct Fmt {
    int w, h, ct, bd;
    const uint8_t* plte; int nplte;
    const uint8_t* trns; size_t ntrns;
};

// Converts one unfiltered scanline to RGBA8.
void convert_row(const uint8_t* cur, const Fmt& f, uint8_t* dst)
{
    const int w = f.w;
    // transparency colour key (grey / RGB images)
    const bool key_grey = f.ct == 0 && f.ntrns >= 2;
    const bool key_rgb = f.ct == 2 && f.ntrns >= 6;
    const uint32_t k0 = f.ntrns >= 2 ? be16(f.trns) : 0;
    const uint32_t k1 = f.ntrns >= 4 ? be16(f.trns + 2) : 0;
    const uint32_t k2 = f.ntrns >= 6 ? be16(f.trns + 4) : 0;

    if (f.bd == 8) {
        switch (f.ct) {
        case 6:
            std::memcpy(dst, cur, (size_t)w * 4);
            break;
        case 2:
            for (int x = 0; x < w; ++x) {
                const uint8_t r = cur[x * 3], g = cur[x * 3 + 1], b = cur[x * 3 + 2];
                dst[x * 4] = r; dst[x * 4 + 1] = g; dst[x * 4 + 2] = b;
                dst[x * 4 + 3] = (key_rgb && r == k0 && g == k1 && b == k2) ? 0 : 255;
            }
            break;
        case 0:
            for (int x = 0; x < w; ++x) {
                const uint8_t g = cur[x];
                dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = g;
                dst[x * 4 + 3] = (key_grey && g == k0) ? 0 : 255;
            }
            break;
        case 4:
            for (int x = 0; x < w; ++x) {
                const uint8_t g = cur[x * 2];
                dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = g;
                dst[x * 4 + 3] = cur[x * 2 + 1];
            }
            break;
        case 3:
            for (int x = 0; x < w; ++x) {
                const int idx = cur[x];
                if (idx < f.nplte) {
                    dst[x * 4] = f.plte[idx * 3]; dst[x * 4 + 1] = f.plte[idx * 3 + 1]; dst[x * 4 + 2] = f.plte[idx * 3 + 2];
                } else {
                    dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = 0;
                }
                dst[x * 4 + 3] = ((size_t)idx < f.ntrns) ? f.trns[idx] : 255;
            }
            break;
        }
    } else if (f.bd == 16) {
        // keep the high byte of each sample (what stb_image does)
        switch (f.ct) {
        case 6:
            for (int x = 0; x < w; ++x) {
                dst[x * 4] = cur[x * 8]; dst[x * 4 + 1] = cur[x * 8 + 2];
                dst[x * 4 + 2] = cur[x * 8 + 4]; dst[x * 4 + 3] = cur[x * 8 + 6];
            }
            break;
        case 2:
            for (int x = 0; x < w; ++x) {
                const uint8_t* p = cur + x * 6;
                dst[x * 4] = p[0]; dst[x * 4 + 1] = p[2]; dst[x * 4 + 2] = p[4];
                dst[x * 4 + 3] = (key_rgb && be16(p) == k0 && be16(p + 2) == k1 && be16(p + 4) == k2) ? 0 : 255;
            }
            break;
        case 0:
            for (int x = 0; x < w; ++x) {
                const uint8_t* p = cur + x * 2;
                dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = p[0];
                dst[x * 4 + 3] = (key_grey && be16(p) == k0) ? 0 : 255;
            }
            break;
        case 4:
            for (int x = 0; x < w; ++x) {
                const uint8_t* p = cur + x * 4;
                dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = p[0];
                dst[x * 4 + 3] = p[2];
            }
            break;
        }
    } else {
        // 1, 2 or 4 bit grey or palette
        const int bd = f.bd;
        const int mask = (1 << bd) - 1;
        for (int x = 0; x < w; ++x) {
            const int bitpos = x * bd;
            const int sh = 8 - bd - (bitpos & 7);
            const int v = (cur[bitpos >> 3] >> sh) & mask;
            if (f.ct == 0) {
                const uint8_t g = (uint8_t)(v * 255 / mask);
                dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = g;
                dst[x * 4 + 3] = (key_grey && (uint32_t)v == k0) ? 0 : 255;
            } else {
                if (v < f.nplte) {
                    dst[x * 4] = f.plte[v * 3]; dst[x * 4 + 1] = f.plte[v * 3 + 1]; dst[x * 4 + 2] = f.plte[v * 3 + 2];
                } else {
                    dst[x * 4] = dst[x * 4 + 1] = dst[x * 4 + 2] = 0;
                }
                dst[x * 4 + 3] = ((size_t)v < f.ntrns) ? f.trns[v] : 255;
            }
        }
    }
}

} // namespace

bool decode_png_fast(const std::vector<uint8_t>& file, Image& out)
{
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (file.size() < 8 + 25 || std::memcmp(file.data(), sig, 8) != 0)
        return false;

    uint32_t w = 0, h = 0;
    int bd = 0, ct = 0, interlace = 0;
    bool have_ihdr = false;
    uint8_t plte[768];
    int nplte = 0;
    uint8_t trns[256];
    size_t ntrns = 0;
    std::vector<uint8_t> z;

    size_t pos = 8;
    while (pos + 12 <= file.size()) {
        const uint32_t len = be32(&file[pos]);
        const uint8_t* type = &file[pos + 4];
        const uint8_t* data = &file[pos + 8];
        if (len > file.size() - pos - 12)
            return false;
        if (!std::memcmp(type, "IHDR", 4)) {
            if (len != 13) return false;
            w = be32(data);
            h = be32(data + 4);
            bd = data[8];
            ct = data[9];
            if (data[10] != 0 || data[11] != 0) return false;
            interlace = data[12];
            have_ihdr = true;
        } else if (!std::memcmp(type, "PLTE", 4)) {
            if (len % 3 != 0 || len > 768) return false;
            std::memcpy(plte, data, len);
            nplte = (int)(len / 3);
        } else if (!std::memcmp(type, "tRNS", 4)) {
            ntrns = len < 256 ? len : 256;
            std::memcpy(trns, data, ntrns);
        } else if (!std::memcmp(type, "IDAT", 4)) {
            z.insert(z.end(), data, data + len);
        } else if (!std::memcmp(type, "IEND", 4)) {
            break;
        }
        pos += 12 + (size_t)len;
    }

    if (!have_ihdr || w == 0 || h == 0 || w > 65535 || h > 65535 || interlace != 0 || z.empty())
        return false;

    int ch = 0;
    bool fmt_ok = false;
    switch (ct) {
    case 0: ch = 1; fmt_ok = bd == 1 || bd == 2 || bd == 4 || bd == 8 || bd == 16; break;
    case 2: ch = 3; fmt_ok = bd == 8 || bd == 16; break;
    case 3: ch = 1; fmt_ok = (bd == 1 || bd == 2 || bd == 4 || bd == 8) && nplte > 0; break;
    case 4: ch = 2; fmt_ok = bd == 8 || bd == 16; break;
    case 6: ch = 4; fmt_ok = bd == 8 || bd == 16; break;
    }
    if (!fmt_ok)
        return false;

    const size_t bits_pp = (size_t)ch * (size_t)bd;
    const size_t bpp = (bits_pp + 7) / 8;
    const size_t stride = ((size_t)w * bits_pp + 7) / 8;
    const uint64_t raw_size = (uint64_t)(stride + 1) * h;
    if (raw_size > (1ull << 30))
        return false;

    static thread_local Inflater inf;
    if (!inf.d)
        return false;
    std::vector<uint8_t> raw((size_t)raw_size);
    size_t actual = 0;
    if (libdeflate_zlib_decompress(inf.d, z.data(), z.size(), raw.data(), raw.size(), &actual) != LIBDEFLATE_SUCCESS ||
        actual != raw.size())
        return false;

    out.w = (int)w;
    out.h = (int)h;
    out.rgba.resize((size_t)w * h * 4);
    const Fmt f = {(int)w, (int)h, ct, bd, plte, nplte, trns, ntrns};
    // unfilter and convert one row at a time so the row is still in cache for the conversion
    std::vector<uint8_t> zero(stride, 0);
    const uint8_t* prev = zero.data();
    for (uint32_t y = 0; y < h; ++y) {
        uint8_t* row = raw.data() + (size_t)y * (stride + 1);
        if (!unfilter_row(row[0], row + 1, prev, stride, bpp))
            return false;
        convert_row(row + 1, f, out.rgba.data() + (size_t)y * w * 4);
        prev = row + 1;
    }
    finish_alpha(out);
    return true;
}

#else // !GPUASTC_HAVE_LIBDEFLATE

bool decode_png_fast(const std::vector<uint8_t>&, Image&)
{
    return false;
}

#endif

bool decode_png(const char* path, Image& out, std::string* err)
{
    std::vector<uint8_t> file;
    if (!read_file(path, file)) {
        if (err) *err = "cannot read file";
        return false;
    }
    if (decode_png_fast(file, out))
        return true;
    return decode_png_stb(file, out, err);
}

} // namespace gpuastc
