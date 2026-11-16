#include "xnb.h"
#include "lzx.h"

#include <cstring>

namespace gpuastc {
namespace {

struct Reader {
    const uint8_t* p;
    size_t n, pos = 0;
    bool bad = false;
    Reader(const uint8_t* d, size_t sz) : p(d), n(sz) {}
    uint8_t u8() { if (pos >= n) { bad = true; return 0; } return p[pos++]; }
    uint32_t u32()
    {
        if (pos + 4 > n) { bad = true; pos = n; return 0; }
        const uint32_t v = p[pos] | (p[pos + 1] << 8) | (p[pos + 2] << 16) | ((uint32_t)p[pos + 3] << 24);
        pos += 4;
        return v;
    }
    uint32_t v7()
    {
        uint32_t v = 0;
        for (int s = 0; s < 35; s += 7) {
            const uint8_t c = u8();
            v |= (uint32_t)(c & 127) << s;
            if (!(c & 128) || bad) break;
        }
        return v;
    }
    std::string str()
    {
        const uint32_t l = v7();
        if (bad || pos + l > n) { bad = true; return std::string(); }
        std::string s((const char*)p + pos, l);
        pos += l;
        return s;
    }
};

void put32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back((uint8_t)(x >> (8 * i))); }

// Unpremultiply is NOT done: XNA content is premultiplied and FNA expects it that way.
} // namespace

bool decode_xnb_memory(const std::vector<uint8_t>& file, Image& out, std::string* err)
{
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (file.size() < 14 || file[0] != 'X' || file[1] != 'N' || file[2] != 'B') return fail("not an XNB file");
    const uint8_t flags = file[5];
    const uint8_t version = file[4];
    if (version < 4 && version != 3) return fail("unsupported XNB version");
    if (flags & 0x40) return fail("LZ4-compressed XNB not supported");

    std::vector<uint8_t> body;
    const uint8_t* c;
    size_t cn;
    if (flags & 0x80) {
        Reader r(file.data(), file.size());
        r.pos = 10;
        const uint32_t dsize = r.u32();
        if (r.bad || dsize > (1u << 30)) return fail("bad XNB size");
        if (!lzx_decompress_xnb(file.data() + 14, file.size() - 14, dsize, body)) return fail("LZX data is corrupt");
        c = body.data();
        cn = body.size();
    } else {
        c = file.data() + 10;
        cn = file.size() - 10;
    }

    Reader r(c, cn);
    const uint32_t nreaders = r.v7();
    if (r.bad || nreaders == 0 || nreaders > 64) return fail("bad type reader list");
    std::string first;
    for (uint32_t i = 0; i < nreaders; ++i) {
        std::string name = r.str();
        r.u32(); // version
        if (i == 0) first = name;
    }
    if (first.find("Texture2DReader") == std::string::npos) return fail("not a Texture2D");
    if (r.v7() != 0) return fail("shared resources not supported");
    const uint32_t idx = r.v7();
    if (idx != 1) return fail("unexpected root object");
    const size_t prefix_end = r.pos;
    const uint32_t fmt = r.u32();
    const uint32_t w = r.u32(), h = r.u32();
    const uint32_t mips = r.u32();
    if (r.bad || w == 0 || h == 0 || w > 16384 || h > 16384 || mips == 0) return fail("bad texture header");
    if (fmt >= 25 && fmt <= 28) return fail(xnb_already_astc());
    if (fmt != 0) return fail("surface format is not Color (RGBA8)");
    const uint32_t sz = r.u32();
    if (r.bad || sz != w * h * 4 || r.pos + sz > cn) return fail("bad pixel data size");

    out.w = (int)w;
    out.h = (int)h;
    out.rgba.assign(c + r.pos, c + r.pos + sz);
    out.alpha = false;
    for (size_t i = 3; i < out.rgba.size(); i += 4)
        if (out.rgba[i] != 255) { out.alpha = true; break; }
    out.premultiplied = true;
    // keep what is needed to write the file back: platform/version/flags and the reader table
    out.meta.assign(file.begin() + 3, file.begin() + 6);
    out.meta.insert(out.meta.end(), c, c + prefix_end);
    return true;
}

int xnb_astc_surface_format(int block)
{
    // FNA's patch appends Astc4x4EXT..Astc8x8EXT after the existing SurfaceFormat values.
    switch (block) {
        case 4: return 25;
        case 5: return 26;
        case 6: return 27;
        default: return 28;
    }
}

std::vector<uint8_t> build_astc_xnb(const std::vector<uint8_t>& meta, int block, int w, int h, const uint8_t* blocks, size_t size)
{
    std::vector<uint8_t> o;
    const uint8_t* m = meta.data();
    const size_t prefix = meta.size() - 3;
    o.reserve(meta.size() + 40 + size);
    o.push_back('X'); o.push_back('N'); o.push_back('B');
    o.push_back(m[0]);                    // platform
    o.push_back(m[1]);                    // version
    o.push_back((uint8_t)(m[2] & ~0x80)); // flags, now uncompressed
    put32(o, 0);                          // total size, patched below
    o.insert(o.end(), m + 3, m + 3 + prefix);
    put32(o, (uint32_t)xnb_astc_surface_format(block));
    put32(o, (uint32_t)w);
    put32(o, (uint32_t)h);
    put32(o, 1); // one mip level
    put32(o, (uint32_t)size);
    o.insert(o.end(), blocks, blocks + size);
    const uint32_t total = (uint32_t)o.size();
    for (int i = 0; i < 4; ++i) o[6 + i] = (uint8_t)(total >> (8 * i));
    return o;
}

} // namespace gpuastc
