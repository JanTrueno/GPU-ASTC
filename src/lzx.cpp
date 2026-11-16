#include "lzx.h"

#include <cstring>

namespace gpuastc {
namespace {

const int kWindowBits = 16; // XNA always uses a 64 KB window
const uint32_t kWindowSize = 1u << kWindowBits;
const int kPosnSlots = 32;
const int kNumChars = 256;
const int kMainElements = kNumChars + kPosnSlots * 8; // 512
const int kLengthElements = 249;
const int kAlignedElements = 8;
const int kPretreeElements = 20;
const int kMaxBits = 16;
const int kTableBits = 11;

enum { BLOCK_VERBATIM = 1, BLOCK_ALIGNED = 2, BLOCK_UNCOMPRESSED = 3 };

// Canonical Huffman decoder: direct table for codes up to kTableBits, slow path beyond.
struct Huff {
    uint16_t fast[1 << kTableBits]; // (symbol << 5) | length, 0 = not in the table
    int count[kMaxBits + 1];
    uint16_t sorted[kMainElements]; // symbols ordered by (length, symbol)
    int first_code[kMaxBits + 2];
    int first_index[kMaxBits + 2];

    bool build(const uint8_t* lens, int num)
    {
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < num; ++i) ++count[lens[i]];
        count[0] = 0;
        int offs[kMaxBits + 2];
        offs[1] = 0;
        for (int l = 1; l <= kMaxBits; ++l) offs[l + 1] = offs[l] + count[l];
        for (int i = 0; i < num; ++i)
            if (lens[i]) sorted[offs[lens[i]]++] = (uint16_t)i;
        int code = 0, idx = 0;
        for (int l = 1; l <= kMaxBits; ++l) {
            first_code[l] = code;
            first_index[l] = idx;
            code = (code + count[l]) << 1;
            idx += count[l];
        }
        std::memset(fast, 0, sizeof(fast));
        for (int l = 1; l <= kTableBits; ++l) {
            for (int k = 0; k < count[l]; ++k) {
                const int c = first_code[l] + k;
                if (c >= (1 << l)) return false; // over-subscribed
                const uint16_t sym = sorted[first_index[l] + k];
                const int shift = kTableBits - l;
                for (int f = 0; f < (1 << shift); ++f)
                    fast[(c << shift) | f] = (uint16_t)((sym << 5) | l);
            }
        }
        return true;
    }
};

struct Bits {
    const uint8_t* in = nullptr;
    size_t size = 0, ip = 0; // next word to load
    uint64_t buf = 0;        // left-aligned
    int left = 0;

    void init(const uint8_t* p, size_t n) { in = p; size = n; ip = 0; buf = 0; left = 0; }
    void refill()
    {
        while (left <= 48) {
            uint32_t w = 0;
            if (ip + 1 < size) w = in[ip] | (in[ip + 1] << 8);
            else if (ip < size) w = in[ip];
            ip += 2;
            buf |= (uint64_t)w << (48 - left);
            left += 16;
        }
    }
    uint32_t peek(int n) { if (left < n) refill(); return (uint32_t)(buf >> (64 - n)); }
    void drop(int n) { buf <<= n; left -= n; }
    uint32_t read(int n) { if (n == 0) return 0; const uint32_t v = peek(n); drop(n); return v; }
    // byte position of the next unread byte once the current word has been finished
    size_t byte_pos() const { return ip - (size_t)(left / 8); }
    void align() { drop(left & 15); }
    int decode(const Huff& h)
    {
        if (left < kMaxBits) refill();
        const uint32_t top = (uint32_t)(buf >> (64 - kMaxBits));
        const uint16_t f = h.fast[top >> (kMaxBits - kTableBits)];
        if (f) { drop(f & 31); return f >> 5; }
        for (int l = kTableBits + 1; l <= kMaxBits; ++l) {
            const int c = (int)(top >> (kMaxBits - l));
            const int k = c - h.first_code[l];
            if (k >= 0 && k < h.count[l]) { drop(l); return h.sorted[h.first_index[l] + k]; }
        }
        return -1;
    }
};

struct Lzx {
    uint8_t window[kWindowSize];
    uint32_t pos = 0;       // absolute write position (window index = pos & mask)
    uint32_t frame_pos = 0; // absolute start of the current frame
    uint32_t r0 = 1, r1 = 1, r2 = 1;
    int block_type = 0;
    uint32_t block_remaining = 0, block_length = 0;
    bool header_read = false, intel_started = false;
    uint32_t intel_filesize = 0, intel_curpos = 0, frame_no = 0;
    uint8_t main_len[kMainElements], length_len[kLengthElements + 1], aligned_len[kAlignedElements];
    Huff main_t, length_t, aligned_t;
    uint8_t extra_bits[64];
    uint32_t position_base[64];

    Lzx()
    {
        std::memset(window, 0, sizeof(window));
        std::memset(main_len, 0, sizeof(main_len));
        std::memset(length_len, 0, sizeof(length_len));
        std::memset(aligned_len, 0, sizeof(aligned_len));
        int j = 0;
        for (int i = 0; i <= 50; i += 2) {
            extra_bits[i] = extra_bits[i + 1] = (uint8_t)j;
            if (i != 0 && j < 17) ++j;
        }
        j = 0;
        for (int i = 0; i <= 50; ++i) {
            position_base[i] = (uint32_t)j;
            j += 1 << extra_bits[i];
        }
    }

    bool read_lengths(Bits& b, uint8_t* lens, int first, int last)
    {
        uint8_t pre[kPretreeElements];
        for (int i = 0; i < kPretreeElements; ++i) pre[i] = (uint8_t)b.read(4);
        Huff pt;
        if (!pt.build(pre, kPretreeElements)) return false;
        for (int x = first; x < last;) {
            int z = b.decode(pt);
            if (z < 0) return false;
            if (z == 17) {
                int y = (int)b.read(4) + 4;
                while (y-- > 0 && x < last) lens[x++] = 0;
            } else if (z == 18) {
                int y = (int)b.read(5) + 20;
                while (y-- > 0 && x < last) lens[x++] = 0;
            } else if (z == 19) {
                int y = (int)b.read(1) + 4;
                z = b.decode(pt);
                if (z < 0) return false;
                z = lens[x] - z;
                if (z < 0) z += 17;
                while (y-- > 0 && x < last) lens[x++] = (uint8_t)z;
            } else {
                z = lens[x] - z;
                if (z < 0) z += 17;
                lens[x++] = (uint8_t)z;
            }
        }
        return true;
    }

    // Decodes one frame of `out_len` bytes from a chunk. Output is the window contents just written.
    bool frame(const uint8_t* in, size_t in_len, uint32_t out_len, uint8_t* out)
    {
        Bits b;
        b.init(in, in_len);
        if (!header_read) {
            if (b.read(1)) {
                const uint32_t hi = b.read(16), lo = b.read(16);
                intel_filesize = (hi << 16) | lo;
            }
            header_read = true;
        }
        // a match may have run past the end of the previous frame: that part is already decoded
        const uint32_t end_pos = frame_pos + out_len;
        if ((int32_t)(end_pos - pos) < 0) return false;
        uint32_t togo = end_pos - pos;
        while (togo > 0) {
            if (block_remaining == 0) {
                block_type = (int)b.read(3);
                const uint32_t hi = b.read(16), lo = b.read(8);
                block_remaining = block_length = (hi << 8) | lo;
                if (block_type == BLOCK_ALIGNED) {
                    for (int i = 0; i < kAlignedElements; ++i) aligned_len[i] = (uint8_t)b.read(3);
                    if (!aligned_t.build(aligned_len, kAlignedElements)) return false;
                }
                if (block_type == BLOCK_VERBATIM || block_type == BLOCK_ALIGNED) {
                    if (!read_lengths(b, main_len, 0, kNumChars)) return false;
                    if (!read_lengths(b, main_len, kNumChars, kMainElements)) return false;
                    if (!main_t.build(main_len, kMainElements)) return false;
                    if (main_len[0xE8] != 0) intel_started = true;
                    if (!read_lengths(b, length_len, 0, kLengthElements)) return false;
                    if (!length_t.build(length_len, kLengthElements)) return false;
                } else if (block_type == BLOCK_UNCOMPRESSED) {
                    intel_started = true;
                    b.align();
                    size_t p = b.byte_pos();
                    if (p + 12 > in_len) return false;
                    r0 = in[p] | (in[p + 1] << 8) | (in[p + 2] << 16) | ((uint32_t)in[p + 3] << 24);
                    r1 = in[p + 4] | (in[p + 5] << 8) | (in[p + 6] << 16) | ((uint32_t)in[p + 7] << 24);
                    r2 = in[p + 8] | (in[p + 9] << 8) | (in[p + 10] << 16) | ((uint32_t)in[p + 11] << 24);
                    // restart the bit reader after the three repeat offsets
                    b.ip = p + 12; b.buf = 0; b.left = 0;
                } else {
                    return false;
                }
            }

            int64_t run = block_remaining < togo ? block_remaining : togo;
            togo -= (uint32_t)run;
            block_remaining -= (uint32_t)run;

            if (block_type == BLOCK_UNCOMPRESSED) {
                b.align();
                const size_t p = b.byte_pos();
                if (p + run > in_len) return false;
                // copy, wrapping at the window end is impossible because frames are window aligned
                for (int64_t k = 0; k < run; ++k) window[(pos + (uint32_t)k) & (kWindowSize - 1)] = in[p + (size_t)k];
                pos += (uint32_t)run;
                b.ip = p + (size_t)run; b.buf = 0; b.left = 0;
                if (block_remaining == 0 && (block_length & 1)) b.ip += 1; // padding byte
                continue;
            }

            while (run > 0) {
                int m = b.decode(main_t);
                if (m < 0) return false;
                if (m < kNumChars) {
                    window[pos++ & (kWindowSize - 1)] = (uint8_t)m;
                    --run;
                    continue;
                }
                m -= kNumChars;
                uint32_t len = (uint32_t)(m & 7);
                if (len == 7) {
                    const int f = b.decode(length_t);
                    if (f < 0) return false;
                    len += (uint32_t)f;
                }
                len += 2;
                uint32_t off = (uint32_t)(m >> 3);
                if (off > 2) {
                    if (block_type != BLOCK_ALIGNED) {
                        const int extra = off >= 36 ? 17 : extra_bits[off];
                        off = position_base[off] - 2 + b.read(extra);
                    } else {
                        int extra = extra_bits[off];
                        off = position_base[off] - 2;
                        if (extra > 3) {
                            extra -= 3;
                            off += b.read(extra) << 3;
                            const int a = b.decode(aligned_t);
                            if (a < 0) return false;
                            off += (uint32_t)a;
                        } else if (extra == 3) {
                            const int a = b.decode(aligned_t);
                            if (a < 0) return false;
                            off += (uint32_t)a;
                        } else if (extra > 0) {
                            off += b.read(extra);
                        } else {
                            off = 1;
                        }
                    }
                    r2 = r1; r1 = r0; r0 = off;
                } else if (off == 0) {
                    off = r0;
                } else if (off == 1) {
                    off = r1; r1 = r0; r0 = off;
                } else {
                    off = r2; r2 = r0; r0 = off;
                }
                if (off > kWindowSize) return false;
                run -= len;
                for (uint32_t k = 0; k < len; ++k) {
                    window[pos & (kWindowSize - 1)] = window[(pos - off) & (kWindowSize - 1)];
                    ++pos;
                }
            }
            if (run < 0) { // the last match overran this run: charge the excess to the block
                const uint32_t over = (uint32_t)(-run);
                if (over > block_remaining) return false;
                block_remaining -= over;
            }
            if (b.byte_pos() > in_len + 2) return false; // read past the end of the chunk
        }

        for (uint32_t k = 0; k < out_len; ++k) out[k] = window[(frame_pos + k) & (kWindowSize - 1)];
        frame_pos += out_len;
        if (intel_started && intel_filesize != 0 && frame_no < 32768 && out_len > 10) {
            // x86 call translation (E8 fix-up)
            int32_t cur = (int32_t)intel_curpos;
            const int32_t fsz = (int32_t)intel_filesize;
            uint8_t* d = out;
            uint8_t* end = out + out_len - 10;
            while (d < end) {
                if (*d++ != 0xE8) { ++cur; continue; }
                const int32_t abs_off = (int32_t)(d[0] | (d[1] << 8) | (d[2] << 16) | ((uint32_t)d[3] << 24));
                if (abs_off >= -cur && abs_off < fsz) {
                    const int32_t rel = abs_off >= 0 ? abs_off - cur : abs_off + fsz;
                    d[0] = (uint8_t)rel; d[1] = (uint8_t)(rel >> 8); d[2] = (uint8_t)(rel >> 16); d[3] = (uint8_t)(rel >> 24);
                }
                d += 4;
                cur += 5;
            }
        }
        intel_curpos += out_len;
        ++frame_no;
        return true;
    }
};

} // namespace

bool lzx_decompress_xnb(const uint8_t* in, size_t in_size, size_t out_size, std::vector<uint8_t>& out)
{
    out.assign(out_size, 0);
    Lzx* lz = new Lzx();
    size_t ip = 0, op = 0;
    bool ok = true;
    while (ip + 2 <= in_size && op < out_size) {
        uint32_t frame_size = 0x8000, block_size;
        if (in[ip] == 0xFF) {
            if (ip + 5 > in_size) { ok = false; break; }
            frame_size = (uint32_t)(in[ip + 1] << 8) | in[ip + 2];
            block_size = (uint32_t)(in[ip + 3] << 8) | in[ip + 4];
            ip += 5;
        } else {
            block_size = (uint32_t)(in[ip] << 8) | in[ip + 1];
            ip += 2;
        }
        if (block_size == 0 || frame_size == 0) break;
        if (ip + block_size > in_size) { ok = false; break; }
        if (frame_size > out_size - op) frame_size = (uint32_t)(out_size - op);
        if (frame_size > kWindowSize / 2) { ok = false; break; }
        if (!lz->frame(in + ip, block_size, frame_size, out.data() + op)) { ok = false; break; }
        ip += block_size;
        op += frame_size;
    }
    delete lz;
    return ok && op == out_size;
}

} // namespace gpuastc
