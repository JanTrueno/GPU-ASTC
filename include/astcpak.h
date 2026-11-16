/* astcpak.h - reader for .astcpak archives written by `gpuastc -pack`.  C99 / C++, header only.
 *
 * An archive holds many ASTC textures in one file. Typical use (the whole file is mapped or read
 * into memory once, then every texture is just a pointer into it):
 *
 *     astcpak pak;
 *     astcpak_open(&pak, file_data, file_size);
 *     astcpak_entry e;
 *     if (astcpak_find(&pak, "player_idle_0", &e)) {
 *         glCompressedTexImage2D(GL_TEXTURE_2D, 0, astcpak_gl_format(&e),
 *                                e.width, e.height, 0, (GLsizei)e.data_size, e.blocks);
 *     }
 *
 * Layout (all integers little endian):
 *   header (64 bytes)  "ASTCPAK1", u32 version, u32 count, u64 index_offset, u64 index_size,
 *                      u32 opt_block, u32 opt_alpha, u32 opt_normal, reserved
 *   data               ASTC blocks of each texture (16 byte aligned; identical textures share one copy)
 *   index              count * 52 byte entries sorted by name, followed by the name bytes:
 *                        u32 name_off, u32 name_len, u32 width, u32 height, u8 block, u8 alpha, u16 0,
 *                        u64 data_off, u64 data_size, u64 src_size, u64 src_mtime
 * The blocks of an entry are the same bytes as the body of a .astc file (no 16 byte header).
 */
#ifndef ASTCPAK_H
#define ASTCPAK_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASTCPAK_VERSION 1u
#define ASTCPAK_HEADER_SIZE 64u
#define ASTCPAK_ENTRY_SIZE 52u

typedef struct astcpak {
    const uint8_t* data;     /* whole file */
    size_t size;
    uint32_t count;
    const uint8_t* entries;  /* index: count entries */
    const uint8_t* names;    /* index: name bytes */
    size_t names_size;
    uint32_t opt_block;      /* options the archive was made with (0 = auto) */
    uint32_t opt_alpha;
    uint32_t opt_normal;
} astcpak;

typedef struct astcpak_entry {
    const char* name;        /* not NUL terminated */
    uint32_t name_len;
    uint32_t width, height;
    uint32_t block;          /* 4, 5 or 6 (square blocks: 4x4, 5x5, 6x6) */
    int alpha;               /* encoded with alpha */
    uint64_t data_offset;
    uint64_t data_size;      /* bytes of ASTC blocks */
    const void* blocks;      /* points into the archive memory */
    uint64_t src_size;       /* size / modification time of the source PNG when packed */
    uint64_t src_mtime;
} astcpak_entry;

static inline uint32_t astcpak_rd32(const uint8_t* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t astcpak_rd64(const uint8_t* p)
{
    return (uint64_t)astcpak_rd32(p) | ((uint64_t)astcpak_rd32(p + 4) << 32);
}

/* Validates the header and index. Returns 1 on success, 0 if the data is not a usable archive. */
static inline int astcpak_open(astcpak* p, const void* data, size_t size)
{
    const uint8_t* d = (const uint8_t*)data;
    uint64_t index_off, index_size;
    if (!p || !d || size < ASTCPAK_HEADER_SIZE || memcmp(d, "ASTCPAK1", 8) != 0)
        return 0;
    if (astcpak_rd32(d + 8) != ASTCPAK_VERSION)
        return 0;
    p->count = astcpak_rd32(d + 12);
    index_off = astcpak_rd64(d + 16);
    index_size = astcpak_rd64(d + 24);
    p->opt_block = astcpak_rd32(d + 32);
    p->opt_alpha = astcpak_rd32(d + 36);
    p->opt_normal = astcpak_rd32(d + 40);
    if (index_off < ASTCPAK_HEADER_SIZE || index_off > size || index_size > size - index_off)
        return 0;
    if ((uint64_t)p->count * ASTCPAK_ENTRY_SIZE > index_size)
        return 0;
    p->data = d;
    p->size = size;
    p->entries = d + index_off;
    p->names = p->entries + (size_t)p->count * ASTCPAK_ENTRY_SIZE;
    p->names_size = (size_t)(index_size - (uint64_t)p->count * ASTCPAK_ENTRY_SIZE);
    return 1;
}

/* Fills *e with entry number i (0 <= i < count; entries are sorted by name). Returns 0 if the entry is damaged. */
static inline int astcpak_get(const astcpak* p, uint32_t i, astcpak_entry* e)
{
    const uint8_t* r;
    uint32_t name_off, name_len;
    if (!p || !e || i >= p->count)
        return 0;
    r = p->entries + (size_t)i * ASTCPAK_ENTRY_SIZE;
    name_off = astcpak_rd32(r);
    name_len = astcpak_rd32(r + 4);
    if ((uint64_t)name_off + name_len > p->names_size)
        return 0;
    e->name = (const char*)(p->names + name_off);
    e->name_len = name_len;
    e->width = astcpak_rd32(r + 8);
    e->height = astcpak_rd32(r + 12);
    e->block = r[16];
    e->alpha = r[17];
    e->data_offset = astcpak_rd64(r + 20);
    e->data_size = astcpak_rd64(r + 28);
    e->src_size = astcpak_rd64(r + 36);
    e->src_mtime = astcpak_rd64(r + 44);
    if (e->data_offset > p->size || e->data_size > p->size - e->data_offset)
        return 0;
    e->blocks = p->data + e->data_offset;
    return 1;
}

/* Binary search by name. Returns 1 and fills *e if found. */
static inline int astcpak_find_n(const astcpak* p, const char* name, size_t len, astcpak_entry* e)
{
    uint32_t lo = 0, hi = p ? p->count : 0;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        astcpak_entry m;
        size_t n;
        int c;
        if (!astcpak_get(p, mid, &m))
            return 0;
        n = m.name_len < len ? m.name_len : len;
        c = memcmp(m.name, name, n);
        if (c == 0)
            c = (m.name_len < len) ? -1 : (m.name_len > len) ? 1 : 0;
        if (c == 0) {
            *e = m;
            return 1;
        }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return 0;
}

static inline int astcpak_find(const astcpak* p, const char* name, astcpak_entry* e)
{
    return astcpak_find_n(p, name, strlen(name), e);
}

/* The OpenGL internal format for glCompressedTexImage2D (GL_KHR_texture_compression_astc_ldr). */
static inline unsigned astcpak_gl_format(const astcpak_entry* e)
{
    return e->block == 6 ? 0x93B4u /* 6x6 */ : e->block == 5 ? 0x93B2u /* 5x5 */ : 0x93B0u /* 4x4 */;
}

#ifdef __cplusplus
}
#endif

#endif /* ASTCPAK_H */
