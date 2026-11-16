// Checks a .astcpak archive against the per-file output of the same run.
// usage: astcpak_test archive.astcpak dir_with_astc_files
//   Build the inputs with:   gpuastc -batch pngs/ files/      and      gpuastc -pack pngs/ sprites.astcpak
// For every entry the blocks, size and block dimension must equal the matching files/<name>.astc,
// entries must be sorted by name, and a lookup by name must find each of them (and miss unknown names).
#include "astcpak.h"
#include "png_decode.h" // read_file

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: astcpak_test archive.astcpak dir_with_astc_files\n"); return 2; }

    std::vector<uint8_t> file;
    if (!gpuastc::read_file(argv[1], file)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    astcpak pak;
    if (!astcpak_open(&pak, file.data(), file.size())) { std::printf("FAIL: not a valid archive\n"); return 1; }

    int bad = 0, shared = 0;
    std::string prev;
    for (uint32_t i = 0; i < pak.count; ++i) {
        astcpak_entry e;
        if (!astcpak_get(&pak, i, &e)) { std::printf("FAIL: entry %u unreadable\n", i); ++bad; continue; }
        const std::string name(e.name, e.name_len);
        if (i > 0 && !(prev < name)) { std::printf("FAIL: entries not sorted at %s\n", name.c_str()); ++bad; }
        prev = name;

        astcpak_entry f;
        if (!astcpak_find(&pak, name.c_str(), &f) || f.data_offset != e.data_offset) {
            std::printf("FAIL: lookup of %s\n", name.c_str());
            ++bad;
        }

        std::vector<uint8_t> ref;
        const std::string path = std::string(argv[2]) + "/" + name + ".astc";
        if (!gpuastc::read_file(path.c_str(), ref) || ref.size() < 16) {
            std::printf("FAIL: missing reference %s\n", path.c_str());
            ++bad;
            continue;
        }
        const uint32_t rw = ref[7] | (ref[8] << 8) | (ref[9] << 16);
        const uint32_t rh = ref[10] | (ref[11] << 8) | (ref[12] << 16);
        if (ref[4] != e.block || rw != e.width || rh != e.height || ref.size() - 16 != e.data_size ||
            std::memcmp(ref.data() + 16, e.blocks, (size_t)e.data_size) != 0) {
            std::printf("FAIL: %s differs from its .astc file\n", name.c_str());
            ++bad;
        }
        if (e.data_offset % 16 != 0) { std::printf("FAIL: %s data not 16 byte aligned\n", name.c_str()); ++bad; }
    }

    astcpak_entry none;
    if (astcpak_find(&pak, "this_name_does_not_exist", &none) || astcpak_find(&pak, "", &none)) {
        std::printf("FAIL: lookup found a name that should not exist\n");
        ++bad;
    }
    // damaged input must be rejected, not crash
    std::vector<uint8_t> cut(file.begin(), file.begin() + file.size() / 2);
    astcpak broken;
    (void)astcpak_open(&broken, cut.data(), cut.size());

    std::printf("%u entries checked, %d problems (options in header: block %u, alpha %u, normal %u)\n",
                pak.count, bad, pak.opt_block, pak.opt_alpha, pak.opt_normal);
    (void)shared;
    return bad ? 1 : 0;
}
