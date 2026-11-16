// Compares the fast PNG decoder with stb_image on the given files.
// usage: png_compare file.png [more.png ...]
#include "png_decode.h"

#include <chrono>
#include <cstdio>
#include <cstring>

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    int bad = 0, handled = 0, skipped = 0;
    double t_fast = 0, t_stb = 0;
    for (int i = 1; i < argc; ++i) {
        std::vector<uint8_t> file;
        if (!gpuastc::read_file(argv[i], file)) { std::printf("%s: cannot read\n", argv[i]); ++bad; continue; }
        gpuastc::Image a, b;
        double t0 = now_ms();
        const bool fast_ok = gpuastc::decode_png_fast(file, a);
        double t1 = now_ms();
        const bool stb_ok = gpuastc::decode_png_stb(file, b);
        double t2 = now_ms();
        if (!fast_ok) { ++skipped; continue; } // not handled by the fast path (falls back to stb)
        ++handled;
        t_fast += t1 - t0;
        t_stb += t2 - t1;
        if (!stb_ok || a.w != b.w || a.h != b.h || a.rgba != b.rgba) {
            std::printf("MISMATCH %s (fast %dx%d, stb %s %dx%d)\n", argv[i], a.w, a.h, stb_ok ? "ok" : "failed", b.w, b.h);
            ++bad;
        }
    }
    std::printf("%d compared, %d identical, %d mismatched, %d not handled by the fast path\n",
                handled, handled - bad, bad, skipped);
    if (handled)
        std::printf("fast decoder %.1f ms, stb_image %.1f ms (%.2fx)\n", t_fast, t_stb, t_stb / (t_fast > 0 ? t_fast : 1));
    return bad ? 1 : 0;
}
