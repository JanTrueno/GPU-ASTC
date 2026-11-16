// gpuastc: encode PNG files to .astc with a GLES 3.1 compute shader.
//
//   gpuastc in.png out.astc [options]            one file
//   gpuastc -batch in_dir out_dir [options]      every .png in a directory
//
// Batch mode keeps one GL context and shader, decodes PNGs on worker threads, and packs
// many small images into one tall texture so they are encoded in a single dispatch.
// The GLES context comes from SDL2, so it works on KMSDRM/offscreen with no window system.
#include "astc_gles.h"
#include "astcpak.h"
#include "png_decode.h"

#include <SDL.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void usage()
{
    std::fprintf(stderr,
        "usage: gpuastc input.png output.astc [options]\n"
        "       gpuastc -batch input_dir output_dir [options]\n"
        "       gpuastc -pack input_dir archive.astcpak [options]   (all textures in one file)\n"
        "       gpuastc -unpack archive.astcpak output_dir          (back to .astc files)\n"
        "  -b auto|4|5|6     block size (default auto: 4x4 up to 256x256, thin or alpha textures;\n"
        "                    5x5 up to 2048x2048; 6x6 above)\n"
        "  -alpha auto|on|off  encode alpha (default auto: on if any alpha < 255)\n"
        "  -norm             normal map (B and A forced to 1)\n"
        "  -j N              decoder threads in batch mode (default: cores - 1)\n"
        "  -prefetch N       read up to N files ahead into the page cache on helper threads (slow storage)\n"
        "  -update           skip files whose .astc is already newer than the .png\n"
        "                    (with -pack: reuse unchanged entries from the existing archive)\n"
        "  -upload           also upload the result into an ASTC texture on the GPU (self-test)\n"
        "  -v                verbose\n");
}

struct Cli {
    int block = 0;      // 0 = auto
    int alpha_mode = 0; // 0 auto, 1 on, 2 off
    bool normal = false;
    bool upload = false;
    bool verbose = false;
    int threads = 0;
    bool update = false; // skip up-to-date outputs
    int prefetch = 0;    // read-ahead distance in files (0 = off)
    bool batch = false;
    bool pack = false;   // -pack: write one archive instead of one file per texture
    bool unpack = false; // -unpack: archive -> .astc files
    std::string in, out;
};

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------

bool write_astc(const std::string& path, int bdim, int w, int h, const uint8_t* data, size_t size)
{
    uint8_t hdr[16] = {0x13, 0xAB, 0xA1, 0x5C, (uint8_t)bdim, (uint8_t)bdim, 1,
                       (uint8_t)w, (uint8_t)(w >> 8), (uint8_t)(w >> 16),
                       (uint8_t)h, (uint8_t)(h >> 8), (uint8_t)(h >> 16), 1, 0, 0};
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr) && std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    return ok;
}

bool ends_with_png(const std::string& s)
{
    if (s.size() < 4) return false;
    std::string e = s.substr(s.size() - 4);
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    return e == ".png";
}

typedef std::vector<std::pair<std::string, std::string>> FileList; // (input path, output path)

bool list_directory(const std::string& in_dir, const std::string& out_dir, FileList& files)
{
    DIR* d = opendir(in_dir.c_str());
    if (!d) return false;
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (ends_with_png(n)) names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    for (const std::string& n : names)
        files.push_back(std::make_pair(in_dir + "/" + n, out_dir + "/" + n.substr(0, n.size() - 4) + ".astc"));
    return true;
}

// ---------------------------------------------------------------------------
// decoded images flow from worker threads to the GL thread through a bounded queue
// ---------------------------------------------------------------------------

struct Job {
    std::string out_path;
    gpuastc::Image img;
    int block = 4;
    bool alpha = false;
    uint64_t src_size = 0, src_mtime = 0; // of the source PNG (for -pack -update)
};

struct WriteJob {
    std::string path;     // output file, or the entry name in -pack mode
    int block = 4, w = 0, h = 0;
    bool alpha = false;
    uint64_t src_size = 0, src_mtime = 0;
    std::vector<uint8_t> data;
    const uint8_t* ext = nullptr; // -pack -update: blocks to copy from the old archive instead of `data`
    size_t ext_size = 0;
};

inline size_t byte_size(const Job& j) { return j.img.rgba.size(); }
inline size_t byte_size(const WriteJob& j) { return j.data.size(); }

// Blocking queue that holds at most max_bytes of payload (but always accepts one item).
template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_bytes) : max_bytes_(max_bytes) {}

    void push(T&& j)
    {
        std::unique_lock<std::mutex> lk(m_);
        const size_t sz = byte_size(j);
        not_full_.wait(lk, [&] { return bytes_ == 0 || bytes_ + sz <= max_bytes_; });
        bytes_ += sz;
        q_.push_back(std::move(j));
        not_empty_.notify_one();
    }

    bool pop(T& j)
    {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        j = std::move(q_.front());
        q_.pop_front();
        bytes_ -= byte_size(j);
        not_full_.notify_all();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_empty_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<T> q_;
    size_t bytes_ = 0;
    const size_t max_bytes_;
    bool closed_ = false;
};

typedef BoundedQueue<Job> JobQueue;
typedef BoundedQueue<WriteJob> WriteQueue;

// ---------------------------------------------------------------------------
// .astcpak archives (format: include/astcpak.h)
// ---------------------------------------------------------------------------

uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h)
{
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

void set32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
void set64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }

// Read-only memory map of a whole file.
struct MappedFile {
    void* p = nullptr;
    size_t n = 0;
    ~MappedFile() { if (p) munmap(p, n); }
    bool open(const char* path)
    {
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return false; }
        void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) return false;
        p = m;
        n = (size_t)st.st_size;
        return true;
    }
};

// Writes the archive sequentially (data first, index last) to <path>.tmp and renames it when done.
// Used only from the writer thread.
struct PackWriter {
    struct Ent {
        std::string name;
        uint32_t w = 0, h = 0, block = 4;
        bool alpha = false;
        uint64_t off = 0, size = 0, src_size = 0, src_mtime = 0;
    };
    std::string final_path, tmp_path;
    FILE* f = nullptr;
    uint64_t pos = 0;
    std::vector<Ent> ents;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t> seen; // (size, hash, hash) -> offset
    int shared = 0; // entries that reuse an identical earlier blob
    bool ok = true;

    bool open(const std::string& path)
    {
        final_path = path;
        tmp_path = path + ".tmp";
        f = std::fopen(tmp_path.c_str(), "wb");
        if (!f) return false;
        std::setvbuf(f, nullptr, _IOFBF, 1 << 20);
        uint8_t zero[ASTCPAK_HEADER_SIZE] = {0};
        ok = std::fwrite(zero, 1, sizeof(zero), f) == sizeof(zero); // header is rewritten in finish()
        pos = ASTCPAK_HEADER_SIZE;
        return ok;
    }

    void pad16()
    {
        while (pos & 15) { std::fputc(0, f); ++pos; }
    }

    void add(const WriteJob& wj)
    {
        const uint8_t* d = wj.ext ? wj.ext : wj.data.data();
        const size_t n = wj.ext ? wj.ext_size : wj.data.size();
        Ent e;
        e.name = wj.path;
        e.w = (uint32_t)wj.w;
        e.h = (uint32_t)wj.h;
        e.block = (uint32_t)wj.block;
        e.alpha = wj.alpha;
        e.src_size = wj.src_size;
        e.src_mtime = wj.src_mtime;
        e.size = n;
        const auto key = std::make_tuple((uint64_t)n, fnv1a(d, n, 1469598103934665603ull), fnv1a(d, n, 0x9E3779B97F4A7C15ull));
        auto it = seen.find(key);
        if (it != seen.end()) {
            e.off = it->second; // identical texture already stored: share it
            ++shared;
        } else {
            pad16();
            e.off = pos;
            if (std::fwrite(d, 1, n, f) != n) ok = false;
            pos += n;
            seen[key] = e.off;
        }
        ents.push_back(e);
    }

    bool finish(uint32_t opt_block, uint32_t opt_alpha, uint32_t opt_normal)
    {
        std::sort(ents.begin(), ents.end(), [](const Ent& a, const Ent& b) { return a.name < b.name; });
        pad16();
        const uint64_t index_off = pos;
        std::vector<uint8_t> idx;
        std::string names;
        auto p32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) idx.push_back((uint8_t)(v >> (8 * i))); };
        auto p64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) idx.push_back((uint8_t)(v >> (8 * i))); };
        for (const Ent& e : ents) {
            p32((uint32_t)names.size()); p32((uint32_t)e.name.size());
            p32(e.w); p32(e.h);
            idx.push_back((uint8_t)e.block); idx.push_back(e.alpha ? 1 : 0); idx.push_back(0); idx.push_back(0);
            p64(e.off); p64(e.size); p64(e.src_size); p64(e.src_mtime);
            names += e.name;
        }
        idx.insert(idx.end(), names.begin(), names.end());
        if (std::fwrite(idx.data(), 1, idx.size(), f) != idx.size()) ok = false;
        pos += idx.size();

        uint8_t hdr[ASTCPAK_HEADER_SIZE] = {0};
        std::memcpy(hdr, "ASTCPAK1", 8);
        set32(hdr + 8, ASTCPAK_VERSION);
        set32(hdr + 12, (uint32_t)ents.size());
        set64(hdr + 16, index_off);
        set64(hdr + 24, idx.size());
        set32(hdr + 32, opt_block);
        set32(hdr + 36, opt_alpha);
        set32(hdr + 40, opt_normal);
        if (std::fseek(f, 0, SEEK_SET) != 0 || std::fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) ok = false;
        if (std::fclose(f) != 0) ok = false;
        f = nullptr;
        if (ok && std::rename(tmp_path.c_str(), final_path.c_str()) != 0) ok = false;
        if (!ok) std::remove(tmp_path.c_str());
        return ok;
    }
};

// -unpack: archive -> one .astc file per entry
int unpack_archive(const std::string& archive, const std::string& out_dir)
{
    MappedFile mf;
    astcpak pak;
    if (!mf.open(archive.c_str()) || !astcpak_open(&pak, mf.p, mf.n)) {
        std::fprintf(stderr, "%s: not a readable .astcpak archive\n", archive.c_str());
        return 1;
    }
    mkdir(out_dir.c_str(), 0755);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < pak.count; ++i) {
        astcpak_entry e;
        if (!astcpak_get(&pak, i, &e)) { ++bad; continue; }
        const std::string path = out_dir + "/" + std::string(e.name, e.name_len) + ".astc";
        if (!write_astc(path, (int)e.block, (int)e.width, (int)e.height, (const uint8_t*)e.blocks, (size_t)e.data_size)) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            ++bad;
        }
    }
    std::printf("%u textures extracted to %s%s\n", pak.count - bad, out_dir.c_str(), bad ? " (some failed)" : "");
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// GL side
// ---------------------------------------------------------------------------

bool has_extension(const char* name)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; ++i) {
        const char* e = (const char*)glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && std::strcmp(e, name) == 0)
            return true;
    }
    return false;
}

struct Gpu {
    std::map<std::pair<int, int>, std::unique_ptr<gpuastc::Encoder>> encoders; // (block, alpha)
    GLuint tex = 0, buf = 0;
    WriteQueue* writer = nullptr; // finished files go here; a writer thread saves them
    int max_height = 4096; // tallest atlas we build
    int max_tex = 2048;
    bool normal = false, upload = false, verbose = false;

    // statistics
    double ms_gpu = 0;
    long long pixels = 0, bytes_out = 0;
    int files_ok = 0, files_failed = 0, dispatches = 0;

    gpuastc::Encoder* encoder(int block, bool alpha)
    {
        auto key = std::make_pair(block, alpha ? 1 : 0);
        auto it = encoders.find(key);
        if (it != encoders.end()) return it->second.get();
        std::unique_ptr<gpuastc::Encoder> e(new gpuastc::Encoder());
        gpuastc::Options o;
        o.block = block;
        o.alpha = alpha;
        o.normal_map = normal;
        std::string log;
        if (!e->create(o, &log)) {
            std::fprintf(stderr, "shader error (%dx%d, alpha %d):\n%s\n", block, block, alpha ? 1 : 0, log.c_str());
            return nullptr;
        }
        gpuastc::Encoder* raw = e.get();
        encoders[key] = std::move(e);
        return raw;
    }
};

// A set of images that are encoded together. All images in a pack share block size, alpha
// mode and padded width, and are stacked vertically (each padded to whole blocks), so every
// image's blocks form one contiguous slab of the output.
struct Pack {
    int block = 4;
    bool alpha = false;
    int width = 0;  // padded width
    int height = 0; // total padded height
    long long pixels = 0;
    std::vector<Job> jobs;
    std::vector<int> row0; // first padded row of each job

    static int pad(int v, int block) { return (v + block - 1) / block * block; }
    void clear()
    {
        jobs.clear();
        row0.clear();
        height = 0;
        pixels = 0;
    }
    void add(Job&& j)
    {
        row0.push_back(height);
        height += pad(j.img.h, block);
        pixels += (long long)j.img.w * j.img.h;
        jobs.push_back(std::move(j));
    }
};

bool flush_pack(Gpu& g, Pack& p)
{
    if (p.jobs.empty()) return true;

    const int block = p.block;
    gpuastc::Encoder* enc = g.encoder(block, p.alpha);
    if (!enc) { g.files_failed += (int)p.jobs.size(); p.clear(); return false; }

    const bool single = p.jobs.size() == 1;
    int W = 0, H = 0;
    const uint8_t* data = nullptr;
    std::vector<uint8_t> atlas;
    if (single) {
        W = p.jobs[0].img.w;
        H = p.jobs[0].img.h;
        data = p.jobs[0].img.rgba.data();
    } else {
        W = p.width;
        H = p.height;
        atlas.assign((size_t)W * H * 4, 0);
        for (size_t k = 0; k < p.jobs.size(); ++k) {
            const gpuastc::Image& im = p.jobs[k].img;
            const int ph = Pack::pad(im.h, block);
            for (int y = 0; y < ph; ++y) {
                const uint8_t* src = &im.rgba[(size_t)std::min(y, im.h - 1) * im.w * 4];
                uint8_t* dst = &atlas[((size_t)(p.row0[k] + y) * W) * 4];
                std::memcpy(dst, src, (size_t)im.w * 4);
                for (int x = im.w; x < W; ++x) // replicate the last pixel into the padding
                    std::memcpy(dst + (size_t)x * 4, src + (size_t)(im.w - 1) * 4, 4);
            }
        }
        data = atlas.data();
    }
    if (W > g.max_tex || H > g.max_tex) {
        std::fprintf(stderr, "%s: %dx%d exceeds GL_MAX_TEXTURE_SIZE (%d)\n", p.jobs[0].out_path.c_str(), W, H, g.max_tex);
        g.files_failed += (int)p.jobs.size();
        p.clear();
        return false;
    }

    const double t0 = now_ms();
    glBindTexture(GL_TEXTURE_2D, g.tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    if (!enc->encode(g.tex, W, H, g.buf)) {
        std::fprintf(stderr, "dispatch failed (GL error 0x%x)\n", glGetError());
        g.files_failed += (int)p.jobs.size();
        p.clear();
        return false;
    }

    const int bx = gpuastc::Encoder::blocks(W, block), by = gpuastc::Encoder::blocks(H, block);
    const size_t total = (size_t)bx * by * 16;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g.buf);
    const uint8_t* mapped = (const uint8_t*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)total, GL_MAP_READ_BIT);
    if (!mapped) {
        std::fprintf(stderr, "map failed (GL error 0x%x)\n", glGetError());
        g.files_failed += (int)p.jobs.size();
        p.clear();
        return false;
    }
    const double t1 = now_ms();
    g.ms_gpu += t1 - t0;
    ++g.dispatches;

    for (size_t k = 0; k < p.jobs.size(); ++k) {
        const Job& j = p.jobs[k];
        const size_t first = single ? 0 : (size_t)(p.row0[k] / block) * bx * 16;
        const size_t size = (size_t)gpuastc::Encoder::blocks(j.img.h, block) * bx * 16;
        WriteJob wj;
        wj.path = j.out_path;
        wj.block = block;
        wj.w = j.img.w;
        wj.h = j.img.h;
        wj.alpha = p.alpha;
        wj.src_size = j.src_size;
        wj.src_mtime = j.src_mtime;
        wj.data.assign(mapped + first, mapped + first + size);
        g.writer->push(std::move(wj)); // saved on the writer thread, overlapping the next dispatch
        ++g.files_ok;
        g.pixels += (long long)j.img.w * j.img.h;
        g.bytes_out += (long long)size + 16;
        if (g.verbose && p.jobs.size() == 1)
            std::printf("%s: %dx%d -> %dx%d blocks of %dx%d\n", j.out_path.c_str(), j.img.w, j.img.h, bx, by, block, block);
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    if (g.upload && single) {
        // Zero-copy path for real use: SSBO -> PBO unpack -> ASTC texture.
        if (!has_extension("GL_KHR_texture_compression_astc_ldr")) {
            std::fprintf(stderr, "upload: GL_KHR_texture_compression_astc_ldr not available\n");
        } else {
            GLuint ctex = 0;
            glGenTextures(1, &ctex);
            glBindTexture(GL_TEXTURE_2D, ctex);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g.buf);
            glCompressedTexImage2D(GL_TEXTURE_2D, 0, gpuastc::Encoder::internal_format(block), W, H, 0, (GLsizei)total, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            const GLenum err = glGetError();
            std::printf("upload to ASTC texture: %s\n", err == GL_NO_ERROR ? "ok" : "failed");
            if (err != GL_NO_ERROR) std::printf("  GL error 0x%x\n", err);
            glDeleteTextures(1, &ctex);
        }
    }

    p.clear();
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

const long long kSmallPixels = 1 << 18;      // images up to this size are packed together
const long long kPackPixels = 1 << 20;       // flush a pack once it holds this many pixels
const long long kMaxPendingPixels = 1 << 24; // flush everything above this many waiting pixels

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { usage(); return 1; }

    Cli cli;
    int first_opt = 3;
    if (!std::strcmp(argv[1], "-batch") || !std::strcmp(argv[1], "-pack") || !std::strcmp(argv[1], "-unpack")) {
        if (argc < 4) { usage(); return 1; }
        cli.batch = true;
        cli.pack = !std::strcmp(argv[1], "-pack");
        cli.unpack = !std::strcmp(argv[1], "-unpack");
        cli.in = argv[2];
        cli.out = argv[3];
        first_opt = 4;
    } else {
        cli.in = argv[1];
        cli.out = argv[2];
    }
    for (int i = first_opt; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
            const char* v = argv[++i];
            cli.block = !std::strcmp(v, "auto") ? 0 : std::atoi(v);
        }
        else if (!std::strcmp(argv[i], "-alpha") && i + 1 < argc) {
            const char* v = argv[++i];
            cli.alpha_mode = !std::strcmp(v, "on") ? 1 : !std::strcmp(v, "off") ? 2 : 0;
        }
        else if (!std::strcmp(argv[i], "-j") && i + 1 < argc) cli.threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-update")) cli.update = true;
        else if (!std::strcmp(argv[i], "-prefetch") && i + 1 < argc) cli.prefetch = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-norm")) cli.normal = true;
        else if (!std::strcmp(argv[i], "-upload")) cli.upload = true;
        else if (!std::strcmp(argv[i], "-v")) cli.verbose = true;
        else { usage(); return 1; }
    }
    if (cli.block != 0 && cli.block != 4 && cli.block != 5 && cli.block != 6) {
        std::fprintf(stderr, "block must be auto, 4, 5 or 6\n");
        return 1;
    }

    if (cli.unpack)
        return unpack_archive(cli.in, cli.out);

    // what to convert
    FileList files;
    if (cli.batch) {
        if (!list_directory(cli.in, cli.out, files)) { std::fprintf(stderr, "cannot read directory %s\n", cli.in.c_str()); return 1; }
        if (cli.pack) {
            // the "output" of each file is its entry name in the archive: the file name without ".png"
            for (auto& f : files) {
                const size_t slash = f.first.find_last_of('/');
                const std::string base = f.first.substr(slash == std::string::npos ? 0 : slash + 1);
                f.second = base.substr(0, base.size() - 4);
            }
        } else {
            mkdir(cli.out.c_str(), 0755); // ok if it already exists
        }
        if (files.empty()) { std::fprintf(stderr, "no .png files in %s\n", cli.in.c_str()); return 1; }
    } else {
        files.push_back(std::make_pair(cli.in, cli.out));
    }

    PackWriter pack_writer;
    MappedFile old_file;
    astcpak old_pak;
    bool old_ok = false; // an existing archive made with the same options, whose entries may be reused
    if (cli.pack) {
        if (cli.update && old_file.open(cli.out.c_str()) && astcpak_open(&old_pak, old_file.p, old_file.n) &&
            old_pak.opt_block == (uint32_t)cli.block && old_pak.opt_alpha == (uint32_t)cli.alpha_mode &&
            old_pak.opt_normal == (cli.normal ? 1u : 0u))
            old_ok = true;
        if (!pack_writer.open(cli.out)) { std::fprintf(stderr, "cannot create %s.tmp\n", cli.out.c_str()); return 1; }
    }

    // GLES 3.1 context through SDL2 (hidden window; works with offscreen/KMSDRM drivers).
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_Window* win = SDL_CreateWindow("gpuastc", 0, 0, 64, 64, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) { std::fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) { std::fprintf(stderr, "SDL_GL_CreateContext (GLES 3.1): %s\n", SDL_GetError()); return 1; }
    if (cli.verbose)
        std::printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    Gpu gpu;
    gpu.normal = cli.normal;
    gpu.upload = cli.upload;
    gpu.verbose = cli.verbose;
    GLint max_tex = 2048;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
    gpu.max_tex = (int)max_tex;
    gpu.max_height = std::min<int>((int)max_tex, 8192);
    glGenTextures(1, &gpu.tex);
    glBindTexture(GL_TEXTURE_2D, gpu.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // texelFetch needs a complete texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenBuffers(1, &gpu.buf);

    // decoder threads
    int nthreads = cli.threads;
    if (nthreads <= 0) {
        const int hw = (int)std::thread::hardware_concurrency();
        nthreads = std::max(1, std::min(hw - 1, 16));
    }
    if (!cli.batch) nthreads = 1;
    nthreads = std::min<int>(nthreads, (int)files.size());

    JobQueue queue(256u << 20); // at most ~256 MB of decoded pixels waiting
    WriteQueue write_queue(64u << 20); // and ~64 MB of finished blocks waiting to be saved
    gpu.writer = &write_queue;
    std::atomic<int> write_failed(0);
    double write_ms = 0; // only touched by the writer thread until it is joined
    std::thread writer([&]() {
        WriteJob wj;
        while (write_queue.pop(wj)) {
            const double t0 = now_ms();
            if (cli.pack) {
                pack_writer.add(wj);
            } else if (!write_astc(wj.path, wj.block, wj.w, wj.h, wj.data.data(), wj.data.size())) {
                std::fprintf(stderr, "cannot write %s%s", wj.path.c_str(), "\n");
                ++write_failed;
            }
            write_ms += now_ms() - t0;
        }
    });
    std::atomic<size_t> next(0);
    std::atomic<int> decode_failed(0);
    std::atomic<int> skipped(0);
    std::atomic<int> running(nthreads);
    std::vector<double> decode_ms(nthreads, 0.0);
    std::vector<double> read_ms(nthreads, 0.0);
    std::vector<std::thread> workers;
    const double wall0 = now_ms();
    for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back([&, t]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= files.size()) break;
                struct stat st_src;
                const bool have_src = stat(files[i].first.c_str(), &st_src) == 0;
                if (cli.pack && old_ok && have_src) { // unchanged since it was packed: copy the old blocks
                    astcpak_entry oe;
                    if (astcpak_find(&old_pak, files[i].second.c_str(), &oe) && oe.src_size == (uint64_t)st_src.st_size &&
                        oe.src_mtime == (uint64_t)st_src.st_mtime) {
                        WriteJob wj;
                        wj.path = files[i].second;
                        wj.w = (int)oe.width;
                        wj.h = (int)oe.height;
                        wj.block = (int)oe.block;
                        wj.alpha = oe.alpha != 0;
                        wj.src_size = oe.src_size;
                        wj.src_mtime = oe.src_mtime;
                        wj.ext = (const uint8_t*)oe.blocks;
                        wj.ext_size = (size_t)oe.data_size;
                        write_queue.push(std::move(wj));
                        ++skipped;
                        continue;
                    }
                }
                if (cli.update && !cli.pack) { // output already there and not older than the input: nothing to do
                    struct stat st_in, st_out;
                    if (stat(files[i].first.c_str(), &st_in) == 0 && stat(files[i].second.c_str(), &st_out) == 0 &&
                        st_out.st_size > 16 && st_out.st_mtime >= st_in.st_mtime) {
                        ++skipped;
                        continue;
                    }
                }
                Job j;
                std::string err;
                std::vector<uint8_t> file_bytes;
                const double t0 = now_ms();
                const bool read_ok = gpuastc::read_file(files[i].first.c_str(), file_bytes);
                const double t1 = now_ms();
                if (!read_ok) err = "cannot read file";
                if (!read_ok || !gpuastc::decode_png_memory(file_bytes, j.img, &err)) {
                    std::fprintf(stderr, "%s: %s\n", files[i].first.c_str(), err.c_str());
                    ++decode_failed;
                    continue;
                }
                read_ms[t] += t1 - t0;
                decode_ms[t] += now_ms() - t1;
                j.alpha = cli.alpha_mode == 1 || (cli.alpha_mode == 0 && j.img.alpha);
                j.block = cli.block ? cli.block : gpuastc::Encoder::pick_block(j.img.w, j.img.h, j.alpha);
                j.out_path = files[i].second;
                if (have_src) { j.src_size = (uint64_t)st_src.st_size; j.src_mtime = (uint64_t)st_src.st_mtime; }
                queue.push(std::move(j));
            }
            if (--running == 0)
                queue.close();
        });
    }

    // Optional read-ahead: helper threads ask the OS to load the next files into the page cache,
    // so the decode threads find them in memory (hides per-file latency on slow storage).
    std::atomic<size_t> pf_next(0);
    std::atomic<bool> pf_stop(false);
    std::vector<std::thread> prefetchers;
    if (cli.prefetch > 0) {
        for (int k = 0; k < 2; ++k) {
            prefetchers.emplace_back([&]() {
                for (;;) {
                    const size_t i = pf_next.fetch_add(1);
                    if (i >= files.size()) break;
                    while (!pf_stop.load() && i >= next.load() + (size_t)cli.prefetch)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (pf_stop.load()) break;
                    const int fd = ::open(files[i].first.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        struct stat st;
                        if (fstat(fd, &st) == 0 && st.st_size > 0)
                            readahead(fd, 0, (size_t)st.st_size);
                        ::close(fd);
                    }
                }
            });
        }
    }

    // GL thread: collect small images into packs, encode big ones directly.
    std::map<std::tuple<int, bool, int>, Pack> packs;
    long long pending = 0;
    Job job;
    while (queue.pop(job)) {
        const int w = job.img.w, h = job.img.h;
        if ((long long)w * h > kSmallPixels) {
            Pack one;
            one.block = job.block;
            one.alpha = job.alpha;
            one.add(std::move(job));
            flush_pack(gpu, one);
            continue;
        }
        const int pw = Pack::pad(w, job.block), ph = Pack::pad(h, job.block);
        Pack& p = packs[std::make_tuple(job.block, job.alpha, pw)];
        if (p.jobs.empty()) { p.block = job.block; p.alpha = job.alpha; p.width = pw; }
        if (p.height + ph > gpu.max_height) { pending -= p.pixels; flush_pack(gpu, p); }
        pending += (long long)w * h;
        p.add(std::move(job));
        if (p.pixels >= kPackPixels) { pending -= p.pixels; flush_pack(gpu, p); }
        if (pending > kMaxPendingPixels) {
            for (auto& kv : packs) flush_pack(gpu, kv.second);
            pending = 0;
        }
    }
    for (auto& kv : packs) flush_pack(gpu, kv.second);
    for (std::thread& t : workers) t.join();
    pf_stop = true;
    for (std::thread& t : prefetchers) t.join();
    write_queue.close();
    writer.join();
    bool pack_ok = true;
    if (cli.pack) {
        pack_ok = pack_writer.finish((uint32_t)cli.block, (uint32_t)cli.alpha_mode, cli.normal ? 1u : 0u);
        if (!pack_ok) std::fprintf(stderr, "cannot write archive %s\n", cli.out.c_str());
    }
    gpu.files_ok -= write_failed.load();
    gpu.files_failed += write_failed.load();
    const double wall = now_ms() - wall0;
    gpu.files_failed += decode_failed.load();

    double dec_total = 0, read_total = 0;
    for (double d : decode_ms) dec_total += d;
    for (double d : read_ms) read_total += d;
    if (cli.batch || cli.verbose) {
        std::printf("%d files converted, %d up to date (skipped), %d failed, %.2f Mpx in %.0f ms (%.1f files/s, %.1f Mpx/s)\n",
                    gpu.files_ok, skipped.load(), gpu.files_failed, gpu.pixels / 1e6, wall,
                    wall > 0 ? gpu.files_ok * 1000.0 / wall : 0.0, wall > 0 ? gpu.pixels / 1e3 / wall : 0.0);
        std::printf("  file read %.0f ms + decode %.0f ms across %d thread(s), GPU upload+encode+readback %.0f ms in %d dispatch(es), file writes %.0f ms (own thread), output %.1f MB\n",
                    read_total, dec_total, nthreads, gpu.ms_gpu, gpu.dispatches, write_ms, gpu.bytes_out / 1e6);
        if (cli.pack && pack_ok)
            std::printf("  archive %s: %zu entries (%d sharing identical data), %.1f MB\n", cli.out.c_str(),
                        pack_writer.ents.size(), pack_writer.shared, pack_writer.pos / 1e6);
    } else if (gpu.files_ok == 1) {
        std::printf("%s: written in %.2f ms (GPU %.2f ms)\n", files[0].second.c_str(), wall, gpu.ms_gpu);
    }

    gpu.encoders.clear();
    glDeleteBuffers(1, &gpu.buf);
    glDeleteTextures(1, &gpu.tex);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return (gpu.files_failed || !pack_ok) ? 1 : 0;
}
