// gpuastc: encode PNG files to .astc with a GLES 3.1 compute shader.
//
//   gpuastc in.png out.astc [options]            one file
//   gpuastc -batch in_dir out_dir [options]      every .png in a directory
//
// Batch mode keeps one GL context and shader, decodes PNGs on worker threads, and packs
// many small images into one tall texture so they are encoded in a single dispatch.
//
// Storage (usually an SD card) is used in two phases: reader threads fetch the source files into
// memory as far ahead as the memory budget allows; results are held in memory until the reading
// is done and then written in path order, so the card is not asked to read and write at the
// same time.
// The GLES context comes from SDL2, so it works on KMSDRM/offscreen with no window system.
#include "astc_gles.h"
#include "astcpak.h"
#include "formats.h"
#include "png_decode.h"
#include "xnb.h"

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
#include <set>
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
        "       gpuastc input.xnb output.xnb [options]              (XNA/FNA texture -> ASTC xnb)\n"
        "       gpuastc -batch input_dir output_dir [options]       (.png and .xnb, subfolders included;\n"
        "                                                            input_dir == output_dir converts in place)\n"
        "       gpuastc -pack input_dir archive.astcpak [options]   (all textures in one file)\n"
        "       gpuastc -unpack archive.astcpak output_dir          (back to .astc files)\n"
        "  -b auto|4|5|6     block size (default auto: 4x4 up to 256x256, thin or alpha textures;\n"
        "                    5x5 up to 2048x2048; 6x6 above)\n"
        "  -alpha auto|on|off  encode alpha (default auto: on if any alpha < 255)\n"
        "  -q fast|normal|best  4x4 search effort (default normal). fast: one colour line per block.\n"
        "                    normal: blocks that fits badly also try two partitions. best: more of that.\n"
        "  -norm             normal map (B and A forced to 1)\n"
        "  -j N              decoder threads in batch mode (default: cores - 1)\n"
        "  -mem MB           memory for source files, decoded images and results waiting between reading,\n"
        "                    encoding and saving (default: 40%% of the free memory, 64 to 768)\n"
        "  -nosync           do not wait for the results to reach the storage before exiting\n"
        "  -cache DIR        keep compiled shaders in DIR between runs (default: $XDG_CACHE_HOME/gpuastc or\n"
        "                    ~/.cache/gpuastc); -nocache compiles every time\n"
        "  -force            convert everything again. By default a file whose output is already newer than\n"
        "                    the input is skipped (with -pack: unchanged entries are reused from the old\n"
        "                    archive). Use -force after changing -b, -alpha, -norm or -premult.\n"
        "  -premult          treat the colour as premultiplied by alpha (always on for .xnb input)\n"
        "  -upload           also upload the result into an ASTC texture on the GPU (self-test)\n"
        "  -v                verbose\n");
}

struct Cli {
    int block = 0;      // 0 = auto
    int alpha_mode = 0; // 0 auto, 1 on, 2 off
    int quality = 1;    // 0 fast, 1 normal, 2 best
    bool normal = false;
    bool upload = false;
    bool verbose = false;
    int threads = 0;
    int mem_mb = 0;      // 0 = from free memory. Split: 1/6 source files, 1/2 decoded images, 1/6 small images
                         // collected for one dispatch, 1/6 finished output
    bool sync = true;    // batch: wait until the results are on the storage
    std::string cache;   // directory for compiled shaders
    bool use_cache = true;
    bool update = true;  // skip up-to-date outputs (-force turns this off)
    bool premult = false; // force premultiplied-alpha handling
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

// Writes through <path>.tmp and renames, so an interrupted run (or converting in place) never leaves a broken file.
bool write_file_atomic(const std::string& path, const uint8_t* data, size_t size)
{
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data, 1, size, f) == size;
    if (std::fclose(f) != 0 || !ok) { std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}


// Reads a whole file with one path lookup (open, then fstat on the descriptor): on FUSE file
// systems every lookup by name is a round trip to the driver process.
bool read_file_once(const char* path, std::vector<uint8_t>& out, uint64_t& size, uint64_t& mtime)
{
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); return false; }
    size = (uint64_t)st.st_size;
    mtime = (uint64_t)st.st_mtime;
    out.resize((size_t)st.st_size);
    size_t done = 0;
    while (done < out.size()) {
        const ssize_t n = ::read(fd, out.data() + done, out.size() - done);
        if (n <= 0) break;
        done += (size_t)n;
    }
    ::close(fd);
    return done == out.size();
}

// Creates the directories above path. `made` remembers the ones already done, so each directory
// costs its mkdir calls once and not once per file.
void make_parents(const std::string& path, std::set<std::string>& made)
{
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return;
    const std::string dir = path.substr(0, slash);
    if (!made.insert(dir).second) return;
    for (size_t i = 1; i < dir.size(); ++i)
        if (dir[i] == '/') mkdir(dir.substr(0, i).c_str(), 0755); // fails harmlessly if it exists
    mkdir(dir.c_str(), 0755);
}

// Free memory in MB (MemAvailable), or 0 if it cannot be read.
int available_memory_mb()
{
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    long kb = 0;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return (int)(kb / 1024);
}

// Asks the file system that holds dir to write out what it has buffered, and waits for it.
void sync_storage(const std::string& dir)
{
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) return;
    if (syncfs(fd) != 0) fsync(fd);
    ::close(fd);
}

typedef std::vector<std::pair<std::string, std::string>> FileList; // (input path, output path / entry name)

void walk_directory(const std::string& root, const std::string& rel, std::vector<std::string>& names)
{
    DIR* d = opendir((rel.empty() ? root : root + "/" + rel).c_str());
    if (!d) return;
    std::vector<std::string> entries;
    while (dirent* e = readdir(d)) entries.push_back(e->d_name);
    closedir(d);
    for (const std::string& n : entries) {
        if (n == "." || n == "..") continue;
        const std::string r = rel.empty() ? n : rel + "/" + n;
        struct stat st;
        if (stat((root + "/" + r).c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) walk_directory(root, r, names);
        else if (gpuastc::input_format_for(n)) names.push_back(r);
    }
}

// Every supported file below in_dir. The output of "dir/a.png" is "out/dir/a.astc"; an .xnb keeps its
// name ("out/dir/a.xnb", now holding ASTC). With `pack` the output is the entry name instead ("dir/a").
bool list_directory(const std::string& in_dir, const std::string& out_dir, bool pack, FileList& files)
{
    DIR* d = opendir(in_dir.c_str());
    if (!d) return false;
    closedir(d);
    std::vector<std::string> names;
    walk_directory(in_dir, "", names);
    std::sort(names.begin(), names.end());
    for (const std::string& n : names) {
        const gpuastc::InputFormat* fmt = gpuastc::input_format_for(n);
        const std::string stem = n.substr(0, n.size() - std::strlen(fmt->ext));
        std::string out;
        if (pack) out = stem;
        else out = out_dir + "/" + (fmt->writes_same_container ? n : stem + ".astc");
        files.push_back(std::make_pair(in_dir + "/" + n, out));
    }
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
    bool premult = false;
    uint64_t src_size = 0, src_mtime = 0; // of the source PNG (for -pack -update)
};

// A source file read into memory, waiting for a decoder thread.
struct RawJob {
    size_t index = 0; // into the file list
    std::vector<uint8_t> bytes;
    bool read_ok = false;
    uint64_t src_size = 0, src_mtime = 0;
};

struct WriteJob {
    std::string path;     // output file, or the entry name in -pack mode
    bool raw = false;     // data is a whole file to be written as it is (content that is not a texture)
    int block = 4, w = 0, h = 0;
    bool alpha = false;
    uint64_t src_size = 0, src_mtime = 0;
    std::vector<uint8_t> data;
    std::vector<uint8_t> meta;    // non-empty: write the source container (.xnb) with ASTC inside
    const uint8_t* ext = nullptr; // -pack -update: blocks to copy from the old archive instead of `data`
    size_t ext_size = 0;
};

inline size_t byte_size(const RawJob& j) { return j.bytes.size(); }
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
    std::map<std::pair<int, int>, std::unique_ptr<gpuastc::Encoder>> encoders; // (block, alpha | premult << 1)
    GLuint tex = 0, buf = 0;
    WriteQueue* writer = nullptr; // finished files go here; a writer thread saves them
    int max_height = 4096; // tallest atlas we build
    int max_tex = 2048;
    bool normal = false, upload = false, verbose = false;
    int quality = 1;
    std::string cache_dir;
    double ms_shader = 0; // compiling or loading the shader variants
    int shaders_cached = 0, shaders_built = 0;

    // statistics
    double ms_gpu = 0;
    long long pixels = 0, bytes_out = 0;
    int files_ok = 0, files_failed = 0, dispatches = 0;

    gpuastc::Encoder* encoder(int block, bool alpha, bool premult)
    {
        auto key = std::make_pair(block, (alpha ? 1 : 0) | (premult ? 2 : 0));
        auto it = encoders.find(key);
        if (it != encoders.end()) return it->second.get();
        std::unique_ptr<gpuastc::Encoder> e(new gpuastc::Encoder());
        gpuastc::Options o;
        o.block = block;
        o.alpha = alpha;
        o.premultiplied = premult;
        o.normal_map = normal;
        o.quality = quality;
        o.cache_dir = cache_dir;
        std::string log;
        const double t0 = now_ms();
        const bool created = e->create(o, &log);
        ms_shader += now_ms() - t0;
        if (created) ++(e->from_cache() ? shaders_cached : shaders_built);
        if (!created) {
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
    bool premult = false;
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
    gpuastc::Encoder* enc = g.encoder(block, p.alpha, p.premult);
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
        wj.meta = std::move(p.jobs[k].img.meta);
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
        else if (!std::strcmp(argv[i], "-q") && i + 1 < argc) {
            const char* v = argv[++i];
            cli.quality = !std::strcmp(v, "fast") ? 0 : !std::strcmp(v, "best") ? 2 : 1;
        }
        else if (!std::strcmp(argv[i], "-j") && i + 1 < argc) cli.threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-mem") && i + 1 < argc) cli.mem_mb = std::max(16, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-update")) cli.update = true; // now the default
        else if (!std::strcmp(argv[i], "-force")) cli.update = false;
        else if (!std::strcmp(argv[i], "-premult")) cli.premult = true;
        else if (!std::strcmp(argv[i], "-prefetch") && i + 1 < argc) ++i; // older option: the reader thread replaces it
        else if (!std::strcmp(argv[i], "-nosync")) cli.sync = false;
        else if (!std::strcmp(argv[i], "-cache") && i + 1 < argc) cli.cache = argv[++i];
        else if (!std::strcmp(argv[i], "-nocache")) cli.use_cache = false;
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
        if (!list_directory(cli.in, cli.out, cli.pack, files)) { std::fprintf(stderr, "cannot read directory %s\n", cli.in.c_str()); return 1; }
        if (!cli.pack) mkdir(cli.out.c_str(), 0755); // ok if it already exists
        if (files.empty()) { std::fprintf(stderr, "no .png or .xnb files in %s\n", cli.in.c_str()); return 1; }
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
    gpu.quality = cli.quality;
    if (cli.use_cache) {
        if (cli.cache.empty()) {
            const char* xdg = std::getenv("XDG_CACHE_HOME");
            const char* home = std::getenv("HOME");
            if (xdg && *xdg) cli.cache = std::string(xdg) + "/gpuastc";
            else if (home && *home) cli.cache = std::string(home) + "/.cache/gpuastc";
        }
        if (!cli.cache.empty()) {
            std::set<std::string> made;
            make_parents(cli.cache + "/x", made);
            gpu.cache_dir = cli.cache;
        }
    }
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

    if (cli.mem_mb <= 0) {
        const int avail = available_memory_mb();
        cli.mem_mb = avail > 0 ? std::max(64, std::min(768, avail * 2 / 5)) : 384;
    }
    const size_t mem = (size_t)cli.mem_mb << 20;
    BoundedQueue<RawJob> raw_queue(mem / 6); // source files waiting for a decoder
    JobQueue queue(mem / 2);                 // decoded pixels waiting for the GPU
    WriteQueue write_queue(mem / 12);        // finished blocks on their way to the writer
    const size_t hold_bytes = mem / 6;       // results the writer keeps back while sources are still being read
    const long long max_pending_pixels = (long long)(mem / 6 / 4); // small RGBA images collected for packing
    gpu.writer = &write_queue;
    const std::string sync_dir = cli.pack ? (cli.out.find('/') == std::string::npos ? std::string(".") : cli.out.substr(0, cli.out.find_last_of('/')))
                                          : cli.out;

    std::atomic<int> write_failed(0);
    std::atomic<int> decode_failed(0);
    std::atomic<int> skipped(0);
    std::atomic<int> not_textures(0), unsupported(0), copied(0);
    std::atomic<bool> reading_done(false);
    double write_ms = 0, sync_ms = 0; // only touched by the writer thread until it is joined
    const int nreaders = (int)std::min<size_t>(4, files.size());
    std::vector<double> read_ms(nreaders, 0.0);
    std::atomic<size_t> next(0);
    std::atomic<int> readers_running(nreaders);
    const double wall0 = now_ms();

    // Writer: keeps results in memory while the reader still has source files to fetch (up to its
    // share of the budget), then writes what it holds in path order and the rest as it arrives.
    std::thread writer([&]() {
        std::set<std::string> made_dirs;
        std::vector<WriteJob> held;
        size_t held_size = 0;
        auto save = [&](WriteJob& wj) {
            const double t0 = now_ms();
            if (cli.pack) {
                pack_writer.add(wj);
            } else {
                make_parents(wj.path, made_dirs);
                bool ok;
                if (wj.raw) {
                    ok = write_file_atomic(wj.path, wj.data.data(), wj.data.size());
                    if (ok) ++copied;
                } else if (!wj.meta.empty()) {
                    const std::vector<uint8_t> x = gpuastc::build_astc_xnb(wj.meta, wj.block, wj.w, wj.h, wj.data.data(), wj.data.size());
                    ok = write_file_atomic(wj.path, x.data(), x.size());
                } else {
                    std::vector<uint8_t> a(16 + wj.data.size());
                    const uint8_t hdr[16] = {0x13, 0xAB, 0xA1, 0x5C, (uint8_t)wj.block, (uint8_t)wj.block, 1,
                                             (uint8_t)wj.w, (uint8_t)(wj.w >> 8), (uint8_t)(wj.w >> 16),
                                             (uint8_t)wj.h, (uint8_t)(wj.h >> 8), (uint8_t)(wj.h >> 16), 1, 0, 0};
                    std::memcpy(a.data(), hdr, 16);
                    std::memcpy(a.data() + 16, wj.data.data(), wj.data.size());
                    ok = write_file_atomic(wj.path, a.data(), a.size());
                }
                if (!ok) {
                    std::fprintf(stderr, "cannot write %s\n", wj.path.c_str());
                    ++write_failed;
                }
            }
            write_ms += now_ms() - t0;
        };
        auto save_held = [&]() {
            std::sort(held.begin(), held.end(), [](const WriteJob& a, const WriteJob& b) { return a.path < b.path; });
            for (WriteJob& h : held) save(h);
            held.clear();
            held_size = 0;
        };
        WriteJob wj;
        while (write_queue.pop(wj)) {
            // an archive is one file written front to back already: nothing to gain from holding
            if (!cli.pack && !reading_done.load() && held_size + byte_size(wj) <= hold_bytes) {
                held_size += byte_size(wj);
                held.push_back(std::move(wj));
                continue;
            }
            save_held();
            save(wj);
        }
        save_held();
    });

    // Readers: the only threads that touch the source files. They hand the bytes to the decoders
    // and run ahead of them as far as the memory budget allows. Several of them, because on the
    // FUSE file systems some firmwares use for the card, opening a file costs a round trip that
    // one thread alone turns into the bottleneck (measured: 1500 files, 8.8 s with one reader,
    // about 2.6 s with three).
    std::vector<std::thread> readers;
    for (int t = 0; t < nreaders; ++t) readers.emplace_back([&, t]() {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= files.size()) break;
            // The separate stat is only needed where a file may be skipped without reading it.
            const bool may_skip = (cli.pack && old_ok) || (cli.update && cli.batch && !cli.pack && files[i].first != files[i].second);
            struct stat st_src;
            const bool have_src = may_skip && stat(files[i].first.c_str(), &st_src) == 0;
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
            if (cli.update && cli.batch && !cli.pack && have_src && files[i].first != files[i].second) { // output already there and not older than the input: nothing to do
                struct stat st_out;
                if (stat(files[i].second.c_str(), &st_out) == 0 && st_out.st_size > 16 && st_out.st_mtime >= st_src.st_mtime) {
                    ++skipped;
                    continue;
                }
            }
            RawJob r;
            r.index = i;
            const double t0 = now_ms();
            r.read_ok = read_file_once(files[i].first.c_str(), r.bytes, r.src_size, r.src_mtime);
            read_ms[t] += now_ms() - t0;
            raw_queue.push(std::move(r));
        }
        if (--readers_running == 0) {
            reading_done = true;
            raw_queue.close();
        }
    });

    std::atomic<int> running(nthreads);
    std::vector<double> decode_ms(nthreads, 0.0);
    std::vector<std::thread> workers;
    for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back([&, t]() {
            RawJob r;
            while (raw_queue.pop(r)) {
                const size_t i = r.index;
                Job j;
                std::string err;
                const double t1 = now_ms();
                if (!r.read_ok) err = "cannot read file";
                const gpuastc::InputFormat* fmt = gpuastc::input_format_for(files[i].first);
                if (!r.read_ok || !fmt || !fmt->decode(r.bytes, j.img, &err)) {
                    if (r.read_ok && err == gpuastc::xnb_already_astc()) { ++skipped; continue; }
                    if (r.read_ok && fmt && fmt->writes_same_container && cli.batch) {
                        // A content file we do not convert (font, sound, model, unsupported texture format):
                        // carry it over unchanged so the output folder is a complete Content folder.
                        if (!cli.pack && files[i].first != files[i].second) {
                            WriteJob wj;
                            wj.path = files[i].second;
                            wj.raw = true;
                            wj.data = std::move(r.bytes);
                            write_queue.push(std::move(wj));
                        }
                        const bool real_problem = err == "LZX data is corrupt" || err.compare(0, 20, "surface format is no") == 0;
                        if (real_problem) { std::fprintf(stderr, "%s: %s (copied unchanged)\n", files[i].first.c_str(), err.c_str()); ++unsupported; }
                        else ++not_textures;
                        continue;
                    }
                    std::fprintf(stderr, "%s: %s\n", files[i].first.c_str(), err.c_str());
                    ++decode_failed;
                    continue;
                }
                decode_ms[t] += now_ms() - t1;
                j.alpha = cli.alpha_mode == 1 || (cli.alpha_mode == 0 && j.img.alpha);
                j.premult = j.alpha && (cli.premult || j.img.premultiplied);
                j.block = cli.block ? cli.block : gpuastc::Encoder::pick_block(j.img.w, j.img.h, j.alpha);
                j.out_path = files[i].second;
                j.src_size = r.src_size;
                j.src_mtime = r.src_mtime;
                r.bytes = std::vector<uint8_t>(); // free the source bytes before waiting on the queue
                queue.push(std::move(j));
            }
            if (--running == 0)
                queue.close();
        });
    }

    // GL thread: collect small images into packs, encode big ones directly.
    std::map<std::tuple<int, int, int>, Pack> packs; // (block, alpha | premult << 1, padded width)
    long long pending = 0;
    Job job;
    while (queue.pop(job)) {
        const int w = job.img.w, h = job.img.h;
        if ((long long)w * h > kSmallPixels) {
            Pack one;
            one.block = job.block;
            one.alpha = job.alpha;
            one.premult = job.premult;
            one.add(std::move(job));
            flush_pack(gpu, one);
            continue;
        }
        const int pw = Pack::pad(w, job.block), ph = Pack::pad(h, job.block);
        Pack& p = packs[std::make_tuple(job.block, (job.alpha ? 1 : 0) | (job.premult ? 2 : 0), pw)];
        if (p.jobs.empty()) { p.block = job.block; p.alpha = job.alpha; p.premult = job.premult; p.width = pw; }
        if (p.height + ph > gpu.max_height) { pending -= p.pixels; flush_pack(gpu, p); }
        pending += (long long)w * h;
        p.add(std::move(job));
        if (p.pixels >= kPackPixels) { pending -= p.pixels; flush_pack(gpu, p); }
        if (pending > max_pending_pixels) {
            for (auto& kv : packs) flush_pack(gpu, kv.second);
            pending = 0;
        }
    }
    for (auto& kv : packs) flush_pack(gpu, kv.second);
    for (std::thread& t : readers) t.join();
    for (std::thread& t : workers) t.join();
    write_queue.close();
    writer.join();
    bool pack_ok = true;
    if (cli.pack) {
        pack_ok = pack_writer.finish((uint32_t)cli.block, (uint32_t)cli.alpha_mode, cli.normal ? 1u : 0u);
        if (!pack_ok) std::fprintf(stderr, "cannot write archive %s\n", cli.out.c_str());
    }
    if (cli.batch && cli.sync) { // the run is not over until the results are on the storage
        const double t0 = now_ms();
        sync_storage(sync_dir);
        sync_ms += now_ms() - t0;
    }
    gpu.files_ok -= write_failed.load();
    gpu.files_failed += write_failed.load();
    const double wall = now_ms() - wall0;
    gpu.files_failed += decode_failed.load();

    double dec_total = 0, read_total = 0;
    for (double d : decode_ms) dec_total += d;
    for (double d : read_ms) read_total += d;
    if (cli.batch || cli.verbose) {
        if (not_textures.load() || unsupported.load())
            std::printf("%d .xnb files are not textures (fonts, sounds, ...) and %d use an unsupported texture format; %d copied unchanged\n",
                        not_textures.load(), unsupported.load(), copied.load());
        std::printf("%d files converted, %d up to date (skipped), %d failed, %.2f Mpx in %.0f ms (%.1f files/s, %.1f Mpx/s)\n",
                    gpu.files_ok, skipped.load(), gpu.files_failed, gpu.pixels / 1e6, wall,
                    wall > 0 ? gpu.files_ok * 1000.0 / wall : 0.0, wall > 0 ? gpu.pixels / 1e3 / wall : 0.0);
        std::printf("  file read %.0f ms across %d thread(s) + decode %.0f ms across %d thread(s), GPU upload+encode+readback %.0f ms in %d dispatch(es), file writes %.0f ms (own thread) + sync %.0f ms, output %.1f MB, memory budget %d MB\n",
                    read_total, nreaders, dec_total, nthreads, gpu.ms_gpu, gpu.dispatches, write_ms, sync_ms, gpu.bytes_out / 1e6, cli.mem_mb);
        std::printf("  shaders: %d compiled, %d from cache, %.0f ms\n", gpu.shaders_built, gpu.shaders_cached, gpu.ms_shader);
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
