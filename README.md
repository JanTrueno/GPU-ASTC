# GPU-ASTC

ASTC texture encoder that runs as an **OpenGL ES 3.1 compute shader**, for embedded Linux devices
without a window system (the GLES context comes from SDL2: KMSDRM or offscreen).

Status: builds and runs on Linux (tested on Ubuntu/WSL with Mesa llvmpipe). Output decodes correctly
with astcenc. Not yet run on a real Mali device.

## Usage

```bash
gpuastc in.png out.astc [-v]                 # one file
gpuastc -batch pngs/ out/ [-update] [-j N]   # every .png in a folder -> out/<name>.astc
gpuastc -pack pngs/ sprites.astcpak          # all textures in one archive (-update, -unpack supported)
```

Options: `-b auto|4|5|6` block size (auto: 4x4 up to 256x256, thin textures and anything with alpha;
5x5 up to 2048x2048; 6x6 above), `-alpha auto|on|off`, `-norm` normal map, `-upload` self-test of the GPU upload.

- **Batch:** one GL context, PNG decoding on worker threads, small images packed into one dispatch,
  files saved on a separate writer thread. `-update` skips PNGs whose output is already up to date.
- **Archive:** `include/astcpak.h` is a header-only C/C++ reader for games you can modify
  (`astcpak_open`, `astcpak_find`, `astcpak_gl_format`). Layout is documented at the top of the header.
- **Encoder:** single partition with 8-bit endpoints; 4x4 blocks with alpha also try a dual-plane mode
  (exact for "transparent + two colours" pixel art); flat blocks become constant-colour blocks.

## Build

```bash
sudo apt install build-essential cmake libsdl2-dev libgles2-mesa-dev libdeflate-dev
cmake -S . -B build && cmake --build build
SDL_VIDEODRIVER=offscreen ./build/gpuastc in.png out.astc -v
```

## Limits

Blocks with three or more opaque colours need partitions, which are not implemented, so hard-edged
multi-colour pixel art can look noticeably worse than astcenc. Smooth images land within about 1-2 dB of
astcenc's fast preset at 4x4.

On Mali-G31 (checked with `malioc`) every shader variant uses 32 registers except alpha 4x4 (57) and
alpha 5x5 (about 50). These are static estimates, not device measurements.

## Tests

`tests/gl_desktop_test.py`, `bench.py`, `pack_test.py` run the shader on desktop GL through moderngl and
compare with astcenc (`pip install moderngl astc-encoder-py pillow numpy`). `malioc_check.py` compiles it for
Mali. `build/png_compare` and `build/astcpak_test` check the PNG decoder and the archive.

Credits and licences: `THIRD_PARTY_NOTICES.md` (ported from niepp/astc_encoder, MIT).
