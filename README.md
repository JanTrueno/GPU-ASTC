# GPU-ASTC

ASTC LDR texture encoder that runs as an **OpenGL ES 3.1 compute shader**.
Aimed at aarch64 Linux embedded devices with no window system: the GLES
context comes from SDL2 (KMSDRM or offscreen video driver).

Status: builds and runs on Linux (tested on Ubuntu/WSL with SDL2's offscreen driver and Mesa's
llvmpipe, GLES 3.2 context). Single-file and batch modes work, output decodes correctly with astcenc
and matches the desktop-GL encoder in 99.9% of blocks. Not yet run on a real Mali device.

## What it does

- Loads PNG (grey, grey+alpha, RGB, RGBA, palette, 16-bit) and converts to RGBA8.
- Encodes ASTC 4x4, 5x5 or 6x6 (`-b auto|4|5|6`), one block per GPU invocation.
- `-b auto` (default) picks the block size from the texture: 4x4 up to 256x256, for thin textures
  (shortest side <= 64, e.g. 1000x16) and for anything with alpha; 5x5 up to 2048x2048; 6x6 above.
- Transparency: RGBA endpoints, enabled automatically when any alpha < 255.
- Writes `.astc` files, and can upload straight into a `GL_COMPRESSED_RGBA_ASTC_*`
  texture through a pixel unpack buffer (`-upload`).

## Limits of the current encoder

4x4 blocks with alpha use a **dual-plane** mode when it beats the single-plane encoding: RGB and alpha get
separate binary weights, so "transparent + two opaque colours" blocks (typical pixel art) are exact. On 106
small sprites with alpha this raised the mean PSNR (measured on the blended result, alpha-weighted) from 38.0
to 43.2 dB (astcenc medium: 55.5); no image got worse. Remaining weakness: blocks with three or more *opaque*
colours (they need partitions), and smooth gradients in alpha textures. The alpha 4x4 variant needs 57
registers on Mali-G31 (50% occupancy); every other variant stays at 32.


Ported from niepp/astc_encoder: single partition, single weight plane, a fixed
4x4 weight grid, 8-bit endpoints and a fixed weight quantization. Quality is
that of a fast real-time encoder. Improving it is the point of this repo.

## Batch mode

```bash
gpuastc -batch textures/ out/ -v        # every .png in textures/ -> out/<name>.astc
```

One process, one GL context, one compiled shader per (block size, alpha) combination. Worker
threads decode PNGs (`-j N`, default cores - 1) while the GL thread encodes. Images up to
512x512 are packed: images with the same block size, alpha mode and padded width are stacked
into one tall texture (each padded to whole blocks with replicated edge pixels) and encoded in
a single dispatch, then each image's blocks are sliced out.
Blocks where every texel is the same colour (about 64% of the blocks in pixel-art sprites) are written as
constant-colour blocks after a single pass, skipping the rest of the work. Larger images are encoded one by one. Finished files are saved by a separate writer thread, so
slow storage (SD cards) overlaps with encoding. `-update` skips any PNG whose `.astc` already exists
and is not older, so re-running a folder only converts what changed.
`tests/pack_test.py` checks the packing on the desktop GPU (each stacked image must encode
identically to the image on its own).

PNG decoding uses libdeflate when it is available at build time (`apt install libdeflate-dev`),
which is about 1.4-1.5x faster than stb_image here (inflate is only ~20% of the time, the PNG
filters and pixel conversion the rest); unusual PNGs (interlaced) fall back to stb_image.
`build/png_compare *.png` checks the two decoders against each other.

## Archives (for games you can modify)

```bash
gpuastc -pack sprites/ sprites.astcpak -v     # every .png in sprites/ -> one file
gpuastc -pack sprites/ sprites.astcpak -update  # rerun: unchanged entries are copied from the old archive
gpuastc -unpack sprites.astcpak out/          # back to one .astc file per texture
```

One sequential write instead of thousands of small files (much kinder to SD cards), and the game opens
a single file. Entries are named after the PNG (without `.png`), sorted for lookup, and identical textures
share one copy of their data (23 of 159 in one test set; the archive was 12% smaller than the separate files).
`-pack -update` stores each source PNG's size and modification time and the options used, so it only
re-encodes what changed, and rebuilds everything when `-b`/`-alpha`/`-norm` change. The archive is written to
`<name>.tmp` and renamed when complete, so an interrupted run never leaves a half-written file.

Reading it from a game is one header, `include/astcpak.h` (C99/C++, no dependencies). Map or read the whole
file once, then each texture is a pointer into it:

```c
astcpak pak;
astcpak_open(&pak, file_data, file_size);          // returns 0 if the file is not a usable archive
astcpak_entry e;
if (astcpak_find(&pak, "player_idle_0", &e))       // binary search by name
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, astcpak_gl_format(&e),
                           e.width, e.height, 0, (GLsizei)e.data_size, e.blocks);
```

The file layout is described at the top of that header. `build/astcpak_test pack.astcpak files_dir` checks an
archive against the per-file output of the same inputs.

## Build

```bash
sudo apt install build-essential cmake libsdl2-dev libgles2-mesa-dev libdeflate-dev
cmake -S . -B build && cmake --build build
SDL_VIDEODRIVER=offscreen ./build/gpuastc in.png out.astc -v
```

On a device with KMSDRM, leave `SDL_VIDEODRIVER` unset or set it to `kmsdrm`.

## Performance notes

Dynamic indexing of large `const` arrays in a shader makes many GPUs keep a per-thread copy
in slow memory. The original port spent about 99% of its time in the trit and weight-scramble
tables. Replacing them with arithmetic gave bit-identical output and ~95x speed on an RTX 4060
(20 ms -> 0.2 ms for 9 Mpx). The later Mali work changes the output by floating-point rounding only
(PSNR within 0.05 dB), so `compare_baseline.py` checks quality instead of bit equality. `tests/bench.py` and `tests/compare_baseline.py` time the shader and compares against saved
baseline blocks (`--save-baseline` to create them).

## Mali-G31 (RK3326, H700) shader stats

Measured with Arm's `malioc` (`tests/malioc_check.py`), GLSL ES 3.10. The shader compiles for the G31.

| | before | after |
|---|---|---|
| 4x4 registers / occupancy | 64 / 50%, 368-424 B spilled | 32 / 100%, no stack |
| 4x4 critical path (cycles) | 231-239, load/store bound | ~155, arithmetic bound |
| 6x6 registers / occupancy | 64 / 50%, 640-768 B stack | 51-55 / 50%, 64 B stack |
| 6x6 critical path (cycles) | ~303 arithmetic, 288 load/store | ~400-416 arithmetic, 25 load/store |

5x5 uses the same code path as 6x6 (51-56 registers, 50% occupancy, 64 B stack).

How: texels are re-fetched instead of held in registers, weights are quantized in a
second pass (4x4) or stored (6x6), and the power iteration runs 3 times instead of 8 (no
measurable quality loss). The loop bounds come from the `uN` uniform so the compiler does not
unroll them; the host must set `uN` to the block size squared. `malioc` cannot give a longest
path for rolled loops, so the cycle rows come from an unrolled copy of the same code. 6x6 trades
more arithmetic for far less memory traffic and is not clearly faster. These are static
estimates, not measurements on a device.

## Tests

`tests/gl_desktop_test.py` runs the shader as `#version 430` through moderngl, decodes
it with astcenc (`astc-encoder-py`) and reports PSNR next to astcenc's own encode.
It checks the algorithm and bit packing, not GLSL ES compile rules.

```bash
python -m venv venv && venv/Scripts/pip install moderngl astc-encoder-py pillow numpy
venv/Scripts/python tests/gl_desktop_test.py some.png
```

## Layout

- `shaders/astc_encode.comp` encoder (no lookup tables)
- `src/astc_gles.*` reusable `gpuastc::Encoder` (no windowing code)
- `src/main.cpp` command-line tool

Credits and licences: see `THIRD_PARTY_NOTICES.md`.
