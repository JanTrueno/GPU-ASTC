# GPU-ASTC

ASTC texture encoder that runs as an **OpenGL ES 3.1 compute shader**, for embedded Linux devices
without a window system (the GLES context comes from SDL2: KMSDRM or offscreen).

It's what compresses Terraria's textures on the handheld itself in our PortMaster port.
It also runs on desktop Linux (Mesa, llvmpipe), and its output decodes correctly with astcenc.

## Usage

```bash
gpuastc in.png out.astc [-v]                 # one file
gpuastc -batch pngs/ out/ [-update] [-j N]   # every .png in a folder -> out/<name>.astc
gpuastc -pack pngs/ sprites.astcpak          # all textures in one archive (-unpack supported)
gpuastc -batch Content/ Content/             # XNA/FNA .xnb textures -> ASTC .xnb, in place
```

`.png` and `.xnb` (LZX-compressed or not, Color format) are read; subfolders are included. XNB output keeps the file name
and holds ASTC (surface format values 25-28 follow from the RHH-Ports FNA3D ASTC patch (appended after Bc7SrgbEXT = 24), but are not yet checked against a converted file). Non-texture .xnb files are copied
unchanged when writing to another folder. XNA colour is premultiplied by alpha; the encoder handles that automatically for .xnb
(`-premult` forces it for PNG). Files whose output is already newer than the input are skipped; `-force` converts everything.
New input formats: one decode function plus a line in `src/formats.cpp`.

Options: `-q fast|normal|best` 4x4 search effort (see Encoder), `-b auto|4|5|6` block size (auto: 4x4 up to 256x256, thin textures and anything with alpha;
5x5 up to 2048x2048; 6x6 above), `-alpha auto|on|off`, `-norm` normal map, `-upload` self-test of the GPU upload,
`-mem MB` memory for source files, images and results waiting between reading, encoding and saving (default: 40% of
the free memory, 64 to 768), `-cache DIR` / `-nocache` where compiled shaders are kept between runs (default
`$XDG_CACHE_HOME/gpuastc` or `~/.cache/gpuastc`), `-nosync` do not wait for the results to reach the storage.

- **Batch:** one GL context, reader threads that fetch the source files ahead of the decoders, decoding on
  worker threads, small images packed into one dispatch, results held in memory until the reading is done and
  then saved in path order by a writer thread. Compiled shaders are cached: on Mali-G31 compiling the three
  common variants takes about 2 s per run, loading them 0.03 s.
- **Archive:** `include/astcpak.h` is a header-only C/C++ reader for games you can modify
  (`astcpak_open`, `astcpak_find`, `astcpak_gl_format`). Layout is documented at the top of the header.
- **Encoder:** one weight per texel at every block size (4x4: 12 levels, 5x5: 5, 6x6: 3; fewer for blocks
  with alpha), 8-bit endpoints refitted to the chosen weights; flat blocks become constant-colour blocks;
  opaque blocks of an alpha texture are stored as RGB. 4x4 blocks that one colour line fits badly also try
  two partitions (`-q normal`, the default; `-q best` tries more, `-q fast` none) and, with alpha, a
  dual-plane mode (exact for "transparent + two colours" pixel art). Smooth opaque 5x5 and 6x6 blocks
  also try a 4x4 grid of 12-level weights and keep whichever has the lower error.

## Build

```bash
sudo apt install build-essential cmake libsdl2-dev libgles2-mesa-dev libdeflate-dev
cmake -S . -B build && cmake --build build
SDL_VIDEODRIVER=offscreen ./build/gpuastc in.png out.astc -v
```

## Limits

Two partitions only exist for 4x4, and three or four are not implemented, so blocks with four or more
unrelated colours (and hard edges at 5x5 / 6x6) still look worse than astcenc. On a mixed set of photos,
screenshots and sprites, 4x4 lands about 3 dB under astcenc's fast preset (2 to 6 dB per image), 5x5 and
6x6 about 4 dB.

Measured on Mali-G31 with a batch of 12 images of 1 Mpx (photo, sprite sheets, screenshot), GPU time for
upload + encode + readback in ms, against the encoder this one started from:

| | 4x4 `-q fast` | 4x4 `-q normal` | 5x5 | 6x6 | before: 4x4 / 5x5 / 6x6 |
|---|---|---|---|---|---|
| Arm's driver (RG351P) | 380 | 545 | 410 | 445 | 540 / 590 / 545 |
| Mesa Panfrost (RG-34XXSP) | 510 | 780 | 635 | 645 | 775 / 690 / 610 |

Time batches, not single images: the first dispatch of a run carries one-time costs. With the shader cache a
repeat run of that batch takes about 1.1 s in all on the RG351P (2.9 s before, most of it compiling).
The pixels go to the shader through a buffer: uploading a
texture for a single image took 35 to 40 ms per megapixel on Arm's driver. `GPUASTC_INPUT=texture` goes back
to a texture, `GPUASTC_TIMING=1` reports the GPU stages separately, `GPUASTC_SHADER=<file>` replaces the
built-in shader source.
`GPUASTC_SHADER_DEFINES="NAME=VALUE;..."` overrides the shader's tuning defines (`ALT_ERR_THRESHOLD`,
`PART_CANDIDATES`, `PART_MAX_MISMATCH`, `GRID_SMOOTH`, `GRID_ERR_THRESHOLD`, `USE_GRID`) for experiments.

## Tests

`tests/gl_desktop_test.py`, `bench.py`, `pack_test.py` run the shader on desktop GL through moderngl and
compare with astcenc; `eval.py` prints a quality table for a folder of images (`--ref` adds astcenc),
`cmp_ref.py` shows which kinds of block astcenc encodes better, `speed.py` times large images (`pip install moderngl astc-encoder-py pillow numpy`). `malioc_check.py` compiles it for
Mali. `build/png_compare` and `build/astcpak_test` check the PNG decoder and the archive.

Credits and licences: `THIRD_PARTY_NOTICES.md` (ported from niepp/astc_encoder, MIT).
