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
`-mem MB` memory for images waiting between reading, encoding and saving (default 384; lower it on 1 GB devices).

- **Batch:** one GL context, decoding on worker threads, small images packed into one dispatch,
  files saved on a separate writer thread.
- **Archive:** `include/astcpak.h` is a header-only C/C++ reader for games you can modify
  (`astcpak_open`, `astcpak_find`, `astcpak_gl_format`). Layout is documented at the top of the header.
- **Encoder:** one weight per texel at every block size (4x4: 12 levels, 5x5: 5, 6x6: 3; fewer for blocks
  with alpha), 8-bit endpoints refitted to the chosen weights; flat blocks become constant-colour blocks;
  opaque blocks of an alpha texture are stored as RGB. 4x4 blocks that one colour line fits badly also try
  two partitions (`-q normal`, the default; `-q best` tries more, `-q fast` none) and, with alpha, a
  dual-plane mode (exact for "transparent + two colours" pixel art).

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
6x6 about 5 dB. Very smooth gradients at 6x6 get only 3 weight levels and can band slightly.

Measured on Mali-G31 (1 Mpx images, upload + encode + readback): 5x5 and 6x6 take 60 to 70 ms, 4x4 takes
70 ms with `-q fast` and 80 to 100 ms with `-q normal` on Arm's driver. On Panfrost the two-partition
search is much slower (120 to 200 ms); use `-q fast` there if speed matters more.
`GPUASTC_SHADER_DEFINES="NAME=VALUE;..."` overrides the shader's tuning defines (`ALT_ERR_THRESHOLD`,
`PART_CANDIDATES`, `PART_MAX_MISMATCH`) for experiments.

## Tests

`tests/gl_desktop_test.py`, `bench.py`, `pack_test.py` run the shader on desktop GL through moderngl and
compare with astcenc; `eval.py` prints a quality table for a folder of images (`--ref` adds astcenc),
`cmp_ref.py` shows which kinds of block astcenc encodes better, `speed.py` times large images (`pip install moderngl astc-encoder-py pillow numpy`). `malioc_check.py` compiles it for
Mali. `build/png_compare` and `build/astcpak_test` check the PNG decoder and the archive.

Credits and licences: `THIRD_PARTY_NOTICES.md` (ported from niepp/astc_encoder, MIT).
