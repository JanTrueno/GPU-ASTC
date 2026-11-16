# GPU-ASTC

ASTC LDR texture encoder that runs as an **OpenGL ES 3.1 compute shader**.
Aimed at aarch64 Linux embedded devices with no window system: the GLES
context comes from SDL2 (KMSDRM or offscreen video driver).

Status: first port, **not yet built or tested**.

## What it does

- Loads PNG (grey, grey+alpha, RGB, RGBA, palette, 16-bit) and converts to RGBA8.
- Encodes ASTC 4x4 or 6x6, one block per GPU invocation.
- Transparency: RGBA endpoints, enabled automatically when any alpha < 255.
- Writes `.astc` files, and can upload straight into a `GL_COMPRESSED_RGBA_ASTC_*`
  texture through a pixel unpack buffer (`-upload`).

## Limits of the current encoder

Ported from niepp/astc_encoder: single partition, single weight plane, a fixed
4x4 weight grid, 8-bit endpoints and a fixed weight quantization. Quality is
that of a fast real-time encoder. Improving it is the point of this repo.

## Build

```bash
sudo apt install build-essential cmake libsdl2-dev libgles2-mesa-dev
cmake -S . -B build && cmake --build build
SDL_VIDEODRIVER=offscreen ./build/gpuastc in.png out.astc -v
```

On a device with KMSDRM, leave `SDL_VIDEODRIVER` unset or set it to `kmsdrm`.

## Layout

- `shaders/astc_encode.comp` encoder, `shaders/astc_tables.glsl` lookup tables
- `src/astc_gles.*` reusable `gpuastc::Encoder` (no windowing code)
- `src/main.cpp` command-line tool

Credits and licences: see `THIRD_PARTY_NOTICES.md`.
