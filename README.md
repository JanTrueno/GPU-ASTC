# GPU-ASTC

ASTC LDR texture encoder that runs as an **OpenGL ES 3.1 compute shader**.
Aimed at aarch64 Linux embedded devices with no window system: the GLES
context comes from SDL2 (KMSDRM or offscreen video driver).

Status: the shader is validated on desktop OpenGL 4.3 (see Tests). The GLES 3.1
build, SDL2 host and `-upload` path have **not been built or run yet**.

## What it does

- Loads PNG (grey, grey+alpha, RGB, RGBA, palette, 16-bit) and converts to RGBA8.
- Encodes ASTC 4x4, 5x5 or 6x6 (`-b auto|4|5|6`), one block per GPU invocation.
- `-b auto` (default) picks the block size from the texture: 4x4 up to 256x256, for thin textures
  (shortest side <= 64, e.g. 1000x16) and for anything with alpha; 5x5 up to 2048x2048; 6x6 above.
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
