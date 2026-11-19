"""Runs shaders/astc_encode.comp on desktop OpenGL 4.3 (as #version 430), decodes the
result with astcenc and reports PSNR. Validates the algorithm and bit packing, not
GLSL ES compile rules. Needs: moderngl, astc-encoder-py, pillow, numpy."""
import sys, os, pathlib, math
import numpy as np
import moderngl
from PIL import Image
import astc_encoder as ae

ROOT = pathlib.Path(__file__).resolve().parent.parent

def shader_source(block, alpha, normal=False, premult=False, extra=""):
    body = pathlib.Path(os.environ.get("GPUASTC_SHADER", ROOT / "shaders/astc_encode.comp")).read_text()
    head = (f"#version 430\n#define BLOCK_DIM {block}\n#define HAS_ALPHA {int(alpha)}\n#define IS_NORMALMAP {int(normal)}\n"
            f"#define PREMULTIPLIED {int(premult)}\n{extra}")
    return head + body

_lut = {}
def bind_lut(ctx, block=4):
    """partition pattern table for the block size, texture unit 1"""
    if (ctx, block) not in _lut:
        import partition_lut
        if block == 4:
            tex = ctx.texture((256, 256), 1, partition_lut.lut().tobytes(), dtype="u4")
        else:
            tex = ctx.texture((256, 264), 1, partition_lut.big_table(block).tobytes(), dtype="u4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        _lut[(ctx, block)] = tex
    _lut[(ctx, block)].use(1)

def gpu_encode(ctx, rgba, block, alpha):
    h, w, _ = rgba.shape
    bx, by = (w + block - 1) // block, (h + block - 1) // block
    prog = ctx.compute_shader(shader_source(block, alpha))
    tex = ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes())
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.use(0)
    bind_lut(ctx, block)
    buf = ctx.buffer(reserve=bx * by * 16)
    buf.bind_to_storage_buffer(0)
    prog["uTexSize"].value = (w, h)
    prog["uBlocks"].value = (bx, by)
    if "uN" in prog: prog["uN"].value = block * block
    prog.run((bx + 7) // 8, (by + 7) // 8, 1)
    ctx.finish()
    return buf.read()

def decode(data, w, h, block):
    cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, block, block, 1, ae.ASTCQualityPreset.MEDIUM)
    actx = ae.ASTCContext(cfg)
    out = ae.ASTCImage(ae.ASTCType.U8, w, h, 1)
    actx.decompress(data, out, ae.ASTCSwizzle.from_str("RGBA"))
    return np.frombuffer(out.data, dtype=np.uint8).reshape(h, w, 4), actx

def cpu_encode(actx, rgba, block):
    h, w, _ = rgba.shape
    img = ae.ASTCImage(ae.ASTCType.U8, w, h, 1)
    img.data = np.ascontiguousarray(rgba).tobytes()
    return actx.compress(img, ae.ASTCSwizzle.from_str("RGBA"))

def psnr(a, b, channels):
    d = a[..., :channels].astype(np.float64) - b[..., :channels].astype(np.float64)
    mse = (d * d).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)

def synthetic(w, h, kind):
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(1)
    if kind == "gradient":
        r = x * 255 // max(w - 1, 1); g = y * 255 // max(h - 1, 1); b = (x + y) * 255 // (w + h - 2)
        a = np.full_like(r, 255)
    else:  # soft alpha ramp over colour gradient
        r = x * 255 // max(w - 1, 1); g = y * 255 // max(h - 1, 1); b = 255 - r
        a = x * 255 // max(w - 1, 1)
    return np.stack([r, g, b, a], -1).astype(np.uint8)

def main():
    ctx = moderngl.create_standalone_context(require=430)
    print("GL:", ctx.info["GL_RENDERER"], ctx.info["GL_VERSION"])
    cases = []
    for p in sys.argv[1:]:
        cases.append((pathlib.Path(p).name, np.array(Image.open(p).convert("RGBA"))))
    cases += [("gradient 61x45", synthetic(61, 45, "gradient")), ("alpha-ramp 64x64", synthetic(64, 64, "ramp"))]
    print(f"{'image':22} {'blk':3} {'alpha':5} {'GPU psnr':>9} {'astcenc medium':>15}")
    for name, rgba in cases:
        h, w, _ = rgba.shape
        has_a = bool((rgba[..., 3] != 255).any())
        ch = 4 if has_a else 3
        for block in (4, 5, 6):
            data = gpu_encode(ctx, rgba, block, has_a)
            dec, actx = decode(data, w, h, block)
            ref = cpu_encode(actx, rgba, block)
            refdec, _ = decode(ref, w, h, block)
            print(f"{name:22} {block}x{block} {str(has_a):5} {psnr(rgba, dec, ch):9.2f} {psnr(rgba, refdec, ch):15.2f}")

if __name__ == "__main__":
    main()
