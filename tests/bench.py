"""Times the compute shader on desktop GL 4.3 and compares its blocks with a saved baseline.
usage: bench.py [--save-baseline] image...   (baseline stored in tests/baseline/)"""
import sys, time, pathlib, hashlib
import numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t

ROOT = pathlib.Path(__file__).resolve().parent
BASE = ROOT / "baseline"

def run(ctx, rgba, block, alpha, n=10):
    h, w, _ = rgba.shape
    prog = ctx.compute_shader(t.shader_source(block, alpha))
    tex = ctx.texture((w, h), 4, rgba.tobytes()); tex.filter = (moderngl.NEAREST,) * 2; tex.use(0); t.bind_lut(ctx)
    bx, by = (w + block - 1) // block, (h + block - 1) // block
    buf = ctx.buffer(reserve=bx * by * 16); buf.bind_to_storage_buffer(0)
    prog["uTexSize"].value = (w, h); prog["uBlocks"].value = (bx, by)
    if "uN" in prog: prog["uN"].value = block * block
    g = ((bx + 7) // 8, (by + 7) // 8, 1)
    prog.run(*g); ctx.finish()
    t0 = time.perf_counter()
    for _ in range(n): prog.run(*g)
    ctx.finish()
    ms = (time.perf_counter() - t0) / n * 1000
    return ms, buf.read()

def main():
    save = "--save-baseline" in sys.argv
    paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    ctx = moderngl.create_standalone_context(require=430)
    BASE.mkdir(exist_ok=True)
    print(f"{'image':28} {'blk':3} {'alpha':5} {'ms':>7} {'Mpx/s':>7}  vs baseline")
    for p in paths:
        img = np.array(Image.open(p).convert("RGBA")); h, w, _ = img.shape
        for alpha in (False, True):
            im = img.copy()
            if alpha and (im[..., 3] == 255).all():
                im[..., 3] = (np.arange(w)[None, :] * 255 // max(w - 1, 1)).astype(np.uint8)  # force alpha variation
            for block in (4, 5, 6):
                ms, data = run(ctx, im, block, alpha)
                key = BASE / f"{pathlib.Path(p).stem}_{block}_{int(alpha)}.bin"
                if save: key.write_bytes(data); status = "saved"
                elif key.exists():
                    base = key.read_bytes(); n = sum(1 for i in range(0, len(data), 16) if data[i:i+16] != base[i:i+16])
                    status = "identical" if n == 0 else f"{n}/{len(data)//16} blocks differ"
                else: status = "no baseline"
                print(f"{pathlib.Path(p).name[:28]:28} {block}x{block} {str(alpha):5} {ms:7.2f} {w*h/1e6/(ms/1000):7.0f}  {status}")

if __name__ == "__main__":
    main()
