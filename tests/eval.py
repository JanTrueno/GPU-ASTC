"""Quality + speed table over a corpus: GPU shader vs astcenc (fastest/fast/medium).
usage: eval.py [-b 4,5,6] [--ref] [--json out.json] image-or-folder...
Alpha images are scored on premultiplied RGB + A (colour under transparent pixels is not counted)."""
import sys, time, pathlib, math, json
import numpy as np, moderngl
from PIL import Image
import astc_encoder as ae
import gl_desktop_test as t

EXTRA = ""

def score(src, dec, has_a):
    s = src.astype(np.float64); d = dec.astype(np.float64)
    if has_a:
        s = np.concatenate([s[..., :3] * s[..., 3:4] / 255.0, s[..., 3:4]], -1)
        d = np.concatenate([d[..., :3] * d[..., 3:4] / 255.0, d[..., 3:4]], -1)
    else:
        s = s[..., :3]; d = d[..., :3]
    mse = ((s - d) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse)

def gpu(ctx, rgba, block, alpha, n=5):
    h, w, _ = rgba.shape
    prog = ctx.compute_shader(t.shader_source(block, alpha, extra=EXTRA))
    tex = ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes()); tex.filter = (moderngl.NEAREST,) * 2; tex.use(0); t.bind_lut(ctx, block)
    bx, by = (w + block - 1) // block, (h + block - 1) // block
    if "INPUT_BUFFER 1" in EXTRA:  # the shader reads the pixels from a buffer at binding 1
        ctx.buffer(np.ascontiguousarray(rgba).tobytes()).bind_to_storage_buffer(1)
    buf = ctx.buffer(reserve=bx * by * 16); buf.bind_to_storage_buffer(0)
    prog["uTexSize"].value = (w, h); prog["uBlocks"].value = (bx, by)
    if "uN" in prog: prog["uN"].value = block * block
    g = ((bx + 7) // 8, (by + 7) // 8, 1)
    prog.run(*g); ctx.finish()
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); prog.run(*g); ctx.finish(); best = min(best, time.perf_counter() - t0)
    data = buf.read(); tex.release(); buf.release(); prog.release()
    return best * 1000, data

def ref(rgba, block, preset):
    h, w, _ = rgba.shape
    cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, block, block, 1, preset)
    c = ae.ASTCContext(cfg)
    img = ae.ASTCImage(ae.ASTCType.U8, w, h, 1); img.data = np.ascontiguousarray(rgba).tobytes()
    data = c.compress(img, ae.ASTCSwizzle.from_str("RGBA"))
    return t.decode(data, w, h, block)[0]

def main():
    global EXTRA
    args = sys.argv[1:]
    blocks = (4, 5, 6); want_ref = False; jout = None; paths = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-b": blocks = tuple(int(x) for x in args[i + 1].split(",")); i += 1
        elif a == "--ref": want_ref = True
        elif a == "-D": EXTRA += "#define " + args[i + 1].replace("=", " ") + chr(10); i += 1
        elif a == "--json": jout = args[i + 1]; i += 1
        else:
            p = pathlib.Path(a); paths += sorted(p.glob("*.png")) if p.is_dir() else [p]
        i += 1
    ctx = moderngl.create_standalone_context(require=430)
    res = {}
    hdr = f"{'image':14} {'blk':3} {'a':1} {'GPU dB':>7} {'ms':>6}"
    if want_ref: hdr += f" {'fastest':>7} {'fast':>7} {'medium':>7}"
    print(hdr)
    tot = {}
    for p in paths:
        rgba = np.array(Image.open(p).convert("RGBA")); h, w, _ = rgba.shape
        has_a = bool((rgba[..., 3] != 255).any())
        for b in blocks:
            ms, data = gpu(ctx, rgba, b, has_a)
            db = score(rgba, t.decode(data, w, h, b)[0], has_a)
            line = f"{p.stem[:14]:14} {b}x{b} {int(has_a)} {db:7.2f} {ms:6.2f}"
            r = {"db": db, "ms": ms}
            if want_ref:
                for name, pre in (("fastest", ae.ASTCQualityPreset.FASTEST), ("fast", ae.ASTCQualityPreset.FAST), ("medium", ae.ASTCQualityPreset.MEDIUM)):
                    r[name] = score(rgba, ref(rgba, b, pre), has_a); line += f" {r[name]:7.2f}"
            print(line, flush=True)
            res[f"{p.stem}_{b}"] = r
            for k, v in r.items(): tot.setdefault((b, k), []).append(v)
    for b in blocks:
        line = f"{'MEAN':14} {b}x{b}   {np.mean(tot[(b,'db')]):7.2f} {np.sum(tot[(b,'ms')]):6.1f}"
        if want_ref: line += "".join(f" {np.mean(tot[(b,k)]):7.2f}" for k in ("fastest", "fast", "medium"))
        print(line)
    if jout: pathlib.Path(jout).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
