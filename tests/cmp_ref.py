"""Where does astcenc beat the shader? Error by distinct-colour count and by astcenc's block type."""
import sys, numpy as np, moderngl
from PIL import Image
import astc_encoder as ae
import gl_desktop_test as t, eval as ev
ctx = moderngl.create_standalone_context(require=430)
rgba = np.array(Image.open(sys.argv[1]).convert("RGBA")); b = 4; h, w, _ = rgba.shape
h -= h % 4; w -= w % 4; rgba = np.ascontiguousarray(rgba[:h, :w])
has_a = bool((rgba[..., 3] != 255).any())
_, mine = ev.gpu(ctx, rgba, b, has_a, n=1)
cfg = ae.ASTCConfig(ae.ASTCProfile.LDR, b, b, 1, ae.ASTCQualityPreset.FAST); c = ae.ASTCContext(cfg)
img = ae.ASTCImage(ae.ASTCType.U8, w, h, 1); img.data = rgba.tobytes()
ref = c.compress(img, ae.ASTCSwizzle.from_str("RGBA"))
def berr(data):
    dec = t.decode(data, w, h, b)[0]
    a = rgba.astype(float); d = dec.astype(float)
    if has_a:  # as eval.py scores alpha images: premultiplied colour + alpha
        a = np.concatenate([a[..., :3] * a[..., 3:4] / 255.0, a[..., 3:4]], -1)
        d = np.concatenate([d[..., :3] * d[..., 3:4] / 255.0, d[..., 3:4]], -1)
    e = ((a - d) ** 2).sum(-1)
    return e.reshape(h // 4, 4, w // 4, 4).sum((1, 3)).reshape(-1)
em, er = berr(mine), berr(ref)
v = (rgba[..., 0].astype(np.int64) << 24) | (rgba[..., 1].astype(np.int64) << 16) | (rgba[..., 2].astype(np.int64) << 8) | rgba[..., 3]
nc = np.array([len(set(r)) for r in v.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)])
def kind(data, i):
    blk = int.from_bytes(data[i*16:i*16+16], "little")
    if (blk & 0x1FF) == 0x1FC: return "void"
    return f"p{((blk >> 11) & 3) + 1}{'d' if blk & 0x400 else ''}"
print("total err mine %.3g ref %.3g" % (em.sum(), er.sum()))
print("colours  blocks   mine-err    ref-err")
for k in list(range(2, 9)) + [99]:
    m = (nc == k) if k < 99 else (nc > 8)
    print(f"{k:7} {m.sum():7} {em[m].sum():10.3g} {er[m].sum():10.3g}")
kinds = {}
for i in range(len(em)):
    e = kinds.setdefault(kind(ref, i), [0, 0.0, 0.0]); e[0] += 1; e[1] += em[i]; e[2] += er[i]
print("astcenc type  blocks   mine-err    ref-err")
for k, (n, a, r) in sorted(kinds.items()): print(f"{k:12} {n:7} {a:10.3g} {r:10.3g}")
