"""How often does the two-partition search pay off? For each image: blocks that enter the search
(single-partition error above the threshold), blocks where a partition encoding won, and the
error removed, split by how closely the chosen pattern matched (PART_MAX_MISMATCH 0, 1, 2).
usage: attempts.py images..."""
import sys, numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, eval as ev

ctx = moderngl.create_standalone_context(require=430)

def encode(rgba, extra):
    ev.EXTRA = extra
    return ev.gpu(ctx, rgba, 4, bool((rgba[..., 3] != 255).any()), n=1)[1]

def block_err(rgba, data):
    h, w, _ = rgba.shape
    dec = t.decode(data, w, h, 4)[0]
    e = ((rgba.astype(float) - dec) ** 2)[..., :3].sum(-1)
    return e.reshape(h // 4, 4, w // 4, 4).sum((1, 3)).reshape(-1)

def parts(data):
    a = np.frombuffer(data, dtype="<u4").reshape(-1, 4)[:, 0]
    return np.where((a & 0x1FF) == 0x1FC, 0, ((a >> 11) & 3) + 1)

for p in sys.argv[1:]:
    rgba = np.array(Image.open(p).convert("RGBA")); h, w, _ = rgba.shape
    rgba = np.ascontiguousarray(rgba[:h - h % 4, :w - w % 4])
    single = encode(rgba, "#define USE_PARTITIONS 0\n"); es = block_err(rgba, single)
    n = len(es); tried = es > 256.0
    print(f"{p}: {n} blocks, {int((parts(single) == 0).sum())} flat, {int(tried.sum())} enter the search ({100 * tried.mean():.1f}%)")
    for mm in (0, 1, 2):
        for cand in (1, 2, 3):
            d = encode(rgba, f"#define PART_MAX_MISMATCH {mm}\n#define PART_CANDIDATES {cand}\n"); e = block_err(rgba, d)
            won = parts(d) == 2
            print(f"   mismatch<={mm} candidates {cand}: {int(won.sum()):6} won ({100 * won.sum() / max(tried.sum(), 1):5.1f}% of tried), error left {e.sum() / es.sum() * 100:5.1f}% of single")
