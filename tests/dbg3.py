import sys, numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, eval as ev, partition_lut as pl
ctx = moderngl.create_standalone_context(require=430)
rgba = np.array(Image.open(sys.argv[1]).convert("RGBA")); h, w, _ = rgba.shape
_, data = ev.gpu(ctx, rgba, 4, False, n=1)
dec = t.decode(data, w, h, 4)[0]
lut = pl.lut(); dist = (lut >> 10) & 31; pats = pl.patterns()
err = ((rgba.astype(float) - dec) ** 2).sum(-1).reshape(h // 4, 4, w // 4, 4).sum((1, 3))
v = (rgba[..., 0].astype(np.int64) << 16) | (rgba[..., 1].astype(np.int64) << 8) | rgba[..., 2]
b = v.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)
shown = 0; stats = {}
for i, r in enumerate(b):
    cols = sorted(set(r))
    if len(cols) != 3: continue
    ds = [int(dist[sum(int(r[k] == c) << k for k in range(16))]) for c in cols]
    if min(ds) != 0: continue
    y, x = divmod(i, w // 4); e = err[y, x]
    blk = int.from_bytes(data[i*16:i*16+16], "little"); parts = ((blk >> 11) & 3) + 1
    key = (parts, "bad" if e > 400 else "ok"); stats[key] = stats.get(key, 0) + 1
    if e > 2000 and shown < 3:
        shown += 1
        seed = (blk >> 13) & 1023
        print(f"block {x},{y} err {e:.0f} parts {parts} seed {seed} pattern {pats[seed]:016b} dists {ds}")
        for c in cols: print(f"  colour {c:06x} mask {sum(int(r[k] == c) << k for k in range(16)):016b}")
        print("  dec", sorted(set(map(tuple, dec[y*4:y*4+4, x*4:x*4+4, :3].reshape(-1, 3).tolist()))))
print(stats)
