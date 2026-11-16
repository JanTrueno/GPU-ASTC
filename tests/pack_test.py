"""Checks the batch packing scheme on desktop GL: images of the same padded width are stacked
into one tall texture (each padded to whole blocks with replicated edge pixels), encoded in one
dispatch, and each image's slab of blocks must equal that image encoded on its own.
usage: pack_test.py dir_of_pngs"""
import sys, pathlib, collections
import numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, bench

ctx = moderngl.create_standalone_context(require=430)
files = sorted(pathlib.Path(sys.argv[1]).rglob("*.png"))
groups = collections.defaultdict(list)
for f in files:
    im = np.array(Image.open(f).convert("RGBA"))
    h, w, _ = im.shape
    if w * h > 1 << 18: continue
    alpha = bool((im[..., 3] != 255).any()); block = 4
    groups[(block, alpha, (w + 3) // 4 * 4)].append((f.name, im))
checked = bad = 0
for (block, alpha, pw), items in groups.items():
    if len(items) < 2: continue
    items = items[:60]
    rows, atlas_rows = [], []
    for name, im in items:
        h, w, _ = im.shape; ph = (h + 3) // 4 * 4
        pad = np.empty((ph, pw, 4), np.uint8)
        for y in range(ph):
            row = im[min(y, h - 1)]
            pad[y, :w] = row; pad[y, w:] = row[w - 1]
        rows.append(sum(a.shape[0] for a in atlas_rows)); atlas_rows.append(pad)
    atlas = np.concatenate(atlas_rows, 0)
    _, packed = bench.run(ctx, atlas, block, alpha, n=1)
    bx = pw // 4
    for (name, im), r0 in zip(items, rows):
        h, w, _ = im.shape
        _, solo = bench.run(ctx, im, block, alpha, n=1)
        slab = packed[(r0 // 4) * bx * 16 : (r0 // 4 + (h + 3) // 4) * bx * 16]
        checked += 1
        if slab != solo:
            bad += 1; print("MISMATCH", name, w, h)
print(f"{len(groups)} size groups, {checked} images checked, {bad} mismatches")
