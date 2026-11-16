"""Decodes the saved baseline blocks and the current shader's blocks with astcenc and compares PSNR
against the source image. Use this when bit-identical output is not expected (float rounding)."""
import sys, pathlib
import numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, bench

ctx = moderngl.create_standalone_context(require=430)
print(f"{'image':22} {'blk':3} {'alpha':5} {'baseline dB':>11} {'current dB':>11} {'identical':>10}")
for p in sys.argv[1:]:
    img = np.array(Image.open(p).convert("RGBA")); h, w, _ = img.shape
    for alpha in (False, True):
        im = img.copy()
        if alpha and (im[..., 3] == 255).all():
            im[..., 3] = (np.arange(w)[None, :] * 255 // max(w - 1, 1)).astype(np.uint8)
        ch = 4 if alpha else 3
        for block in (4, 5, 6):
            _, data = bench.run(ctx, im, block, alpha, n=1)
            base = (bench.BASE / f"{pathlib.Path(p).stem}_{block}_{int(alpha)}.bin").read_bytes()
            d1, _ = t.decode(base, w, h, block); d2, _ = t.decode(data, w, h, block)
            same = sum(1 for i in range(0, len(data), 16) if data[i:i+16] == base[i:i+16]) / (len(data) // 16)
            print(f"{pathlib.Path(p).name[:22]:22} {block}x{block} {str(alpha):5} {t.psnr(im, d1, ch):11.3f} {t.psnr(im, d2, ch):11.3f} {same*100:9.1f}%")
