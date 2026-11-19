import sys, numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, eval as ev
ctx = moderngl.create_standalone_context(require=430)
rgba = np.array(Image.open(sys.argv[1]).convert("RGBA")); b = int(sys.argv[2]); h, w, _ = rgba.shape
has_a = bool((rgba[..., 3] != 255).any())
_, data = ev.gpu(ctx, rgba, b, has_a, n=1)
dec = t.decode(data, w, h, b)[0]
err = ((rgba.astype(float) - dec) ** 2).sum(-1)
bx, by = (w + b - 1) // b, (h + b - 1) // b
pad = np.zeros((by * b, bx * b)); pad[:h, :w] = err
be = pad.reshape(by, b, bx, b).sum((1, 3))
kinds = {}
for i in range(bx * by):
    blk = int.from_bytes(data[i*16:i*16+16], "little")
    k = "void" if (blk & 0x1FF) == 0x1FC else f"mode{blk & 0x7FF:03x}_p{((blk >> 11) & 3) + 1}"
    e = kinds.setdefault(k, [0, 0.0]); e[0] += 1; e[1] += be[i // bx, i % bx]
for k, (n, e) in sorted(kinds.items()): print(f"{k:14} blocks {n:7} err/block {e/n:10.1f}")
y, x = np.unravel_index(be.argmax(), be.shape)
print("worst block", x, y, "err", be[y, x], "bits %032x" % int.from_bytes(data[(y*bx+x)*16:(y*bx+x)*16+16], "little"))
print("src\n", rgba[y*b:y*b+b, x*b:x*b+b].reshape(-1, 4)); print("dec\n", dec[y*b:y*b+b, x*b:x*b+b].reshape(-1, 4))
