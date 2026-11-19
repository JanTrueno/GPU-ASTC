"""Shader time on a large tiled copy of each image (desktop GL): usage speed.py [-b 4] [-D NAME=VALUE]... images"""
import sys, time, pathlib, numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, eval as ev
args = sys.argv[1:]; block = 4; extra = ""; paths = []; i = 0
while i < len(args):
    if args[i] == "-b": block = int(args[i + 1]); i += 1
    elif args[i] == "-D": extra += "#define " + args[i + 1].replace("=", " ") + "\n"; i += 1
    else: paths.append(pathlib.Path(args[i]))
    i += 1
ev.EXTRA = extra
ctx = moderngl.create_standalone_context(require=430)
out = []
for p in paths:
    rgba = np.array(Image.open(p).convert("RGBA")); h, w, _ = rgba.shape
    big = np.tile(rgba, (4096 // h + 1, 4096 // w + 1, 1))[:4096, :4096]
    has_a = bool((rgba[..., 3] != 255).any())
    ms, _ = ev.gpu(ctx, big, block, has_a, n=8)
    out.append(f"{p.stem}={ms:.1f}")
print(" ".join(out))
