"""Scores .astc files encoded on a device (tests/device_encode.sh, q.tar unpacked) against their source images,
next to this PC's encode of the same image with the same shader. A driver that compiles the shader wrongly
shows up as a difference; so far every correct driver has been within 0.01 dB.
usage: device_score.py <folder with the source .png files> <unpacked q folder> [more q folders...]"""
import sys, pathlib
import numpy as np, moderngl
from PIL import Image
import gl_desktop_test as t, eval as ev

def main():
    src = pathlib.Path(sys.argv[1])
    ctx = moderngl.create_standalone_context(require=430)
    ref = {}
    for q in sys.argv[2:]:
        worst = 0.0; lines = []
        for f in sorted(pathlib.Path(q).glob("*.astc")):
            name, b = f.stem.rsplit("_", 1); b = int(b)
            rgba = np.array(Image.open(src / f"{name}.png").convert("RGBA")); h, w, _ = rgba.shape
            a = bool((rgba[..., 3] != 255).any())
            got = ev.score(rgba, t.decode(f.read_bytes()[16:], w, h, b)[0], a)
            if (name, b) not in ref:
                ev.EXTRA = ""
                ref[(name, b)] = ev.score(rgba, t.decode(ev.gpu(ctx, rgba, b, a, n=1)[1], w, h, b)[0], a)
            d = got - ref[(name, b)]; worst = min(worst, d)
            lines.append(f"  {name} {b}x{b}: {got:.2f} dB ({d:+.2f})")
        print(f"{q}: worst difference to this PC {worst:+.2f} dB")
        print("\n".join(lines))

if __name__ == "__main__":
    main()
