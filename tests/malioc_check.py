"""Compiles each shader variant as GLSL ES 3.10 with Arm's malioc and prints the key stats.
usage: malioc_check.py [core] [path-to-malioc.exe]"""
import subprocess, sys, pathlib, tempfile, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
core = args[0] if args else "Mali-G31"
exe = args[1] if len(args) > 1 else "C:/Program Files/Arm/Arm Performance Studio 2026.6/mali_offline_compiler/malioc.exe"
src = (ROOT / "shaders/astc_encode.comp").read_text()
with tempfile.TemporaryDirectory() as d:
    for block in (4, 5, 6):
        for alpha in (0, 1):
            f = pathlib.Path(d) / f"astc_{block}_{alpha}.comp"
            f.write_text(f"#version 310 es\n#define BLOCK_DIM {block}\n#define HAS_ALPHA {alpha}\n#define IS_NORMALMAP 0\n" + src)
            r = subprocess.run([exe, "--core", core, str(f)], capture_output=True, text=True)
            out = r.stdout + r.stderr
            if "--verbose" in sys.argv: print(out)
            def g(p, grp=1):
                m = re.search(p, out); return m.group(grp) if m else "?"
            regs = g(r"Work registers:\s*(\d+)"); occ = g(r"at (\d+%) occupancy")
            spill = g(r"Spill region:\s*(\d+)"); stack = g(r"Stack size:\s*(\d+)")
            m = re.search(r"Total instruction cycles:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
            l = re.search(r"Longest path cycles:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)", out)
            print(f"{block}x{block} alpha={alpha}: regs {regs} ({occ} occ), stack {stack} B (spill {spill} B), "
                  f"total A/LS/T {'/'.join(m.groups()) if m else '?'}, longest A/LS/T {'/'.join(l.groups()[:3]) if l else '?'} bound {l.group(4) if l else '?'}")
            if "ERROR" in out: print(out)
